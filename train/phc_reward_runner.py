import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@contextmanager
def _temp_cwd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


class PHCRewardRunner:
    def __init__(
        self,
        config_path: str,
        actor_ckpt: str,
        num_envs: int,
        max_steps: int,
        headless: bool = True,
        no_virtual_display: bool = True,
    ):
        self.config_path = Path(config_path).expanduser().resolve()
        self.actor_ckpt = Path(actor_ckpt).expanduser().resolve()
        self.num_envs = int(num_envs)
        self.max_steps = int(max_steps)
        self.headless = bool(headless)
        self.no_virtual_display = bool(no_virtual_display)
        if not self.config_path.is_file():
            raise FileNotFoundError(f"PHC config not found: {self.config_path}")
        if not self.actor_ckpt.is_file():
            raise FileNotFoundError(f"PHC actor checkpoint not found: {self.actor_ckpt}")

        self.phc_root = self.config_path.parents[4]
        self.phc_pkg_root = self.phc_root / "phc"
        self.poselib_root = self.phc_root / "poselib"
        self._runner = None
        self._player = None
        self._task = None
        self._build()

    def _ensure_path(self):
        for path in (self.phc_root, self.phc_pkg_root, self.poselib_root):
            path_str = str(path)
            if path_str in sys.path:
                sys.path.remove(path_str)
        for path in (self.phc_root, self.phc_pkg_root, self.poselib_root):
            sys.path.insert(0, str(path))

    def _resolve_phc_path(self, value: str) -> str:
        path = Path(value).expanduser()
        if path.is_absolute():
            return str(path.resolve())
        return str((self.phc_root / path).resolve())

    def _build(self):
        self._ensure_path()
        for module_name in list(sys.modules.keys()):
            if module_name == "utils" or module_name.startswith("utils."):
                del sys.modules[module_name]
        with _temp_cwd(self.phc_root):
            from easydict import EasyDict
            from omegaconf import OmegaConf

            from phc import run_hydra
            from phc.utils.config import set_np_formatting
            from phc.utils.flags import flags

            cfg_hydra = OmegaConf.load(str(self.config_path))
            cfg_hydra.headless = self.headless
            cfg_hydra.no_virtual_display = self.no_virtual_display
            cfg_hydra.test = True
            cfg_hydra.train = False
            cfg_hydra.play = True
            cfg_hydra.no_log = True
            cfg_hydra.im_eval = False
            cfg_hydra.env.task = "HumanoidImExternalEval"
            cfg_hydra.env.num_envs = self.num_envs
            cfg_hydra.env.motion_file = self._resolve_phc_path(cfg_hydra.env.motion_file)
            cfg_hydra.env.models = [self._resolve_phc_path(path) for path in cfg_hydra.env.models]
            cfg_hydra.eval.planner.name = "external"
            cfg_hydra.eval.save_tracking_pkl = False
            cfg_hydra.eval.run_name = "ddpo_external_runtime"
            cfg_hydra.eval.max_total_episodes = None
            cfg_hydra.output_path = self._resolve_phc_path(cfg_hydra.output_path)

            cfg = EasyDict(OmegaConf.to_container(cfg_hydra, resolve=True))
            set_np_formatting()
            (
                flags.debug,
                flags.follow,
                flags.fixed,
                flags.divide_group,
                flags.no_collision_check,
                flags.fixed_path,
                flags.real_path,
                flags.show_traj,
                flags.server_mode,
                flags.slow,
                flags.real_traj,
                flags.im_eval,
                flags.no_virtual_display,
                flags.render_o3d,
            ) = (
                cfg.debug,
                cfg.follow,
                False,
                False,
                False,
                False,
                False,
                True,
                cfg.server_mode,
                False,
                False,
                cfg.im_eval,
                cfg.no_virtual_display,
                cfg.render_o3d,
            )
            flags.test = cfg.test
            flags.add_proj = cfg.add_proj
            flags.has_eval = cfg.has_eval
            flags.trigger_input = False

            cfg_train = cfg.learning
            cfg_train["params"]["config"]["network_path"] = cfg.output_path
            cfg_train["params"]["config"]["train_dir"] = cfg.output_path
            cfg_train["params"]["config"]["num_actors"] = cfg.env.num_envs
            cfg_train["params"]["load_checkpoint"] = False
            cfg_train["params"]["load_path"] = str(self.actor_ckpt)

            run_hydra.cfg = cfg
            run_hydra.cfg_train = cfg_train

            runner = run_hydra.build_alg_runner(run_hydra.RLGPUAlgoObserver())
            runner.load(cfg_train)
            player = runner.create_player()
            player.restore(str(self.actor_ckpt))

        self._runner = runner
        self._player = player
        self._task = player.env.task

    def evaluate_batch(
        self,
        ref_motion_batch: np.ndarray,
        lengths: np.ndarray,
        meta: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        task = self._task
        player = self._player

        task.load_reference_batch(ref_motion_batch=ref_motion_batch, lengths=lengths, meta=meta)
        done_indices = []
        obs_dict = player.env_reset()
        player.get_batch_size(obs_dict["obs"], 1)
        if player.is_rnn:
            player.init_rnn()

        for _ in range(self.max_steps):
            obs_dict = player.env_reset(done_indices)
            if task.is_external_batch_finished():
                break

            action = player.get_action(obs_dict, is_determenistic=True)
            obs_dict, _, done, _ = player.env_step(player.env, action)
            all_done_indices = done.nonzero(as_tuple=False)
            done_indices = all_done_indices[:: player.num_agents]

            if player.is_rnn and len(all_done_indices) > 0:
                for state in player.states:
                    state[:, all_done_indices, :] = 0.0

            done_indices = done_indices[:, 0] if len(done_indices) > 0 else []
        else:
            raise RuntimeError("PHC external evaluation reached max_steps before finishing the batch.")

        episodes = task.finalize_external_batch()
        episodes = sorted(episodes, key=lambda episode: int(episode.get("sample_idx", 0)))
        return episodes
