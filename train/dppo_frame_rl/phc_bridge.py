import os
import sys
import importlib.util
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

import torch
from easydict import EasyDict
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


class PHCExternalEvalBridge:
    """
    In-process bridge to PHC HumanoidImExternalEval.

    The bridge owns one rl_games player and repeatedly feeds externally
    generated reference motions into the task.
    """

    DEFAULT_ACTOR_CKPT = "/home/gxy/hay-thesis/PHC/output/HumanoidIm/phc_kp_mcp_iccv/Humanoid.pth"

    def __init__(
        self,
        phc_num_envs: int,
        phc_max_steps: int,
        device_id: int = 0,
        exp_name: str = "phc_kp_mcp_iccv",
        actor_ckpt: str = "",
    ):
        self.phc_num_envs = int(phc_num_envs)
        self.phc_max_steps = int(phc_max_steps)
        self.device_id = int(device_id)
        self.exp_name = exp_name
        self.actor_ckpt = actor_ckpt

        self._bootstrap_imports()
        self.cfg, self.cfg_train = self._compose_cfg()
        self.runner, self.player = self._build_player()
        self.task = self.player.env.task

    def _bootstrap_imports(self) -> None:
        os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

        hay_root = Path(__file__).resolve().parents[3]
        self._phc_root = hay_root / "PHC"
        self._phc_pkg_root = self._phc_root / "phc"
        self._phc_poselib_root = self._phc_root / "poselib"
        for path in (self._phc_root, self._phc_pkg_root, self._phc_poselib_root):
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)

        poselib_init = self._phc_poselib_root / "poselib" / "__init__.py"
        if "poselib" in sys.modules:
            sys.modules.pop("poselib")
        spec = importlib.util.spec_from_file_location(
            "poselib",
            poselib_init,
            submodule_search_locations=[str(poselib_init.parent)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load poselib package from {poselib_init}")
        poselib_module = importlib.util.module_from_spec(spec)
        sys.modules["poselib"] = poselib_module
        spec.loader.exec_module(poselib_module)

        utils_init = self._phc_pkg_root / "utils" / "__init__.py"
        if "utils" in sys.modules:
            sys.modules.pop("utils")
        utils_spec = importlib.util.spec_from_file_location(
            "utils",
            utils_init,
            submodule_search_locations=[str(utils_init.parent)],
        )
        if utils_spec is None or utils_spec.loader is None:
            raise ImportError(f"Failed to load PHC utils package from {utils_init}")
        utils_module = importlib.util.module_from_spec(utils_spec)
        sys.modules["utils"] = utils_module
        utils_spec.loader.exec_module(utils_module)

        import phc.run_hydra as phc_run_hydra  # pylint: disable=import-outside-toplevel
        from phc.utils.config import set_np_formatting, set_seed  # pylint: disable=import-outside-toplevel
        from phc.utils.flags import flags  # pylint: disable=import-outside-toplevel

        self._phc_run_hydra = phc_run_hydra
        self._set_np_formatting = set_np_formatting
        self._set_seed = set_seed
        self._flags = flags

    def _resolve_phc_path(self, value: str) -> str:
        path = Path(value).expanduser()
        if path.is_absolute():
            return str(path.resolve())
        return str((self._phc_root / path).resolve())

    @contextmanager
    def _phc_cwd(self):
        """
        PHC still relies on several relative asset paths at runtime.
        Keep cwd switching local to bridge calls so the training script
        can continue to use absolute paths everywhere else.
        """
        old_cwd = os.getcwd()
        os.chdir(self._phc_root)
        try:
            yield
        finally:
            os.chdir(old_cwd)

    def _compose_cfg(self):
        # Keep the PHC config family aligned with the HumanoidImMDM command
        # used elsewhere, but switch the task to external eval because this
        # bridge feeds externally generated reference motions into PHC.
        overrides = [
            "learning=im_mcp",
            f"exp_name={self.exp_name}",
            "env=env_im_getup_mcp",
            "env.task=HumanoidImExternalEval",
            "robot=smpl_humanoid",
            "robot.freeze_hand=True",
            "robot.box_body=False",
            "env.z_activation=relu",
            f"env.motion_file={str((self._phc_root / 'sample_data' / 'amass_isaac_standing_upright_slim.pkl').resolve())}",
            f"env.models=['{str((self._phc_root / 'output' / 'HumanoidIm' / 'phc_kp_pnn_iccv' / 'Humanoid.pth').resolve())}']",
            f"env.num_envs={self.phc_num_envs}",
            "env.obs_v=7",
            "epoch=-1",
            "+eval=base",
            "eval.planner.name=external",
            "headless=True",
            "no_virtual_display=True",
            "test=True",
            "no_log=True",
            "has_eval=False",
            f"device_id={self.device_id}",
            f"rl_device=cuda:{self.device_id}",
            "device=cuda",
        ]

        cfg_dir = str((self._phc_pkg_root / "data" / "cfg").resolve())
        with initialize_config_dir(version_base=None, config_dir=cfg_dir, job_name="dppo_frame_phc"):
            cfg_hydra = compose(config_name="config", overrides=overrides)

        cfg = EasyDict(OmegaConf.to_container(cfg_hydra, resolve=True))
        self._set_np_formatting()
        if not os.path.isabs(cfg.output_path):
            cfg.output_path = str((self._phc_root / cfg.output_path).resolve())

        flags = self._flags
        flags.debug = cfg.debug
        flags.follow = cfg.follow
        flags.fixed = False
        flags.divide_group = False
        flags.no_collision_check = False
        flags.fixed_path = False
        flags.real_path = False
        flags.show_traj = True
        flags.server_mode = cfg.server_mode
        flags.slow = False
        flags.real_traj = False
        flags.im_eval = cfg.im_eval
        flags.no_virtual_display = cfg.no_virtual_display
        flags.render_o3d = cfg.render_o3d
        flags.test = cfg.test
        flags.add_proj = cfg.add_proj
        flags.has_eval = cfg.has_eval
        flags.trigger_input = False

        cfg.train = not cfg.test
        self._set_seed(cfg.get("seed", -1), cfg.get("torch_deterministic", False))

        cfg_train = cfg.learning
        cfg_train["params"]["config"]["network_path"] = cfg.output_path
        cfg_train["params"]["config"]["train_dir"] = cfg.output_path
        cfg_train["params"]["config"]["num_actors"] = cfg.env.num_envs

        if self.actor_ckpt:
            load_path = self._resolve_phc_path(self.actor_ckpt)
        elif os.path.exists(self.DEFAULT_ACTOR_CKPT):
            load_path = self.DEFAULT_ACTOR_CKPT
        elif cfg.epoch == -1:
            load_path = os.path.join(
                cfg.output_path,
                cfg_train["params"]["config"]["name"] + ".pth",
            )
        else:
            load_path = ""

        if not load_path:
            raise ValueError("PHC actor checkpoint path is empty.")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"PHC checkpoint not found: {load_path}")

        self._actor_ckpt = load_path
        cfg_train["params"]["load_path"] = load_path
        cfg_train["params"]["load_checkpoint"] = False

        self._phc_run_hydra.cfg = cfg
        self._phc_run_hydra.cfg_train = cfg_train
        return cfg, cfg_train

    def _build_player(self):
        algo_observer = self._phc_run_hydra.RLGPUAlgoObserver()
        runner = self._phc_run_hydra.build_alg_runner(algo_observer)
        runner.load(self.cfg_train)
        runner.reset()
        with self._phc_cwd():
            player = runner.create_player()
        player.restore(self._actor_ckpt)
        player.max_steps = self.phc_max_steps
        if getattr(player, "is_rnn", False):
            player.init_rnn()
        return runner, player

    def evaluate_batch(
        self,
        ref_motion_batch,
        lengths_30hz,
        meta: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        with self._phc_cwd():
            # Queue one external batch into the PHC task.
            self.task.load_reference_batch(ref_motion_batch, lengths_30hz, meta=meta)
            done_indices = []

            # Match the rl_games player setup used by the old PHC runner.
            # The first reset also lets the player infer batch dimensions.
            obs_dict = self.player.env_reset()
            self.player.get_batch_size(obs_dict["obs"], 1)
            if getattr(self.player, "is_rnn", False):
                self.player.init_rnn()

            # Step PHC until every queued sample has finished or we hit the
            # rollout budget. finished envs are immediately recycled by passing
            # their done indices back into env_reset(done_indices).
            for _ in range(self.phc_max_steps):
                obs_dict = self.player.env_reset(done_indices)
                if self.task.is_external_batch_finished():
                    break
                with torch.no_grad():
                    action = self.player.get_action(obs_dict, is_determenistic=True)
                _, _, done, _ = self.player.env_step(self.player.env, action)
                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[:: self.player.num_agents]

                # Keep the player-side recurrent state aligned with recycled envs.
                if getattr(self.player, "is_rnn", False) and len(all_done_indices) > 0:
                    for state in self.player.states:
                        state[:, all_done_indices, :] = 0.0

                done_indices = done_indices[:, 0] if len(done_indices) > 0 else []

            else:
                raise RuntimeError("PHC external evaluation reached max_steps before finishing the batch.")

            # PHC stores per-sample tracking results internally during rollout.
            # finalize_external_batch() flushes that queue back to Python.
            episodes = self.task.finalize_external_batch()
            return sorted(episodes, key=lambda episode: int(episode.get("sample_idx", 0)))

    def close(self) -> None:
        env = getattr(self.player, "env", None)
        if env is not None and hasattr(env, "close"):
            env.close()
        self.task = None
        self.player = None
        self.runner = None
