from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
PULSE_ROOT = REPO_ROOT.parent / "PULSE"
PULSE_PHC_ROOT = PULSE_ROOT / "phc"
ISAACGYM_PYTHON = REPO_ROOT.parent / "isaacgym" / "python"
OUTPUT_ROOT = REPO_ROOT / "visualize" / "output"

for candidate in (ISAACGYM_PYTHON, REPO_ROOT, REPO_ROOT / "src", PULSE_ROOT, PULSE_PHC_ROOT):
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from mdm_core import activate_import_context

activate_import_context(REPO_ROOT)
os.chdir(REPO_ROOT)

from easydict import EasyDict
from isaacgym import gymapi, gymutil
import torch
from mdm_core.data_loaders.get_data import get_dataset_loader
from mdm_core.data_loaders.tensors import collate
from mdm_core.model.cfg_sampler import ClassifierFreeSampleModel
from mdm_core.model.mdm_hidden_probe import MDMHiddenProbe
from mdm_core.model.repa_projector import LayerWeightedFusion, RepaProjector
from mdm_core.utils.fixseed import fixseed
from mdm_core.utils.model_util import create_model_and_diffusion, load_model_wo_clip
from phc.utils.flags import flags
from phc.utils.parse_task import parse_task


DEFAULT_STAGE2_MODEL = REPO_ROOT / "save" / "stage2_repa" / "stage2_repa_run_002" / "best_model.pt"
DEFAULT_STAGE1_BRIDGE = REPO_ROOT / "save" / "stage1_repa" / "stage1_repa_run_001" / "best.pt"
DEFAULT_PULSE_DECODER = PULSE_ROOT / "output" / "HumanoidIm" / "pulse_vae_iclr" / "Humanoid.pth"
DEFAULT_PULSE_MOTION = PULSE_ROOT / "sample_data" / "amass_isaac_standing_upright_slim.pkl"
NUM_FRAMES = 196
SOURCE_FPS = 20
TARGET_FPS = 30


@contextmanager
def _temp_cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bridge a stage2 MDM checkpoint into the PULSE decoder and play the latent sequence in Isaac Gym.",
    )
    parser.add_argument("--prompt", required=True, type=str)
    parser.add_argument("--exp_name", required=True, type=str)
    parser.add_argument("--stage2-model-path", default=DEFAULT_STAGE2_MODEL, type=Path)
    parser.add_argument("--stage1-bridge-path", default=DEFAULT_STAGE1_BRIDGE, type=Path)
    parser.add_argument("--pulse-decoder-ckpt", default=DEFAULT_PULSE_DECODER, type=Path)
    parser.add_argument("--pulse-motion-file", default=DEFAULT_PULSE_MOTION, type=Path)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--guidance-param", default=2.5, type=float)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--hold-steps", default=30, type=int)
    parser.add_argument("--max-steps", default=0, type=int)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-virtual-display", action="store_true")
    return parser.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device(device_arg)


def _load_model_args(model_path: Path):
    args_path = model_path.resolve().parent / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"missing args.json next to checkpoint: {args_path}")
    with args_path.open("r", encoding="utf-8") as handle:
        return argparse.Namespace(**json.load(handle))


def _load_stage1_bridge(checkpoint_path: Path, hidden_dim: int, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    stage1_args = checkpoint.get("args", {})
    hidden_layers = list(stage1_args.get("hidden_layers", [6, 7, 8]))
    projector_hidden_dim = int(stage1_args.get("projector_hidden_dim", 512))
    output_dim = int(checkpoint["projector"]["net.4.weight"].shape[0])

    fusion = LayerWeightedFusion(num_layers=len(hidden_layers)).to(device)
    projector = RepaProjector(
        hidden_dim=hidden_dim,
        projector_hidden_dim=projector_hidden_dim,
        output_dim=output_dim,
    ).to(device)
    fusion.load_state_dict(checkpoint["fusion"])
    projector.load_state_dict(checkpoint["projector"])
    fusion.eval()
    projector.eval()
    return fusion, projector, hidden_layers


def _build_text_only_loader():
    data = get_dataset_loader(
        name="humanml",
        batch_size=1,
        num_frames=NUM_FRAMES,
        split="test",
        hml_mode="text_only",
    )
    data.dataset.t2m_dataset.fixed_length = NUM_FRAMES
    return data


def _to_device(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _build_model_kwargs(prompt: str, guidance_param: float, device: torch.device):
    collate_args = [{"inp": torch.zeros(NUM_FRAMES), "tokens": None, "lengths": NUM_FRAMES, "text": prompt}]
    _, model_kwargs = collate(collate_args)
    model_kwargs = _to_device(model_kwargs, device)
    if guidance_param != 1.0:
        model_kwargs["y"]["scale"] = torch.ones(1, device=device) * float(guidance_param)
    return model_kwargs


def _resample_latents(latents: np.ndarray, source_fps: int, target_fps: int) -> np.ndarray:
    if source_fps == target_fps:
        return latents.copy()
    if latents.shape[0] <= 1:
        return np.repeat(latents, max(1, int(round(target_fps / max(source_fps, 1)))), axis=0)

    target_len = max(1, int(round(latents.shape[0] * float(target_fps) / float(source_fps))))
    old_t = np.arange(latents.shape[0], dtype=np.float32)
    new_t = np.linspace(0, latents.shape[0] - 1, target_len, dtype=np.float32)
    out = np.empty((target_len, latents.shape[1]), dtype=np.float32)
    for dim in range(latents.shape[1]):
        out[:, dim] = np.interp(new_t, old_t, latents[:, dim])
    return out


def _generate_pulse_latent_sequence(args, device: torch.device, output_dir: Path):
    model_args = _load_model_args(args.stage2_model_path)
    data = _build_text_only_loader()
    model_kwargs = _build_model_kwargs(args.prompt, args.guidance_param, device=device)

    raw_model, diffusion = create_model_and_diffusion(model_args, data)
    raw_model.to(device)
    raw_model.eval()

    state_dict = torch.load(args.stage2_model_path, map_location="cpu")
    load_model_wo_clip(raw_model, state_dict)

    sample_model = raw_model
    if args.guidance_param != 1.0:
        sample_model = ClassifierFreeSampleModel(raw_model)
        sample_model.to(device)
        sample_model.eval()

    fusion, projector, hidden_layers = _load_stage1_bridge(
        checkpoint_path=args.stage1_bridge_path,
        hidden_dim=raw_model.latent_dim,
        device=device,
    )
    layer_indices = [layer - 1 for layer in hidden_layers]

    # -------------------------------------------------------
    # Sample a text-conditioned HumanML motion once, then run
    # a deterministic t=0 probe pass to extract frame-wise MDM
    # hidden states for the stage1 fusion/projector bridge.
    # -------------------------------------------------------
    fixseed(args.seed)
    with torch.no_grad():
        sampled = diffusion.p_sample_loop(
            sample_model,
            (1, raw_model.njoints, raw_model.nfeats, NUM_FRAMES),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,
            init_image=None,
            progress=False,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        with MDMHiddenProbe(raw_model, layer_indices=layer_indices) as probe:
            timesteps = torch.zeros((1,), dtype=torch.long, device=device)
            hidden_states = probe.extract(sampled, timesteps, model_kwargs["y"])
            fused, weights = fusion(hidden_states)
            latent_20fps = projector(fused)[0].detach().cpu().numpy().astype(np.float32)

    latent_30fps = _resample_latents(latent_20fps, source_fps=SOURCE_FPS, target_fps=TARGET_FPS)

    np.save(output_dir / "latent_20fps.npy", latent_20fps)
    np.save(output_dir / "latent_30fps.npy", latent_30fps)

    summary = {
        "prompt": args.prompt,
        "seed": args.seed,
        "guidance_param": float(args.guidance_param),
        "latent_20fps_shape": list(latent_20fps.shape),
        "latent_30fps_shape": list(latent_30fps.shape),
        "hidden_layers": hidden_layers,
        "fusion_weights": weights.detach().cpu().tolist(),
    }
    return latent_30fps, summary


def _set_pulse_flags(headless: bool, no_virtual_display: bool):
    flags.test = True
    flags.debug = False
    flags.follow = False
    flags.fixed = False
    flags.divide_group = False
    flags.no_collision_check = False
    flags.fixed_path = False
    flags.real_path = False
    flags.show_traj = False
    flags.server_mode = False
    flags.slow = False
    flags.real_traj = False
    flags.im_eval = False
    flags.no_virtual_display = bool(no_virtual_display or headless)
    flags.render_o3d = False


def _build_pulse_env(args, device: torch.device):
    cfg_root = PULSE_ROOT / "phc" / "data" / "cfg"
    cfg = OmegaConf.load(cfg_root / "config.yaml")
    cfg.env = OmegaConf.load(cfg_root / "env" / "env_pulse_amp.yaml")
    cfg.robot = OmegaConf.load(cfg_root / "robot" / "smpl_humanoid.yaml")
    cfg.learning = OmegaConf.load(cfg_root / "learning" / "pulse_z_task.yaml")
    cfg.sim = OmegaConf.load(cfg_root / "sim" / "default_sim.yaml")

    cfg.exp_name = args.exp_name
    cfg.headless = bool(args.headless)
    cfg.test = True
    cfg.train = False
    cfg.no_log = True
    cfg.no_virtual_display = bool(args.no_virtual_display or args.headless)

    cfg.device = "cuda" if device.type == "cuda" else "cpu"
    cfg.device_id = 0 if device.index is None else int(device.index)
    cfg.rl_device = str(device)

    cfg.env.task = "HumanoidSpeedZ"
    cfg.env.num_envs = 1
    cfg.env.models = [str(args.pulse_decoder_ckpt.resolve())]
    cfg.env.motion_file = str(args.pulse_motion_file.resolve())
    cfg.env.stateInit = "Start"
    cfg.robot.real_weight_porpotion_boxes = False

    cfg_train = cfg.learning
    cfg_train.params.seed = int(args.seed)
    cfg_train.params.config.name = args.exp_name

    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.num_client_threads = int(cfg.sim.slices)
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = 4
    sim_params.physx.use_gpu = cfg.sim.pipeline in ["gpu"]
    sim_params.physx.num_subscenes = int(cfg.sim.subscenes)
    sim_params.physx.max_gpu_contact_pairs = 4 * 1024 * 1024 if flags.test and not flags.im_eval else 16 * 1024 * 1024
    sim_params.use_gpu_pipeline = cfg.sim.pipeline in ["gpu"]
    gymutil.parse_sim_config(cfg["sim"], sim_params)

    task_args = EasyDict(
        {
            "task": cfg.env.task,
            "device_id": cfg.device_id,
            "rl_device": cfg.rl_device,
            "physics_engine": gymapi.SIM_PHYSX if not cfg.sim.use_flex else gymapi.SIM_FLEX,
            "headless": cfg.headless,
            "device": cfg.device,
        }
    )

    with _temp_cwd(PULSE_ROOT):
        task, env = parse_task(task_args, cfg, cfg_train, sim_params)
    return task, env


def _cleanup_pulse(task):
    viewer = getattr(task, "viewer", None)
    if viewer is not None:
        task.gym.destroy_viewer(viewer)
    virtual_display = getattr(task, "virtual_display", None)
    if virtual_display is not None:
        try:
            virtual_display.stop()
        except Exception:
            pass
    sim = getattr(task, "sim", None)
    if sim is not None:
        task.gym.destroy_sim(sim)


def _play_in_pulse(args, latent_30fps: np.ndarray, output_dir: Path, device: torch.device):
    _set_pulse_flags(headless=args.headless, no_virtual_display=args.no_virtual_display)
    task, env = _build_pulse_env(args=args, device=device)

    summary = {
        "executed_steps": 0,
        "terminated_early": False,
        "reset_step": None,
    }

    try:
        env.reset()
        last_latent = latent_30fps[-1]
        total_steps = latent_30fps.shape[0] + max(0, int(args.hold_steps))
        if args.max_steps > 0:
            total_steps = min(total_steps, int(args.max_steps))

        # -------------------------------------------------------
        # Feed the projected latent sequence into HumanoidSpeedZ.
        # The environment itself supplies self_obs and decoder use.
        # -------------------------------------------------------
        for step_idx in range(total_steps):
            if step_idx < latent_30fps.shape[0]:
                current = latent_30fps[step_idx]
            else:
                current = last_latent

            action = torch.from_numpy(current[None]).to(device=device, dtype=torch.float32)
            env.step(action)

            if not args.headless:
                task.render(False)

            summary["executed_steps"] = step_idx + 1
            if bool(task.reset_buf[0].item()):
                summary["terminated_early"] = True
                summary["reset_step"] = step_idx + 1
                break
    finally:
        _cleanup_pulse(task)

    (output_dir / "pulse_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main():
    args = parse_args()
    args.stage2_model_path = args.stage2_model_path.resolve()
    args.stage1_bridge_path = args.stage1_bridge_path.resolve()
    args.pulse_decoder_ckpt = args.pulse_decoder_ckpt.resolve()
    args.pulse_motion_file = args.pulse_motion_file.resolve()

    for path in (
        args.stage2_model_path,
        args.stage1_bridge_path,
        args.pulse_decoder_ckpt,
        args.pulse_motion_file,
    ):
        if not path.exists():
            raise FileNotFoundError(f"required path not found: {path}")

    device = _resolve_device(args.device)
    output_dir = (OUTPUT_ROOT / args.exp_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    latent_30fps, latent_summary = _generate_pulse_latent_sequence(args=args, device=device, output_dir=output_dir)
    pulse_summary = _play_in_pulse(args=args, latent_30fps=latent_30fps, output_dir=output_dir, device=device)

    result = {
        "prompt": args.prompt,
        "stage2_model_path": str(args.stage2_model_path),
        "stage1_bridge_path": str(args.stage1_bridge_path),
        "pulse_decoder_ckpt": str(args.pulse_decoder_ckpt),
        "pulse_motion_file": str(args.pulse_motion_file),
        "latent": latent_summary,
        "pulse": pulse_summary,
    }
    (output_dir / "bridge_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"output directory: {output_dir}")
    print(f"latent 20fps: {output_dir / 'latent_20fps.npy'}")
    print(f"latent 30fps: {output_dir / 'latent_30fps.npy'}")
    print(f"bridge summary: {output_dir / 'bridge_summary.json'}")


if __name__ == "__main__":
    main()
