from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from textwrap import wrap

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, REPO_ROOT / "src"):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from mdm_core import activate_import_context

activate_import_context(REPO_ROOT)
os.chdir(REPO_ROOT)

import mdm_core.data_loaders.humanml.utils.paramUtil as paramUtil
from mdm_core.data_loaders.get_data import get_dataset_loader
from mdm_core.data_loaders.humanml.scripts.motion_process import recover_from_ric
from mdm_core.data_loaders.tensors import collate
from mdm_core.model.cfg_sampler import ClassifierFreeSampleModel
from mdm_core.utils.fixseed import fixseed
from mdm_core.utils.model_util import create_model_and_diffusion, load_model_wo_clip


PROMPT_DEFAULT = "squat and walk forward for 3 steps and then kick the air"
BASE_MODEL_PATH = REPO_ROOT / "save" / "humanml_enc_512_50steps" / "model000750000.pt"
STAGE2_MODEL_PATH = REPO_ROOT / "save" / "stage2_repa" / "stage2_repa_run_002" / "best_model.pt"
OUTPUT_ROOT = REPO_ROOT / "visualize" / "output"
CLOSD_ROOT_CANDIDATES = (
    REPO_ROOT.parent / "CLoSD",
    REPO_ROOT.parent / "CLoSD-LMM",
)
NUM_FRAMES = 196
FPS = 20
GUIDANCE_PARAM = 2.5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare text-to-motion generations from the original MDM checkpoint and a stage2 checkpoint.",
    )
    parser.add_argument("--prompt", type=str, default=PROMPT_DEFAULT)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--stage2-model-path", type=Path, default=STAGE2_MODEL_PATH)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=10)
    return parser.parse_args()


def _resolve_device(device_arg: str):
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_arg)


def _load_model_args(model_path: Path):
    args_path = model_path.resolve().parent / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"missing args.json next to checkpoint: {args_path}")
    with args_path.open("r", encoding="utf-8") as handle:
        return argparse.Namespace(**json.load(handle))


def _to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


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


def _build_model_kwargs(prompt: str, device: torch.device):
    collate_args = [{"inp": torch.zeros(NUM_FRAMES), "tokens": None, "lengths": NUM_FRAMES, "text": prompt}]
    _, model_kwargs = collate(collate_args)
    model_kwargs = _to_device(model_kwargs, device)
    model_kwargs["y"]["scale"] = torch.ones(1, device=device) * GUIDANCE_PARAM
    return model_kwargs


def _load_model(model_path: Path, model_args, data, device: torch.device):
    model, diffusion = create_model_and_diffusion(model_args, data)
    state_dict = torch.load(model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)
    model = ClassifierFreeSampleModel(model)
    model.to(device)
    model.eval()
    return model, diffusion


def _sample_motion(model_path: Path, model_args, data, prompt: str, seed: int, device: torch.device):
    model_kwargs = _build_model_kwargs(prompt=prompt, device=device)
    model, diffusion = _load_model(model_path=model_path, model_args=model_args, data=data, device=device)

    # -------------------------------------------------------
    # Use the same seed before each sampling pass so the base
    # model and the stage2 model are compared fairly.
    # -------------------------------------------------------
    fixseed(seed)
    with torch.no_grad():
        sample = diffusion.p_sample_loop(
            model,
            (1, model.njoints, model.nfeats, NUM_FRAMES),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,
            init_image=None,
            progress=False,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

    # -------------------------------------------------------
    # Convert from HumanML vector features back to xyz joints.
    # -------------------------------------------------------
    n_joints = 22 if sample.shape[1] == 263 else 21
    sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
    sample = recover_from_ric(sample, n_joints)
    sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

    motion = sample.cpu().numpy()[0].transpose(2, 0, 1)[:NUM_FRAMES]
    return motion


def _import_physical_metrics():
    for repo_root in CLOSD_ROOT_CANDIDATES:
        if not repo_root.is_dir():
            continue
        repo_root_str = str(repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)
        from closd.diffusion_planner.data_loaders.humanml.utils.metrics import (
            calculate_floating,
            calculate_foot_sliding,
            calculate_penetration,
        )

        return calculate_penetration, calculate_floating, calculate_foot_sliding
    raise FileNotFoundError(
        "Could not find a local CLoSD repo. Expected one of: "
        + ", ".join(str(path) for path in CLOSD_ROOT_CANDIDATES)
    )


def _compute_physical_metrics(motion: np.ndarray):
    calculate_penetration, calculate_floating, calculate_foot_sliding = _import_physical_metrics()
    motions = torch.from_numpy(motion.transpose(1, 2, 0)).float().unsqueeze(0)
    lengths = torch.tensor([motion.shape[0]], dtype=torch.long)
    return {
        "penetration": float(calculate_penetration(motions, lengths)),
        "floating": float(calculate_floating(motions, lengths)),
        "skating": float(calculate_foot_sliding(motions, lengths)),
    }


def _prepare_motion_for_plot(motion: np.ndarray):
    data = motion.astype(np.float32, copy=True) * 1.3
    mins = data.min(axis=0).min(axis=0)
    maxs = data.max(axis=0).max(axis=0)
    data[:, :, 1] -= mins[1]
    traj = data[:, 0, [0, 2]].copy()
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]
    return data, mins, maxs, traj


def _style_axis(ax, radius: float):
    ax.view_init(elev=120, azim=-90)
    ax.set_xlim3d([-radius / 2, radius / 2])
    ax.set_ylim3d([0, radius])
    ax.set_zlim3d([-radius / 3.0, radius * 2 / 3.0])
    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_axis_off()


def _draw_floor(ax, mins, maxs, traj, index: int):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    verts = [[
        [mins[0] - traj[index, 0], 0, mins[2] - traj[index, 1]],
        [mins[0] - traj[index, 0], 0, maxs[2] - traj[index, 1]],
        [maxs[0] - traj[index, 0], 0, maxs[2] - traj[index, 1]],
        [maxs[0] - traj[index, 0], 0, mins[2] - traj[index, 1]],
    ]]
    floor = Poly3DCollection(verts)
    floor.set_facecolor((0.5, 0.5, 0.5, 0.5))
    ax.add_collection3d(floor)


def _render_animation(save_path: Path, motions, panel_titles, prompt: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    prepared = [_prepare_motion_for_plot(motion) for motion in motions]
    skeleton = paramUtil.t2m_kinematic_chain
    colors = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]

    fig = plt.figure(figsize=(3.2 * len(motions), 3.2))
    axes = [fig.add_subplot(1, len(motions), idx + 1, projection="3d") for idx in range(len(motions))]
    fig.suptitle("\n".join(wrap(prompt, 48)), fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    # -------------------------------------------------------
    # Draw all panels in one animation so the comparison video
    # gets a shared prompt title and panel labels.
    # -------------------------------------------------------
    def update(index):
        for axis, title, (data, mins, maxs, traj) in zip(axes, panel_titles, prepared):
            axis.cla()
            _style_axis(axis, radius=3)
            axis.set_title(title, fontsize=10)
            _draw_floor(axis, mins, maxs, traj, index)
            for chain, color in zip(skeleton, colors):
                linewidth = 4.0 if len(chain) >= 5 else 2.0
                axis.plot3D(
                    data[index, chain, 0],
                    data[index, chain, 1],
                    data[index, chain, 2],
                    linewidth=linewidth,
                    color=color,
                )

    anim = FuncAnimation(fig, update, frames=NUM_FRAMES, interval=1000 / FPS, repeat=False)
    anim.save(str(save_path), writer="ffmpeg", fps=FPS)
    plt.close(fig)


def main():
    args = parse_args()
    device = _resolve_device(args.device)
    exp_dir = (OUTPUT_ROOT / args.exp_name).resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)

    stage2_model_path = args.stage2_model_path.resolve()
    if not BASE_MODEL_PATH.is_file():
        raise FileNotFoundError(f"base checkpoint not found: {BASE_MODEL_PATH}")
    if not stage2_model_path.is_file():
        raise FileNotFoundError(f"stage2 checkpoint not found: {stage2_model_path}")

    model_args = _load_model_args(BASE_MODEL_PATH)
    data = _build_text_only_loader()

    # -------------------------------------------------------
    # Generate one text-conditioned sequence from the original
    # checkpoint and one from the stage2 checkpoint.
    # -------------------------------------------------------
    mdm_motion = _sample_motion(
        model_path=BASE_MODEL_PATH,
        model_args=model_args,
        data=data,
        prompt=args.prompt,
        seed=args.seed,
        device=device,
    )
    mine_motion = _sample_motion(
        model_path=stage2_model_path,
        model_args=model_args,
        data=data,
        prompt=args.prompt,
        seed=args.seed,
        device=device,
    )

    mdm_mp4 = exp_dir / "mdm.mp4"
    mine_mp4 = exp_dir / "mine.mp4"
    comparison_mp4 = exp_dir / "comparison.mp4"
    metrics_json = exp_dir / "metrics.json"

    _render_animation(save_path=mdm_mp4, motions=[mdm_motion], panel_titles=["MDM"], prompt=args.prompt)
    _render_animation(save_path=mine_mp4, motions=[mine_motion], panel_titles=["Mine"], prompt=args.prompt)
    _render_animation(
        save_path=comparison_mp4,
        motions=[mdm_motion, mine_motion],
        panel_titles=["MDM", "Mine"],
        prompt=args.prompt,
    )

    metrics = {
        "prompt": args.prompt,
        "seed": args.seed,
        "num_frames": NUM_FRAMES,
        "fps": FPS,
        "mdm": _compute_physical_metrics(mdm_motion),
        "mine": _compute_physical_metrics(mine_motion),
    }
    metrics_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"output directory: {exp_dir}")
    print(f"mdm video: {mdm_mp4}")
    print(f"mine video: {mine_mp4}")
    print(f"comparison video: {comparison_mp4}")
    print(f"metrics: {metrics_json}")

'''
python visualize/compare_stage2_text2motion.py \
  --exp_name stage_2_002 \
  --prompt "a man crawls forward like a zombie and then stands up."
'''
if __name__ == "__main__":
    main()
