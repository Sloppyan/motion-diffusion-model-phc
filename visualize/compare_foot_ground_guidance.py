from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation


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
from mdm_core.guidance import FootGroundGuidance, FootGroundGuidanceConfig
from mdm_core.model.cfg_sampler import ClassifierFreeSampleModel
from mdm_core.utils.fixseed import fixseed
from mdm_core.utils.model_util import create_model_and_diffusion, load_model_wo_clip


PROMPT_DEFAULT = "go forward and squat and the kick the air."
MODEL_PATH_DEFAULT = REPO_ROOT / "save" / "humanml_enc_512_50steps" / "model000750000.pt"
OUTPUT_ROOT = REPO_ROOT / "visualize" / "output" / "guidance"
MAX_FRAMES = 196
FPS = 20


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render and compare baseline MDM output against foot-ground guided MDM output.",
    )
    parser.add_argument("--prompt", type=str, default=PROMPT_DEFAULT)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--motion-length", type=float, default=6.0)
    parser.add_argument("--guidance-param", type=float, default=2.5)
    parser.add_argument("--fg-last-steps", type=int, default=10)
    parser.add_argument("--fg-tau", type=float, default=0.005)
    parser.add_argument("--fg-contact-k", type=float, default=50.0)
    parser.add_argument("--fg-lambda-pen", type=float, default=10000.0)
    parser.add_argument("--fg-lambda-float", type=float, default=10000.0)
    parser.add_argument("--fg-lambda-skate", type=float, default=2.0)
    parser.add_argument("--fg-step-size", type=float, default=1e-4)
    parser.add_argument("--fg-grad-clip", type=float, default=0.0)
    parser.add_argument("--view-axis", type=str, default="auto", choices=["auto", "x", "z"])
    parser.add_argument("--below-ground", type=float, default=0.12)
    return parser.parse_args()


def _resolve_device(device_arg: str):
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_arg)


def _slugify(value: str):
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    if not slug:
        slug = "prompt"
    return slug[:80]


def _load_model_args(model_path: Path):
    args_path = model_path.resolve().parent / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"missing args.json next to checkpoint: {args_path}")

    import json

    with args_path.open("r", encoding="utf-8") as handle:
        model_args = argparse.Namespace(**json.load(handle))
    return model_args


def _to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _build_text_only_loader(num_frames: int):
    data = get_dataset_loader(
        name="humanml",
        batch_size=1,
        num_frames=MAX_FRAMES,
        split="test",
        hml_mode="text_only",
    )
    data.dataset.t2m_dataset.fixed_length = num_frames
    return data


def _build_model_kwargs(prompt: str, num_frames: int, device: torch.device, guidance_param: float):
    collate_args = [{"inp": torch.zeros(num_frames), "tokens": None, "lengths": num_frames, "text": prompt}]
    _, model_kwargs = collate(collate_args)
    model_kwargs = _to_device(model_kwargs, device)
    model_kwargs["y"]["scale"] = torch.ones(1, device=device) * guidance_param
    return model_kwargs


def _load_model(model_path: Path, model_args, data, device: torch.device, guidance_param: float):
    model, diffusion = create_model_and_diffusion(model_args, data)
    state_dict = torch.load(model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)
    if guidance_param != 1:
        model = ClassifierFreeSampleModel(model)
    model.to(device)
    model.eval()
    return model, diffusion


def _sample_motion(model, diffusion, data, prompt: str, num_frames: int, seed: int, guidance_param: float, device, denoised_fn=None):
    model_kwargs = _build_model_kwargs(prompt=prompt, num_frames=num_frames, device=device, guidance_param=guidance_param)

    # Keep the seed identical before each pass so both videos share the same
    # random sampling trajectory except for the guidance intervention itself.
    fixseed(seed)
    with torch.no_grad():
        sample = diffusion.p_sample_loop(
            model,
            (1, model.njoints, model.nfeats, MAX_FRAMES),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            denoised_fn=denoised_fn,
        )

    n_joints = 22 if sample.shape[1] == 263 else 21
    sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
    sample = recover_from_ric(sample, n_joints)
    sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)
    return sample.cpu().numpy()[0].transpose(2, 0, 1)[:num_frames]


def _build_guidance(args, data):
    return FootGroundGuidance(
        mean=data.dataset.t2m_dataset.mean,
        std=data.dataset.t2m_dataset.std,
        config=FootGroundGuidanceConfig(
            last_steps=args.fg_last_steps,
            tau=args.fg_tau,
            contact_k=args.fg_contact_k,
            lambda_pen=args.fg_lambda_pen,
            lambda_float=args.fg_lambda_float,
            lambda_skate=args.fg_lambda_skate,
            step_size=args.fg_step_size,
            grad_clip=args.fg_grad_clip,
        ),
    )


def _prepare_motion_for_render(motion):
    data = motion.astype("float32", copy=True) * 1.3
    data[..., 0] -= data[0, 0, 0]
    data[..., 2] -= data[0, 0, 2]
    return data


def _select_side_axis(data, view_axis: str):
    if view_axis in {"x", "z"}:
        return 0 if view_axis == "x" else 2, view_axis

    root_x = data[:, 0, 0]
    root_z = data[:, 0, 2]
    range_x = float(root_x.max() - root_x.min())
    range_z = float(root_z.max() - root_z.min())
    return (2, "z") if range_z >= range_x else (0, "x")


def _style_axis(ax, title: str, x_limits, y_limits):
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_title(title, fontsize=10, pad=10)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def _draw_ground(ax, x_limits, below_ground):
    ax.axhline(0.0, color="#1F1F1F", linewidth=1.6)
    ax.fill_between(
        [x_limits[0], x_limits[1]],
        [0.0, 0.0],
        [-below_ground, -below_ground],
        color="#D9D9D9",
        alpha=0.42,
    )


def _compute_shared_view(motions, args):
    prepared_list = [_prepare_motion_for_render(motion) for motion in motions]
    concat_motion = prepared_list[0] if len(prepared_list) == 1 else np.concatenate(prepared_list, axis=0)
    axis_index, axis_name = _select_side_axis(concat_motion, args.view_axis)

    horizontal = concat_motion[:, :, axis_index]
    vertical = concat_motion[:, :, 1]
    horizontal_min = float(horizontal.min())
    horizontal_max = float(horizontal.max())
    horizontal_pad = max(0.18, 0.08 * (horizontal_max - horizontal_min + 1e-6))
    x_limits = (horizontal_min - horizontal_pad, horizontal_max + horizontal_pad)

    vertical_min = float(vertical.min())
    vertical_max = float(vertical.max())
    below_ground = max(0.06, args.below_ground)
    upper_pad = max(0.2, 0.08 * max(vertical_max, 1e-6))
    y_limits = (min(vertical_min - 0.02, -below_ground), vertical_max + upper_pad)
    return prepared_list, axis_index, axis_name, x_limits, y_limits, below_ground


def _compute_motion_metrics(motion, tau=0.005):
    lowest_heights = motion[:, :, 1].min(axis=1)
    penetration = np.maximum(-(lowest_heights + tau), 0.0) * 1000.0
    floating = np.maximum(lowest_heights - tau, 0.0) * 1000.0

    feet = motion[:, [10, 11], :]
    contact = 1.0 / (1.0 + np.exp(-50.0 * (tau - feet[:, :, 1])))
    contact_pairs = contact[:-1] * contact[1:]
    offsets = feet[1:, :, :][:, :, [0, 2]] - feet[:-1, :, :][:, :, [0, 2]]
    sliding = np.linalg.norm(offsets, axis=2)
    skating = float((contact_pairs * sliding).sum() / max(contact_pairs.sum(), 1e-6) * 1000.0)

    return {
        "min_y_m": float(motion[:, :, 1].min()),
        "max_y_m": float(motion[:, :, 1].max()),
        "penetration_mm": float(penetration.mean()),
        "floating_mm": float(floating.mean()),
        "skating_mm": skating,
    }


def _render_motion(save_path: Path, prepared, title: str, axis_index: int, axis_name: str, x_limits, y_limits, below_ground):
    chains = paramUtil.t2m_kinematic_chain
    colors = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]
    fig = plt.figure(figsize=(5.6, 4.6))
    ax = fig.add_subplot(111)
    plt.tight_layout()

    # Render a world-fixed side projection: the subject stays upright, the
    # ground becomes a literal horizontal reference line, and we keep space
    # both above and below that line so penetration/floating is immediately visible.
    def update(index):
        ax.cla()
        _style_axis(ax, f"{title} | Side View ({axis_name.upper()})", x_limits, y_limits)
        _draw_ground(ax, x_limits, below_ground)

        for chain, color in zip(chains, colors):
            linewidth = 4.0 if len(chain) >= 5 else 2.0
            ax.plot(prepared[index, chain, axis_index], prepared[index, chain, 1], linewidth=linewidth, color=color)

    animation = FuncAnimation(fig, update, frames=prepared.shape[0], interval=1000 / FPS, repeat=False)
    animation.save(str(save_path), fps=FPS)
    plt.close(fig)


def _stack_videos(left_video: Path, right_video: Path, output_video: Path):
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "warning",
        "-i",
        str(left_video),
        "-i",
        str(right_video),
        "-filter_complex",
        "hstack=inputs=2",
        str(output_video),
    ]
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"model checkpoint not found: {args.model_path}")

    device = _resolve_device(args.device)
    model_args = _load_model_args(args.model_path)
    if getattr(model_args, "cond_mask_prob", None) == 0:
        args.guidance_param = 1

    num_frames = min(MAX_FRAMES, int(args.motion_length * FPS))
    data = _build_text_only_loader(num_frames=num_frames)
    model, diffusion = _load_model(
        model_path=args.model_path,
        model_args=model_args,
        data=data,
        device=device,
        guidance_param=args.guidance_param,
    )
    guidance = _build_guidance(args, data)

    baseline_motion = _sample_motion(
        model=model,
        diffusion=diffusion,
        data=data,
        prompt=args.prompt,
        num_frames=num_frames,
        seed=args.seed,
        guidance_param=args.guidance_param,
        device=device,
        denoised_fn=None,
    )
    guided_motion = _sample_motion(
        model=model,
        diffusion=diffusion,
        data=data,
        prompt=args.prompt,
        num_frames=num_frames,
        seed=args.seed,
        guidance_param=args.guidance_param,
        device=device,
        denoised_fn=guidance,
    )
    prepared_list, axis_index, axis_name, x_limits, y_limits, below_ground = _compute_shared_view(
        [baseline_motion, guided_motion],
        args,
    )

    output_dir = args.output_dir / f"{_slugify(args.prompt)}_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    mdm_video = output_dir / "mdm.mp4"
    guided_video = output_dir / "mdm_guided.mp4"
    compare_video = output_dir / "mdm_vs_guided.mp4"
    prompt_file = output_dir / "prompt.txt"
    metrics_file = output_dir / "metrics.json"

    metrics = {
        "prompt": args.prompt,
        "baseline": _compute_motion_metrics(baseline_motion),
        "guided": _compute_motion_metrics(guided_motion),
    }

    _render_motion(mdm_video, prepared_list[0], "MDM", axis_index, axis_name, x_limits, y_limits, below_ground)
    _render_motion(guided_video, prepared_list[1], "MDM + Foot-Ground Guidance", axis_index, axis_name, x_limits, y_limits, below_ground)
    _stack_videos(mdm_video, guided_video, compare_video)
    prompt_file.write_text(args.prompt + "\n", encoding="utf-8")
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"MDM video: {mdm_video}")
    print(f"Guided video: {guided_video}")
    print(f"Comparison video: {compare_video}")
    print(f"Metrics file: {metrics_file}")
    print("Baseline metrics:", metrics["baseline"])
    print("Guided metrics:", metrics["guided"])

"""
python /home/gxy/hay-thesis/motion-diffusion-model-phc/visualize/compare_foot_ground_guidance.py \
--prompt "walk forward for 3 steps and jump forward for 1 step"
"""
if __name__ == "__main__":
    main()
