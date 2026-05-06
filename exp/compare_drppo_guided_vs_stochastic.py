#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

try:  # Isaac Gym must be imported before torch in this process.
    import isaacgym  # noqa: F401
except ImportError:
    isaacgym = None

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.dppo_frame_rl.data import build_prompt_entry_pools, sample_prompt_batch
from train.dppo_frame_rl.logging import merge_metrics, summarize_episodes
from train.dppo_frame_rl.reward import build_frame_reward_batch
from train.drppo_rl import DRPPORuntime, FrameReverseNoisePolicy


T2M_KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]
CHAIN_COLORS = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]
SMPL_2_MUJOCO = [
    0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9,
    12, 15, 13, 16, 18, 20, 22, 14, 17, 19, 21, 23,
]
MUJOCO_TO_SMPL = np.argsort(np.asarray(SMPL_2_MUJOCO, dtype=np.int64))
ISAAC_TO_HML_MAT = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare vanilla stochastic reverse sampling against DRPPO-guided reverse noise on one prompt batch."
    )
    parser.set_defaults(fixed_initial_noise=True)

    ##############################################
    # Core inputs.
    ##############################################
    parser.add_argument("--drppo_path", required=True, type=str)
    parser.add_argument("--model_path", default="", type=str)
    parser.add_argument("--data_root", default="", type=str)
    parser.add_argument("--run_name", required=True, type=str)
    parser.add_argument("--output_root", default=str(SCRIPT_DIR / "output"), type=str)
    parser.add_argument("--device_id", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)

    ##############################################
    # Batch sampling config.
    ##############################################
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--prompt_batch_size", default=32, type=int)
    parser.add_argument("--train_sampling_mode", choices=["uniform", "failure_only", "mixed"], default=None)
    parser.add_argument("--failure_cases_file", default="", type=str)
    parser.add_argument("--success_sampling_weight", default=None, type=float)
    parser.add_argument("--failure_sampling_weight", default=None, type=float)

    ##############################################
    # Optional overrides for reward / rollout config.
    ##############################################
    parser.set_defaults(reward_norm=None, deterministic_policy=False)
    parser.add_argument("--guidance_param", default=None, type=float)
    parser.add_argument("--max_motion_frames", default=None, type=int)
    parser.add_argument("--frame_gamma", default=None, type=float)
    parser.add_argument("--frame_lambda", default=None, type=float)
    parser.add_argument("--dense_reward_weight", default=None, type=float)
    parser.add_argument("--reward_norm", dest="reward_norm", action="store_true")
    parser.add_argument("--pose_reward_weight", default=None, type=float)
    parser.add_argument("--pose_reward_alpha", default=None, type=float)
    parser.add_argument("--velocity_reward_weight", default=None, type=float)
    parser.add_argument("--velocity_reward_alpha", default=None, type=float)
    parser.add_argument("--success_bonus", default=None, type=float)
    parser.add_argument("--fail_penalty", default=None, type=float)
    parser.add_argument("--phc_num_envs", default=None, type=int)
    parser.add_argument("--phc_max_steps", default=None, type=int)
    parser.add_argument(
        "--phc_actor_ckpt",
        default=None,
        type=str,
    )

    ##############################################
    # Sampling controls.
    ##############################################
    parser.add_argument("--deterministic_policy", action="store_true")
    parser.add_argument("--fixed_initial_noise", action="store_true")
    parser.add_argument("--no_fixed_initial_noise", dest="fixed_initial_noise", action="store_false")
    parser.add_argument("--fixed_reverse_seed", action="store_true")
    parser.add_argument("--render_sample_index", default=0, type=int)
    parser.add_argument("--fps", default=30, type=int)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _unwrap_config_value(value):
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def load_drppo_checkpoint(path: Path) -> Tuple[Dict, Dict]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected DRPPO checkpoint payload type: {type(payload)!r}")
    ckpt_args = payload.get("args", {})
    if not isinstance(ckpt_args, dict):
        raise ValueError("Checkpoint args must be a dict.")
    return payload, ckpt_args


def resolve_checkpoint_path(path_like: str) -> Path:
    path = Path(path_like).expanduser().resolve()
    if path.is_file():
        return path
    if path.is_dir():
        best_path = path / "best.pt"
        if best_path.is_file():
            return best_path
        latest_path = path / "latest.pt"
        if latest_path.is_file():
            return latest_path
        raise FileNotFoundError(f"No best.pt or latest.pt found under checkpoint directory: {path}")
    raise FileNotFoundError(f"Checkpoint path not found: {path}")


def resolve_arg(cli_value, ckpt_args: Dict, key: str, default):
    if cli_value is not None and cli_value != "":
        return cli_value
    return ckpt_args.get(key, default)


def build_runtime_args(args: argparse.Namespace, ckpt_args: Dict) -> SimpleNamespace:
    model_path = resolve_arg(args.model_path, ckpt_args, "model_path", "")
    data_root = resolve_arg(args.data_root, ckpt_args, "data_root", "")
    if not model_path:
        raise ValueError("--model_path was not provided and could not be resolved from the checkpoint args.")
    if not data_root:
        raise ValueError("--data_root was not provided and could not be resolved from the checkpoint args.")

    phc_actor_ckpt = resolve_arg(
        args.phc_actor_ckpt,
        ckpt_args,
        "phc_actor_ckpt",
        "/home/gxy/hay-thesis/PHC/output/HumanoidIm/phc_kp_mcp_iccv/Humanoid.pth",
    )

    return SimpleNamespace(
        device_id=int(args.device_id),
        max_motion_frames=int(resolve_arg(args.max_motion_frames, ckpt_args, "max_motion_frames", 196)),
        model_path=str(model_path),
        guidance_param=float(resolve_arg(args.guidance_param, ckpt_args, "guidance_param", 2.5)),
        deterministic_denoising=False,
        stochastic_first_k_steps=0,
        data_root=str(data_root),
        phc_num_envs=int(resolve_arg(args.phc_num_envs, ckpt_args, "phc_num_envs", 16)),
        phc_max_steps=int(resolve_arg(args.phc_max_steps, ckpt_args, "phc_max_steps", 420)),
        phc_actor_ckpt=str(phc_actor_ckpt),
        dense_reward_weight=float(resolve_arg(args.dense_reward_weight, ckpt_args, "dense_reward_weight", 1.0)),
        reward_norm=bool(resolve_arg(args.reward_norm, ckpt_args, "reward_norm", False)),
        pose_reward_weight=float(resolve_arg(args.pose_reward_weight, ckpt_args, "pose_reward_weight", 0.0)),
        pose_reward_alpha=float(resolve_arg(args.pose_reward_alpha, ckpt_args, "pose_reward_alpha", 1.0)),
        velocity_reward_weight=float(resolve_arg(args.velocity_reward_weight, ckpt_args, "velocity_reward_weight", 0.0)),
        velocity_reward_alpha=float(resolve_arg(args.velocity_reward_alpha, ckpt_args, "velocity_reward_alpha", 1.0)),
        success_bonus=float(resolve_arg(args.success_bonus, ckpt_args, "success_bonus", 0.0)),
        fail_penalty=float(resolve_arg(args.fail_penalty, ckpt_args, "fail_penalty", -5.0)),
        frame_gamma=float(resolve_arg(args.frame_gamma, ckpt_args, "frame_gamma", 0.99)),
        frame_lambda=float(resolve_arg(args.frame_lambda, ckpt_args, "frame_lambda", 0.95)),
        num_controlled_reverse_steps=int(resolve_arg(None, ckpt_args, "num_controlled_reverse_steps", 5)),
    )


def build_runtime(args: argparse.Namespace, ckpt_args: Dict) -> DRPPORuntime:
    runtime_args = build_runtime_args(args, ckpt_args)
    return DRPPORuntime(runtime_args)


def load_reverse_noise_policy(payload: Dict, ckpt_args: Dict, runtime: DRPPORuntime) -> nn.Module:
    policy = FrameReverseNoisePolicy(
        text_dim=runtime.actor.clip_dim,
        sigma_embed_dim=int(ckpt_args.get("sigma_embed_dim", 128)),
        hidden_dim=int(ckpt_args.get("reverse_policy_hidden_dim", 256)),
        action_dim=int(runtime.actor.njoints * runtime.actor.nfeats),
        num_diffusion_steps=int(runtime.diffusion.num_timesteps),
        num_layers=int(ckpt_args.get("reverse_policy_num_layers", 3)),
        log_std_init=float(ckpt_args.get("reverse_log_std_init", 0.0)),
    )
    policy.load_state_dict(payload["reverse_noise_policy"])
    policy.to(runtime.device)
    policy.eval()
    return policy


def select_entries(args: argparse.Namespace, ckpt_args: Dict):
    data_root = resolve_arg(args.data_root, ckpt_args, "data_root", "")
    max_motion_frames = int(resolve_arg(args.max_motion_frames, ckpt_args, "max_motion_frames", 196))
    sampling_mode = resolve_arg(args.train_sampling_mode, ckpt_args, "train_sampling_mode", "uniform")
    failure_cases_file = resolve_arg(args.failure_cases_file, ckpt_args, "failure_cases_file", "")
    success_sampling_weight = float(resolve_arg(args.success_sampling_weight, ckpt_args, "success_sampling_weight", 1.0))
    failure_sampling_weight = float(resolve_arg(args.failure_sampling_weight, ckpt_args, "failure_sampling_weight", 1.0))

    pools = build_prompt_entry_pools(
        data_root=str(data_root),
        split=args.split,
        max_motion_frames=max_motion_frames,
        failure_cases_file=failure_cases_file or None,
    )
    rng = random.Random(args.seed)
    return sample_prompt_batch(
        pools=pools,
        mode=sampling_mode,
        batch_size=int(args.prompt_batch_size),
        rng=rng,
        success_sampling_weight=success_sampling_weight,
        failure_sampling_weight=failure_sampling_weight,
    )


def sample_initial_noise(runtime: DRPPORuntime, batch_size: int, seed: int) -> torch.Tensor:
    shape = (
        batch_size,
        runtime.actor.njoints,
        runtime.actor.nfeats,
        runtime.max_motion_frames,
    )
    devices = [] if runtime.device.index is None else [runtime.device.index]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        return torch.randn(shape, device=runtime.device)


def tracking_motion_to_plot_coords(motion: np.ndarray) -> np.ndarray:
    motion = np.asarray(motion, dtype=np.float32)
    if motion.ndim != 3:
        raise ValueError(f"Expected motion with shape [T, J, 3], got {motion.shape}")
    if motion.shape[1] >= 24:
        motion = motion[:, :24, :]
        motion = motion[:, MUJOCO_TO_SMPL, :]
    motion22 = motion[:, :22, :]
    hml = np.matmul(motion22, ISAAC_TO_HML_MAT)
    hml = hml.copy()
    hml[:, :, 1] -= float(hml[:, :, 1].min())
    return hml[:, :, [0, 2, 1]]


def generated_motion_to_plot_coords(motion22_30hz: np.ndarray) -> np.ndarray:
    motion = np.asarray(motion22_30hz, dtype=np.float32).copy()
    if motion.ndim != 3 or motion.shape[1] < 22:
        raise ValueError(f"Expected generated motion with shape [T, 22, 3], got {motion.shape}")
    motion = motion[:, :22, :]
    motion[:, :, 1] -= float(motion[:, :, 1].min())
    return motion[:, :, [0, 2, 1]]


def pad_motion_to_length(motion: np.ndarray, target_length: int) -> np.ndarray:
    if motion.shape[0] >= target_length:
        return motion
    pad = np.repeat(motion[-1:, :, :], target_length - motion.shape[0], axis=0)
    return np.concatenate([motion, pad], axis=0)


def compute_plot_limits(motions: Sequence[np.ndarray]) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    stacked = np.concatenate(motions, axis=1)
    mins = stacked.min(axis=(0, 1))
    maxs = stacked.max(axis=(0, 1))
    spans = np.maximum(maxs - mins, 1e-3)
    pad = np.maximum(spans * 0.08, 0.05)
    xlim = (float(mins[0] - pad[0]), float(maxs[0] + pad[0]))
    ylim = (float(mins[1] - pad[1]), float(maxs[1] + pad[1]))
    zlim = (0.0, float(maxs[2] + pad[2]))
    return xlim, ylim, zlim


def draw_motion_panel(
    ax,
    motion: np.ndarray,
    frame_idx: int,
    title: str,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    zlim: Tuple[float, float],
) -> None:
    floor = [
        [xlim[0], ylim[0], 0.0],
        [xlim[0], ylim[1], 0.0],
        [xlim[1], ylim[1], 0.0],
        [xlim[1], ylim[0], 0.0],
    ]
    ax.cla()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], max(zlim[1] - zlim[0], 1e-3)))
    ax.set_title(title, fontsize=10, pad=10)
    ax.add_collection3d(Poly3DCollection([floor], facecolor=(0.75, 0.75, 0.75, 0.18), edgecolor="none"))

    traj = motion[: frame_idx + 1, 0]
    ax.plot(traj[:, 0], traj[:, 1], np.zeros_like(traj[:, 0]), color="#777777", linewidth=1.2, alpha=0.9)

    joints = motion[frame_idx]
    for chain, color in zip(T2M_KINEMATIC_CHAIN, CHAIN_COLORS):
        ax.plot(joints[chain, 0], joints[chain, 1], joints[chain, 2], color=color, linewidth=3.0)
    ax.set_axis_off()


def render_pair_video(
    save_path: Path,
    generated_motion: np.ndarray,
    tracked_motion: np.ndarray,
    left_title: str,
    right_title: str,
    fps: int,
) -> None:
    max_frames = max(generated_motion.shape[0], tracked_motion.shape[0])
    generated_motion = pad_motion_to_length(generated_motion, max_frames)
    tracked_motion = pad_motion_to_length(tracked_motion, max_frames)
    xlim, ylim, zlim = compute_plot_limits([generated_motion, tracked_motion])

    fig = plt.figure(figsize=(12, 6), dpi=120)
    ax_left = fig.add_subplot(121, projection="3d")
    ax_right = fig.add_subplot(122, projection="3d")
    writer = imageio.get_writer(str(save_path), fps=fps, codec="libx264", quality=8)

    ##############################################
    # Render one paired frame at a time so each mp4
    # directly compares generated motion and PHC tracking.
    ##############################################
    for frame_idx in range(max_frames):
        draw_motion_panel(ax_left, generated_motion, frame_idx, left_title, xlim, ylim, zlim)
        draw_motion_panel(ax_right, tracked_motion, frame_idx, right_title, xlim, ylim, zlim)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        writer.append_data(frame)

    writer.close()
    plt.close(fig)


def maybe_render_case_video(
    save_path: Path,
    entry,
    mode_name: str,
    summary: Dict[str, float],
    generated_motion_30hz: np.ndarray,
    tracked_motion: np.ndarray,
    fps: int,
) -> None:
    left_title = (
        f"{mode_name}\n"
        f"{entry.db_key}:{int(entry.caption_idx)} | dense={summary['dense_reward_mean']:.3f}\n"
        f"pose={summary['imitation_pose_reward_mean']:.3f} | total={summary['undiscounted_sequence_reward_mean']:.3f}"
    )
    right_title = (
        f"PHC Tracking\n"
        f"success={summary['success_rate']:.3f} | return={summary['phc_return_mean']:.3f}\n"
        f"exec_ratio={summary['exec_ratio_mean']:.3f}"
    )
    render_pair_video(
        save_path=save_path,
        generated_motion=generated_motion_to_plot_coords(generated_motion_30hz),
        tracked_motion=tracking_motion_to_plot_coords(tracked_motion),
        left_title=left_title,
        right_title=right_title,
        fps=fps,
    )


def _evaluate_final_sample(
    runtime: DRPPORuntime,
    entries: Sequence,
    final_sample: torch.Tensor,
    lengths_20fps: torch.Tensor,
    mode_name: str,
) -> Dict:
    generated_joints = torch.from_numpy(runtime._sample_to_joint_batch(final_sample)).to(runtime.device)
    gt_joints = runtime._load_gt_joint_batch(entries)

    ref_motion_batch, lengths_30hz = runtime._sample_to_reference_batch(final_sample, lengths_20fps)
    episodes = runtime._evaluate_reference_batch(
        ref_motion_batch=ref_motion_batch,
        lengths_30hz=lengths_30hz,
        entries=entries,
    )
    reward_batch = build_frame_reward_batch(
        episodes=episodes,
        max_motion_frames=runtime.max_motion_frames,
        gamma_frame=runtime.args.frame_gamma,
        dense_reward_weight=runtime.args.dense_reward_weight,
        reward_norm=runtime.args.reward_norm,
        success_bonus=runtime.args.success_bonus,
        fail_penalty=runtime.args.fail_penalty,
        device=runtime.device,
        generated_joints_20fps=generated_joints,
        gt_joints_20fps=gt_joints,
        pose_reward_weight=getattr(runtime.args, "pose_reward_weight", 0.0),
        pose_reward_alpha=getattr(runtime.args, "pose_reward_alpha", 1.0),
        pose_reward_joint_ids=runtime._imitation_joint_ids,
        velocity_reward_weight=getattr(runtime.args, "velocity_reward_weight", 0.0),
        velocity_reward_alpha=getattr(runtime.args, "velocity_reward_alpha", 1.0),
        velocity_reward_joint_ids=runtime._imitation_joint_ids,
    )
    summary = merge_metrics(
        summarize_episodes(episodes),
        {
            "dense_reward_mean": reward_batch.dense_reward_mean,
            "imitation_pose_reward_mean": reward_batch.imitation_pose_reward_mean,
            "pose_mse_mean": reward_batch.pose_mse_mean,
            "imitation_velocity_reward_mean": reward_batch.imitation_velocity_reward_mean,
            "terminal_reward_mean": reward_batch.terminal_reward_mean,
            "undiscounted_sequence_reward_mean": reward_batch.undiscounted_sequence_reward_mean,
            "exec_ratio_mean": reward_batch.exec_ratio_mean,
        },
    )
    generated_motion_30hz_list = []
    tracked_motion_list = []
    for sample_idx, entry in enumerate(entries):
        generated_motion_20fps = generated_joints[sample_idx, : entry.length_20fps].detach().cpu().numpy().astype(np.float32)
        generated_motion_30hz_list.append(runtime._fps_20_to_30(generated_motion_20fps))
        tracked_motion_list.append(np.asarray(episodes[sample_idx]["pred_motion"], dtype=np.float32))
    return {
        "mode": mode_name,
        "final_sample": final_sample,
        "generated_joints": generated_joints,
        "generated_motion_30hz_list": generated_motion_30hz_list,
        "tracked_motion_list": tracked_motion_list,
        "episodes": episodes,
        "reward_batch": reward_batch,
        "summary": summary,
    }


def sample_full_stochastic(
    runtime: DRPPORuntime,
    entries: Sequence,
    initial_noise: torch.Tensor,
    reverse_seed: int,
) -> Dict:
    texts = [entry.caption for entry in entries]
    lengths_20fps = torch.tensor([entry.length_20fps for entry in entries], dtype=torch.long, device=runtime.device)
    text_embeds = runtime.actor.encode_text(texts)
    model_kwargs = runtime._build_model_kwargs(
        texts=texts,
        lengths_20fps=lengths_20fps,
        max_motion_frames=runtime.max_motion_frames,
        text_embeds=text_embeds,
    )
    final_sample = runtime._sample_clean_motion(
        x_t=initial_noise,
        model_kwargs=model_kwargs,
        deterministic_denoising=False,
        reverse_noise_seed=reverse_seed,
    )
    return _evaluate_final_sample(runtime, entries, final_sample, lengths_20fps, mode_name="baseline_stochastic")


def sample_guided_controlled_steps(
    runtime: DRPPORuntime,
    entries: Sequence,
    reverse_noise_policy: nn.Module,
    initial_noise: torch.Tensor,
    policy_deterministic: bool,
    sampling_seed: int,
) -> Dict:
    texts = [entry.caption for entry in entries]
    lengths_20fps = torch.tensor([entry.length_20fps for entry in entries], dtype=torch.long, device=runtime.device)
    devices = [] if runtime.device.index is None else [runtime.device.index]

    def _sample_once() -> torch.Tensor:
        with torch.no_grad():
            text_embeds = runtime.actor.encode_text(texts)
            model_kwargs = runtime._build_model_kwargs(
                texts=texts,
                lengths_20fps=lengths_20fps,
                max_motion_frames=runtime.max_motion_frames,
                text_embeds=text_embeds,
            )
            x_t = initial_noise.clone()
            controlled_timesteps = runtime._controlled_timesteps_tensor()
            controlled_set = {int(step.item()) for step in controlled_timesteps}

            ##############################################
            # Compare against full stochastic baseline:
            # - controlled steps: policy-provided reverse noise
            # - uncontrolled steps: deterministic mean updates
            ##############################################
            for step in reversed(range(runtime.diffusion.num_timesteps)):
                t = torch.full((len(texts),), step, dtype=torch.long, device=runtime.device)
                out = runtime.diffusion.p_mean_variance(
                    runtime.sampling_actor,
                    x_t,
                    t,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                )
                if step in controlled_set:
                    x_t_frames = x_t.squeeze(2).permute(0, 2, 1).contiguous()
                    eps_actions, _ = reverse_noise_policy.sample(
                        x_t_frames=x_t_frames,
                        text_embeds=text_embeds,
                        step_ids=step,
                        deterministic=policy_deterministic,
                    )
                    eps_tensor = eps_actions.permute(0, 2, 1).unsqueeze(2).contiguous()
                    nonzero_mask = (t != 0).float().view(-1, *([1] * (x_t.dim() - 1)))
                    step_std = torch.exp(0.5 * out["log_variance"])
                    x_prev = out["mean"] + nonzero_mask * step_std * eps_tensor
                else:
                    x_prev = out["mean"]
                x_t = x_prev
        return x_t.detach()

    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(sampling_seed))
        final_sample = _sample_once()
    return _evaluate_final_sample(runtime, entries, final_sample, lengths_20fps, mode_name="guided_controlled_steps")


def _flatten_valid_joints(joints: torch.Tensor, length_20fps: int) -> torch.Tensor:
    length = max(1, int(length_20fps))
    return joints[:length].reshape(-1)


def compute_motion_difference_rows(
    entries: Sequence,
    baseline: Dict,
    guided: Dict,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    rows: List[Dict[str, object]] = []
    rms_values: List[float] = []
    cosine_values: List[float] = []

    base_joints = baseline["generated_joints"]
    guided_joints = guided["generated_joints"]
    base_episodes = baseline["episodes"]
    guided_episodes = guided["episodes"]
    base_rewards = baseline["reward_batch"].frame_rewards.sum(dim=1).detach().cpu().numpy()
    guided_rewards = guided["reward_batch"].frame_rewards.sum(dim=1).detach().cpu().numpy()

    for idx, entry in enumerate(entries):
        vec_a = _flatten_valid_joints(base_joints[idx], entry.length_20fps)
        vec_b = _flatten_valid_joints(guided_joints[idx], entry.length_20fps)
        diff = vec_a - vec_b
        rms = float(torch.sqrt(torch.mean(diff.pow(2))).item())
        denom = max(float(torch.linalg.norm(vec_a).item() * torch.linalg.norm(vec_b).item()), 1e-8)
        cosine = float(torch.dot(vec_a, vec_b).item() / denom)

        rms_values.append(rms)
        cosine_values.append(cosine)
        rows.append(
            {
                "sample_idx": idx,
                "db_key": entry.db_key,
                "caption_idx": entry.caption_idx,
                "length_20fps": int(entry.length_20fps),
                "caption": entry.caption,
                "baseline_success": bool(base_episodes[idx].get("success", False)),
                "guided_success": bool(guided_episodes[idx].get("success", False)),
                "baseline_phc_return_mean": float(base_episodes[idx].get("return_mean", 0.0)),
                "guided_phc_return_mean": float(guided_episodes[idx].get("return_mean", 0.0)),
                "baseline_total_reward": float(base_rewards[idx]),
                "guided_total_reward": float(guided_rewards[idx]),
                "joints_rms": rms,
                "joints_cosine": cosine,
            }
        )

    summary = {
        "joints_rms_mean": float(np.mean(rms_values)) if rms_values else 0.0,
        "joints_rms_std": float(np.std(rms_values)) if rms_values else 0.0,
        "joints_cosine_mean": float(np.mean(cosine_values)) if cosine_values else 1.0,
        "joints_cosine_std": float(np.std(cosine_values)) if cosine_values else 0.0,
    }
    return rows, summary


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    drppo_path = resolve_checkpoint_path(args.drppo_path)

    payload, ckpt_args = load_drppo_checkpoint(drppo_path)
    runtime = build_runtime(args, ckpt_args)
    policy = load_reverse_noise_policy(payload, ckpt_args, runtime)
    entries, sampling_metrics = select_entries(args, ckpt_args)

    out_dir = Path(args.output_root).expanduser().resolve() / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    initial_seed = int(args.seed)
    guided_initial_seed = int(args.seed) if args.fixed_initial_noise else int(args.seed + 17)
    baseline_reverse_seed = int(args.seed) if args.fixed_reverse_seed else int(args.seed + 1)
    guided_sampling_seed = int(args.seed) if args.fixed_reverse_seed else int(args.seed + 2)
    baseline_initial_noise = sample_initial_noise(runtime, batch_size=len(entries), seed=initial_seed)
    guided_initial_noise = (
        baseline_initial_noise
        if args.fixed_initial_noise
        else sample_initial_noise(runtime, batch_size=len(entries), seed=guided_initial_seed)
    )

    baseline = sample_full_stochastic(
        runtime=runtime,
        entries=entries,
        initial_noise=baseline_initial_noise,
        reverse_seed=baseline_reverse_seed,
    )
    guided = sample_guided_controlled_steps(
        runtime=runtime,
        entries=entries,
        reverse_noise_policy=policy,
        initial_noise=guided_initial_noise,
        policy_deterministic=bool(args.deterministic_policy),
        sampling_seed=guided_sampling_seed,
    )

    rows, diff_summary = compute_motion_difference_rows(entries, baseline, guided)
    write_csv(out_dir / "samples.csv", rows)

    render_sample_index = int(args.render_sample_index)
    if render_sample_index < 0 or render_sample_index >= len(entries):
        raise IndexError(f"--render_sample_index {render_sample_index} out of range for batch size {len(entries)}")
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    baseline_video_path = videos_dir / "baseline_stochastic.mp4"
    guided_video_path = videos_dir / "guided_controlled_steps.mp4"
    maybe_render_case_video(
        save_path=baseline_video_path,
        entry=entries[render_sample_index],
        mode_name="Baseline Stochastic",
        summary=baseline["summary"],
        generated_motion_30hz=baseline["generated_motion_30hz_list"][render_sample_index],
        tracked_motion=baseline["tracked_motion_list"][render_sample_index],
        fps=int(args.fps),
    )
    maybe_render_case_video(
        save_path=guided_video_path,
        entry=entries[render_sample_index],
        mode_name="Guided Controlled Steps",
        summary=guided["summary"],
        generated_motion_30hz=guided["generated_motion_30hz_list"][render_sample_index],
        tracked_motion=guided["tracked_motion_list"][render_sample_index],
        fps=int(args.fps),
    )

    summary = {
        "config": {
            "drppo_path": str(drppo_path),
            "model_path": str(runtime.args.model_path),
            "data_root": str(runtime.args.data_root),
            "split": args.split,
            "prompt_batch_size": len(entries),
            "seed": int(args.seed),
            "baseline_initial_noise_seed": initial_seed,
            "guided_initial_noise_seed": guided_initial_seed,
            "fixed_initial_noise": bool(args.fixed_initial_noise),
            "baseline_reverse_seed": baseline_reverse_seed,
            "guided_sampling_seed": guided_sampling_seed,
            "deterministic_policy": bool(args.deterministic_policy),
            "render_sample_index": render_sample_index,
            "reward_norm": bool(runtime.args.reward_norm),
            "num_controlled_reverse_steps": int(runtime.num_controlled_reverse_steps),
            "controlled_timesteps": list(runtime.controlled_timestep_ids()),
            "diffusion_num_timesteps": int(runtime.diffusion.num_timesteps),
        },
        "sampling_metrics": sampling_metrics,
        "baseline_summary": baseline["summary"],
        "guided_summary": guided["summary"],
        "difference_summary": {
            **diff_summary,
            "delta_success_rate": float(guided["summary"]["success_rate"] - baseline["summary"]["success_rate"]),
            "delta_phc_return_mean": float(
                guided["summary"]["phc_return_mean"] - baseline["summary"]["phc_return_mean"]
            ),
            "delta_total_reward_mean": float(
                guided["summary"]["undiscounted_sequence_reward_mean"]
                - baseline["summary"]["undiscounted_sequence_reward_mean"]
            ),
        },
        "videos": {
            "baseline_stochastic": str(baseline_video_path),
            "guided_controlled_steps": str(guided_video_path),
        },
    }
    save_json(out_dir / "summary.json", summary)

    print(
        "[compare] "
        f"diffusion_num_timesteps={int(runtime.diffusion.num_timesteps)} "
        f"controlled_timesteps={list(runtime.controlled_timestep_ids())}"
    )
    print(
        "[baseline] "
        + " ".join(f"{k}={v:.4f}" for k, v in sorted(baseline["summary"].items()))
    )
    print(
        "[guided] "
        + " ".join(f"{k}={v:.4f}" for k, v in sorted(guided["summary"].items()))
    )
    print(
        "[diff] "
        + " ".join(f"{k}={v:.4f}" for k, v in sorted(summary["difference_summary"].items()))
    )
    print(
        f"[saved] summary={out_dir / 'summary.json'} samples={out_dir / 'samples.csv'} "
        f"baseline_video={baseline_video_path} guided_video={guided_video_path}"
    )

'''
python exp/compare_drppo_guided_vs_stochastic.py \
  --drppo_path /home/gxy/hay-thesis/motion-diffusion-model-phc/save/drppo_data_failure_00013_lr_ \
  --run_name compare_00013_3 \
  --output_root /home/gxy/hay-thesis/motion-diffusion-model-phc/exp/output \
  --prompt_batch_size 4 \
  --phc_num_envs 4 \
  --phc_max_steps 400 \
  --seed 3
'''
if __name__ == "__main__":
    main()
