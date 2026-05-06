#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

try:  # Isaac Gym must be imported before torch in this process.
    import isaacgym  # noqa: F401
except ImportError:
    isaacgym = None

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.dppo_frame_rl.data import build_prompt_entry_pools, sample_prompt_batch
from train.dppo_frame_rl.reward import _align_reward_steps_to_frames, build_frame_reward_batch
from train.drppo_rl import DRPPORuntime, FrameReverseNoisePolicy


class StandardGaussianReverseNoisePolicy(nn.Module):
    """Fallback reverse-noise policy eps ~ N(0, I) for controlled steps."""

    def __init__(self, action_dim: int = 263):
        super().__init__()
        self.action_dim = int(action_dim)

    def dist(
        self,
        x_t_frames: torch.Tensor,
        text_embeds: torch.Tensor,
        step_ids,
    ) -> Independent:
        del text_embeds, step_ids
        mean = torch.zeros_like(x_t_frames)
        std = torch.ones_like(x_t_frames)
        return Independent(Normal(mean, std), 1)

    def sample(
        self,
        x_t_frames: torch.Tensor,
        text_embeds: torch.Tensor,
        step_ids,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.dist(x_t_frames, text_embeds, step_ids)
        actions = dist.mean if deterministic else dist.sample()
        return actions, dist.log_prob(actions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose DRPPO batch reward decomposition on one rollout batch.")

    ##############################################
    # Core runtime / checkpoint inputs.
    ##############################################
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--data_root", required=True, type=str)
    parser.add_argument("--run_name", required=True, type=str)
    parser.add_argument("--output_root", default=str(SCRIPT_DIR / "output"), type=str)
    parser.add_argument("--device_id", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)

    ##############################################
    # Prompt batch sampling config.
    ##############################################
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--max_motion_frames", default=196, type=int)
    parser.add_argument("--prompt_batch_size", default=256, type=int)
    parser.add_argument("--train_sampling_mode", choices=["uniform", "failure_only", "mixed"], default="uniform")
    parser.add_argument("--failure_cases_file", default="", type=str)
    parser.add_argument("--success_sampling_weight", default=1.0, type=float)
    parser.add_argument("--failure_sampling_weight", default=1.0, type=float)

    ##############################################
    # Reverse-noise policy config.
    ##############################################
    parser.add_argument("--noise_source", choices=["gaussian", "drppo"], default="gaussian", type=str)
    parser.add_argument("--drppo_path", default="", type=str)
    parser.add_argument("--deterministic_policy", action="store_true")
    parser.add_argument("--num_controlled_reverse_steps", default=5, type=int)
    parser.add_argument("--sigma_embed_dim", default=128, type=int)
    parser.add_argument("--reverse_policy_hidden_dim", default=256, type=int)
    parser.add_argument("--reverse_policy_num_layers", default=3, type=int)
    parser.add_argument("--reverse_log_std_init", default=0.0, type=float)
    parser.add_argument("--fixed_sampling_seed", action="store_true")
    parser.add_argument("--sampling_seed", default=None, type=int)

    ##############################################
    # Reward / PHC config.
    ##############################################
    parser.add_argument("--guidance_param", default=2.5, type=float)
    parser.add_argument("--frame_gamma", default=0.99, type=float)
    parser.add_argument("--frame_lambda", default=0.9, type=float)
    parser.add_argument("--dense_reward_weight", default=1.0, type=float)
    parser.set_defaults(reward_norm=False)
    parser.add_argument("--reward_norm", dest="reward_norm", action="store_true")
    parser.add_argument("--pose_reward_weight", default=0.0, type=float)
    parser.add_argument("--pose_reward_alpha", default=1.0, type=float)
    parser.add_argument("--velocity_reward_weight", default=0.0, type=float)
    parser.add_argument("--velocity_reward_alpha", default=1.0, type=float)
    parser.add_argument("--success_bonus", default=0.0, type=float)
    parser.add_argument("--fail_penalty", default=-5.0, type=float)
    parser.add_argument("--phc_num_envs", default=16, type=int)
    parser.add_argument("--phc_max_steps", default=420, type=int)
    parser.add_argument(
        "--phc_actor_ckpt",
        default="/home/gxy/hay-thesis/PHC/output/HumanoidIm/phc_kp_mcp_iccv/Humanoid.pth",
        type=str,
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_runtime(args: argparse.Namespace) -> DRPPORuntime:
    runtime_args = SimpleNamespace(
        device_id=int(args.device_id),
        max_motion_frames=int(args.max_motion_frames),
        model_path=str(args.model_path),
        guidance_param=float(args.guidance_param),
        deterministic_denoising=False,
        stochastic_first_k_steps=0,
        data_root=str(args.data_root),
        phc_num_envs=int(args.phc_num_envs),
        phc_max_steps=int(args.phc_max_steps),
        phc_actor_ckpt=str(args.phc_actor_ckpt),
        dense_reward_weight=float(args.dense_reward_weight),
        reward_norm=bool(args.reward_norm),
        pose_reward_weight=float(args.pose_reward_weight),
        pose_reward_alpha=float(args.pose_reward_alpha),
        velocity_reward_weight=float(args.velocity_reward_weight),
        velocity_reward_alpha=float(args.velocity_reward_alpha),
        success_bonus=float(args.success_bonus),
        fail_penalty=float(args.fail_penalty),
        frame_gamma=float(args.frame_gamma),
        frame_lambda=float(args.frame_lambda),
        num_controlled_reverse_steps=int(args.num_controlled_reverse_steps),
    )
    return DRPPORuntime(runtime_args)


def load_reverse_noise_policy(args: argparse.Namespace, runtime: DRPPORuntime) -> nn.Module:
    if args.noise_source == "gaussian":
        return StandardGaussianReverseNoisePolicy(action_dim=int(runtime.actor.njoints))

    checkpoint_path = Path(args.drppo_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"DRPPO checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu")
    ckpt_args = payload.get("args", {}) if isinstance(payload, dict) else {}
    sigma_embed_dim = int(ckpt_args.get("sigma_embed_dim", args.sigma_embed_dim))
    hidden_dim = int(ckpt_args.get("reverse_policy_hidden_dim", args.reverse_policy_hidden_dim))
    num_layers = int(ckpt_args.get("reverse_policy_num_layers", args.reverse_policy_num_layers))
    log_std_init = float(ckpt_args.get("reverse_log_std_init", args.reverse_log_std_init))

    policy = FrameReverseNoisePolicy(
        text_dim=512,
        sigma_embed_dim=sigma_embed_dim,
        hidden_dim=hidden_dim,
        action_dim=int(runtime.actor.njoints),
        num_diffusion_steps=int(runtime.diffusion.num_timesteps),
        num_layers=num_layers,
        log_std_init=log_std_init,
    )
    policy.load_state_dict(payload["reverse_noise_policy"])
    policy.to(runtime.device)
    policy.eval()
    return policy


def select_batch_entries(args: argparse.Namespace):
    pools = build_prompt_entry_pools(
        data_root=args.data_root,
        split=args.split,
        max_motion_frames=args.max_motion_frames,
        failure_cases_file=args.failure_cases_file or None,
    )
    rng = random.Random(args.seed)
    entries, sampling_metrics = sample_prompt_batch(
        pools=pools,
        mode=args.train_sampling_mode,
        batch_size=args.prompt_batch_size,
        rng=rng,
        success_sampling_weight=args.success_sampling_weight,
        failure_sampling_weight=args.failure_sampling_weight,
    )
    return entries, sampling_metrics


def resolve_sampling_seed(args: argparse.Namespace) -> Optional[int]:
    if args.sampling_seed is not None:
        return int(args.sampling_seed)
    if args.fixed_sampling_seed:
        return int(args.seed)
    return None


def compute_per_sample_reward_rows(
    episodes: Sequence[Dict],
    entries,
    generated_joints_20fps: torch.Tensor,
    gt_joints_20fps: torch.Tensor,
    runtime: DRPPORuntime,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    joint_ids = torch.as_tensor(runtime._imitation_joint_ids, dtype=torch.long, device=runtime.device)

    for idx, (episode, entry) in enumerate(zip(episodes, entries)):
        target_len_20fps = max(1, min(int(episode["length"]), runtime.max_motion_frames))
        dense_np = _align_reward_steps_to_frames(
            reward_steps=np.asarray(episode["reward_steps"], dtype=np.float32),
            exec_len_30hz=int(episode["exec_len"]),
            target_len_20fps=target_len_20fps,
        )
        exec_len_20fps = int(dense_np.shape[0])

        ##############################################
        # Rebuild each reward component with the exact
        # same formulas used in build_frame_reward_batch.
        ##############################################
        dense_total = 0.0
        pose_total = 0.0
        vel_total = 0.0
        pose_mse_mean = 0.0

        if exec_len_20fps > 0:
            dense_t = torch.from_numpy(dense_np).to(runtime.device)
            if runtime.args.reward_norm:
                dense_t = dense_t / float(target_len_20fps)
            weighted_dense = dense_t * float(runtime.args.dense_reward_weight)
            dense_total = float(weighted_dense.sum().item())

            pred_joints = generated_joints_20fps[idx, :exec_len_20fps].index_select(1, joint_ids)
            gt_joints = gt_joints_20fps[idx, :exec_len_20fps].index_select(1, joint_ids)
            pose_dist = ((pred_joints - gt_joints) ** 2).sum(dim=-1).mean(dim=-1)
            pose_mse_mean = float(pose_dist.mean().item())

            if float(runtime.args.pose_reward_weight) > 0.0:
                pose_reward = torch.exp(-float(runtime.args.pose_reward_alpha) * pose_dist)
                if runtime.args.reward_norm:
                    pose_reward = pose_reward / float(target_len_20fps)
                weighted_pose = float(runtime.args.pose_reward_weight) * pose_reward
                pose_total = float(weighted_pose.sum().item())

            if float(runtime.args.velocity_reward_weight) > 0.0 and exec_len_20fps > 1:
                pred_vel = pred_joints[1:] - pred_joints[:-1]
                gt_vel = gt_joints[1:] - gt_joints[:-1]
                velocity_dist = ((pred_vel - gt_vel) ** 2).sum(dim=-1).mean(dim=-1)
                velocity_reward = torch.exp(-float(runtime.args.velocity_reward_alpha) * velocity_dist)
                if runtime.args.reward_norm:
                    velocity_reward = velocity_reward / float(max(1, target_len_20fps - 1))
                weighted_velocity = float(runtime.args.velocity_reward_weight) * velocity_reward
                vel_total = float(weighted_velocity.sum().item())

        terminal_total = float(runtime.args.fail_penalty) if bool(episode["terminate"]) else float(runtime.args.success_bonus)
        total = dense_total + pose_total + vel_total + terminal_total

        reward_raw = np.asarray(episode.get("reward_raw", np.zeros((0, 4), dtype=np.float32)), dtype=np.float32)
        rows.append(
            {
                "db_key": entry.db_key,
                "caption_idx": int(entry.caption_idx),
                "caption": entry.caption,
                "target_len_20fps": int(entry.length_20fps),
                "gt_len_20fps": int(entry.gt_length_20fps),
                "exec_len_30hz": int(episode["exec_len"]),
                "exec_len_20fps": int(exec_len_20fps),
                "success": int(bool(episode.get("success", False))),
                "terminate": int(bool(episode.get("terminate", False))),
                "exec_ratio_mean": float(max(1, min(exec_len_20fps, target_len_20fps)) / float(target_len_20fps)),
                "phc_return_mean": float(episode.get("return_mean", 0.0)),
                "phc_body_pos_reward_mean": float(reward_raw[:, 0].mean()) if reward_raw.size > 0 else 0.0,
                "phc_vel_reward_mean": float(reward_raw[:, 2].mean()) if reward_raw.shape[-1] > 2 and reward_raw.size > 0 else 0.0,
                "dense_reward_mean": dense_total,
                "imitation_pose_reward_mean": pose_total,
                "pose_mse_mean": pose_mse_mean,
                "pose_mse_sum": float(pose_mse_mean * exec_len_20fps),
                "pose_mse_count": int(exec_len_20fps),
                "imitation_velocity_reward_mean": vel_total,
                "terminal_reward_mean": terminal_total,
                "undiscounted_sequence_reward_mean": total,
            }
        )
    return rows


def mean_of(rows: Sequence[Dict[str, float]], key: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([float(row[key]) for row in rows]))


def summarize_rows(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = [
        "success",
        "terminate",
        "exec_ratio_mean",
        "phc_return_mean",
        "phc_body_pos_reward_mean",
        "phc_vel_reward_mean",
        "dense_reward_mean",
        "imitation_pose_reward_mean",
        "pose_mse_mean",
        "imitation_velocity_reward_mean",
        "terminal_reward_mean",
        "undiscounted_sequence_reward_mean",
    ]
    summary = {key: mean_of(rows, key) for key in keys}
    pose_count = float(sum(int(row["pose_mse_count"]) for row in rows))
    if pose_count > 0.0:
        pose_sum = float(sum(float(row["pose_mse_sum"]) for row in rows))
        summary["pose_mse_mean"] = pose_sum / pose_count
    return summary


def builder_consistency_summary(reward_batch, rows: Sequence[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    rows_summary = summarize_rows(rows)
    pairs = {
        "dense_reward_mean": float(reward_batch.dense_reward_mean),
        "imitation_pose_reward_mean": float(reward_batch.imitation_pose_reward_mean),
        "pose_mse_mean": float(reward_batch.pose_mse_mean),
        "imitation_velocity_reward_mean": float(reward_batch.imitation_velocity_reward_mean),
        "terminal_reward_mean": float(reward_batch.terminal_reward_mean),
        "undiscounted_sequence_reward_mean": float(reward_batch.undiscounted_sequence_reward_mean),
        "exec_ratio_mean": float(reward_batch.exec_ratio_mean),
    }
    out: Dict[str, Dict[str, float]] = {}
    for key, builder_value in pairs.items():
        rows_mean = float(rows_summary[key])
        out[key] = {
            "builder": builder_value,
            "rows_mean": rows_mean,
            "abs_diff": abs(builder_value - rows_mean),
        }
    return out


def write_csv(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.noise_source == "drppo" and not args.drppo_path:
        raise ValueError("--drppo_path is required when noise_source=drppo.")

    output_dir = Path(args.output_root).expanduser().resolve() / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    entries, sampling_metrics = select_batch_entries(args)
    runtime = build_runtime(args)
    reverse_noise_policy = load_reverse_noise_policy(args, runtime)
    sampling_seed = resolve_sampling_seed(args)

    try:
        ##############################################
        # Run one DRPPO batch rollout without PPO update.
        # This matches training-time reward computation.
        ##############################################
        sampled = runtime._generate_batch(
            entries=entries,
            reverse_noise_policy=reverse_noise_policy,
            policy_deterministic=bool(args.deterministic_policy),
            sampling_seed=sampling_seed,
        )
        generated_joints = torch.from_numpy(runtime._sample_to_joint_batch(sampled["final_sample"])).to(runtime.device)
        gt_joints = runtime._load_gt_joint_batch(entries)
        ref_motion_batch, lengths_30hz = runtime._sample_to_reference_batch(sampled["final_sample"], sampled["lengths_20fps"])
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
        rows = compute_per_sample_reward_rows(
            episodes=episodes,
            entries=entries,
            generated_joints_20fps=generated_joints,
            gt_joints_20fps=gt_joints,
            runtime=runtime,
        )
    finally:
        runtime.close()

    write_csv(output_dir / "samples.csv", rows)

    ##############################################
    # Print batch-level summary first, then all rows.
    # This makes it easy to compare against wandb.
    ##############################################
    builder_summary = {
        "dense_reward_mean": float(reward_batch.dense_reward_mean),
        "imitation_pose_reward_mean": float(reward_batch.imitation_pose_reward_mean),
        "pose_mse_mean": float(reward_batch.pose_mse_mean),
        "imitation_velocity_reward_mean": float(reward_batch.imitation_velocity_reward_mean),
        "terminal_reward_mean": float(reward_batch.terminal_reward_mean),
        "undiscounted_sequence_reward_mean": float(reward_batch.undiscounted_sequence_reward_mean),
        "exec_ratio_mean": float(reward_batch.exec_ratio_mean),
    }
    rows_summary = summarize_rows(rows)
    consistency = builder_consistency_summary(reward_batch, rows)

    print(
        "[batch] "
        f"size={len(rows)} "
        f"success_rate={rows_summary.get('success', 0.0):.4f} "
        f"return={rows_summary.get('phc_return_mean', 0.0):.4f} "
        f"dense={rows_summary.get('dense_reward_mean', 0.0):.4f} "
        f"pose={rows_summary.get('imitation_pose_reward_mean', 0.0):.4f} "
        f"vel={rows_summary.get('imitation_velocity_reward_mean', 0.0):.4f} "
        f"terminal={rows_summary.get('terminal_reward_mean', 0.0):.4f} "
        f"total={rows_summary.get('undiscounted_sequence_reward_mean', 0.0):.4f}"
    )
    print(json.dumps({"builder_summary": builder_summary, "rows_summary": rows_summary, "consistency": consistency}, ensure_ascii=False, indent=2))

    for sample_idx, row in enumerate(rows):
        print(
            f"[sample {sample_idx + 1:03d}/{len(rows):03d}] "
            f"{'SUCCESS' if row['success'] else 'FAIL'} "
            f"db_key={row['db_key']} caption_idx={row['caption_idx']} "
            f"return={row['phc_return_mean']:.4f} "
            f"dense={row['dense_reward_mean']:.4f} "
            f"pose={row['imitation_pose_reward_mean']:.4f} "
            f"vel={row['imitation_velocity_reward_mean']:.4f} "
            f"terminal={row['terminal_reward_mean']:.4f} "
            f"total={row['undiscounted_sequence_reward_mean']:.4f}"
        )

    payload = {
        "run_name": args.run_name,
        "noise_source": args.noise_source,
        "drppo_path": str(Path(args.drppo_path).expanduser().resolve()) if args.drppo_path else "",
        "sampling_seed": sampling_seed,
        "sampling_metrics": sampling_metrics,
        "num_samples": len(rows),
        "builder_summary": builder_summary,
        "rows_summary": rows_summary,
        "consistency": consistency,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "summary_json": str(output_dir / "summary.json")}, ensure_ascii=False))

'''
python exp/diagnose_drppo_batch_reward.py \
  --model_path /home/gxy/hay-thesis/motion-diffusion-model-phc/save/humanml_enc_512_50steps/model000750000.pt \
  --data_root /home/gxy/hay-thesis/HumanML3D/HumanML3D \
  --run_name drppo_0009_batch_probe \
  --output_root /home/gxy/hay-thesis/motion-diffusion-model-phc/exp/output \
  --split train \
  --prompt_batch_size 256 \
  --train_sampling_mode failure_only \
  --failure_cases_file /home/gxy/hay-thesis/PHC/output/eval_mdm/mdm_eval_train_full/falling_cases.txt \
  --noise_source gaussian \
  --device_id 0 \
  --guidance_param 2.5 \
  --num_controlled_reverse_steps 5 \
  --frame_gamma 0.99 \
  --frame_lambda 0.9 \
  --reward_norm \
  --dense_reward_weight 5.0 \
  --pose_reward_weight 1.0 \
  --pose_reward_alpha 5.0 \
  --velocity_reward_weight 0.0 \
  --velocity_reward_alpha 0.5 \
  --success_bonus 0.0 \
  --fail_penalty -1.0 \
  --phc_num_envs 256 \
  --phc_max_steps 400 \
  --fixed_sampling_seed \
  --seed 0
'''
if __name__ == "__main__":
    main()
