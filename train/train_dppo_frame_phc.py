import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

try:  # Isaac Gym must be imported before torch in this process.
    import isaacgym  # noqa: F401
except ImportError:
    isaacgym = None

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.dppo_frame_rl.data import (
    build_prompt_entry_pools,
    build_prompt_entries,
    sample_prompt_batch,
    select_eval_batch,
)
from train.dppo_frame_rl.logging import merge_metrics
from train.dppo_frame_rl.models.frame_critic import FrameCritic
from train.dppo_frame_rl.runtime import FrameDPPORuntime
from model.lora_attention import lora_state_dict

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--data_root", required=True, type=str)
    parser.add_argument("--run_name", required=True, type=str)

    parser.add_argument("--device_id", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--max_motion_frames", default=196, type=int)
    parser.add_argument("--prompt_batch_size", default=16, type=int)
    parser.add_argument("--ppo_minibatch_size", default=8, type=int)
    parser.add_argument("--critic_minibatch_size", default=16, type=int)
    parser.add_argument("--num_outer_steps", default=100, type=int)
    parser.add_argument("--actor_num_epochs", default=1, type=int)
    parser.add_argument("--critic_num_epochs", default=2, type=int)
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("--critic_lr", default=3e-4, type=float)
    parser.add_argument("--lr_schedule", default="constant", choices=["constant", "cosine"], type=str)
    parser.add_argument("--min_lr", default=0.0, type=float)
    parser.add_argument("--critic_lr_schedule", default="constant", choices=["constant", "cosine"], type=str)
    parser.add_argument("--critic_min_lr", default=0.0, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--clip_range", default=1e-2, type=float)
    parser.add_argument("--clip_range_schedule", default="constant", choices=["constant", "exponential"], type=str)
    parser.add_argument("--clip_range_base", default=None, type=float)
    parser.add_argument("--clip_range_rate", default=3.0, type=float)
    parser.add_argument("--kl_coef", default=0.0, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)

    parser.add_argument("--guidance_param", default=2.5, type=float)
    parser.add_argument("--ft_denoising_steps", default=0, type=int)
    parser.add_argument("--gamma_denoising", default=0.995, type=float)
    parser.add_argument("--min_sampling_denoising_std", default=1e-2, type=float)
    parser.add_argument("--min_logprob_denoising_std", default=1e-2, type=float)
    parser.add_argument("--lora_rank", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16.0, type=float)
    parser.add_argument("--lora_layer_scope", default="all", choices=["all", "last3"], type=str)
    parser.add_argument("--resume_lora_path", default="", type=str)

    parser.add_argument("--frame_gamma", default=0.995, type=float)
    parser.add_argument("--frame_lambda", default=0.95, type=float)
    parser.add_argument("--dense_reward_weight", default=1.0, type=float)
    parser.add_argument("--dense_reward_norm", action="store_true")
    parser.add_argument("--success_bonus", default=0.0, type=float)
    parser.add_argument("--fail_penalty", default=-5.0, type=float)
    parser.add_argument("--value_target_norm", default="none", choices=["none", "popart"], type=str)
    parser.add_argument("--popart_beta", default=5e-4, type=float)
    parser.add_argument("--popart_epsilon", default=1e-5, type=float)

    parser.add_argument("--train_split", default="train", type=str)
    parser.add_argument("--eval_split", default="test", type=str)
    parser.add_argument("--train_sampling_mode", choices=["uniform", "failure_only", "mixed"], default="uniform")
    parser.add_argument("--failure_cases_file", default="", type=str)
    parser.add_argument("--failure_eval_cases_file", default="", type=str)
    parser.add_argument("--success_sampling_weight", default=1.0, type=float)
    parser.add_argument("--failure_sampling_weight", default=1.0, type=float)

    parser.add_argument("--phc_num_envs", default=16, type=int)
    parser.add_argument("--phc_max_steps", default=420, type=int)
    parser.add_argument(
        "--phc_actor_ckpt",
        default="/home/gxy/hay-thesis/PHC/output/HumanoidIm/phc_kp_mcp_iccv/Humanoid.pth",
        type=str,
    )
    parser.add_argument("--eval_interval", default=5, type=int)
    parser.add_argument("--eval_prompt_batch_size", default=4, type=int)
    parser.add_argument("--fixed_sampling_seed", action="store_true")

    parser.add_argument("--save_root", default="save", type=str)
    parser.add_argument("--save_interval", default=10, type=int)
    parser.add_argument("--log_interval", default=1, type=int)
    parser.add_argument("--wandb_mode", default="disabled", choices=["disabled", "offline", "online"], type=str)
    parser.add_argument("--wandb_project", default="mdm-phc-dppo-frame", type=str)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prefix_metrics(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def masked_value_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float()
    denom = mask_f.sum().clamp(min=1.0)
    return 0.5 * (((pred - target) ** 2) * mask_f).sum() / denom


def normalize_advantages(advantages: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    valid = advantages[mask]
    if valid.numel() == 0:
        return {
            "advantages": torch.zeros_like(advantages),
            "adv_mean": advantages.new_zeros(()),
            "adv_std": advantages.new_zeros(()),
        }
    mean = valid.mean()
    std = valid.std(unbiased=False) + 1e-8
    normalized = ((advantages - mean) / std) * mask.float()
    return {"advantages": normalized, "adv_mean": mean, "adv_std": std}


def iter_sample_minibatches(num_samples: int, batch_size: int, device: torch.device):
    permutation = torch.randperm(num_samples, device=device)
    for start in range(0, num_samples, batch_size):
        yield permutation[start : start + batch_size]


def _cosine_decay(step: int, total_steps: int, max_lr: float, min_lr: float) -> float:
    if total_steps <= 1:
        return float(max_lr)
    progress = float(step - 1) / float(total_steps - 1)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return float(min_lr + (max_lr - min_lr) * cosine)


def resolve_scheduled_lr(
    schedule: str,
    step: int,
    total_steps: int,
    base_lr: float,
    min_lr: float,
) -> float:
    if schedule == "constant":
        return float(base_lr)
    return _cosine_decay(step=step, total_steps=total_steps, max_lr=base_lr, min_lr=min_lr)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def train_critic(args, critic, optimizer, rollout) -> Dict[str, float]:
    metrics = {
        "value_loss": 0.0,
    }
    if getattr(critic, "popart_enabled", False):
        metrics.update(
            {
                "popart_mu": 0.0,
            }
        )
    num_updates = 0

    if getattr(critic, "popart_enabled", False):
        # Update running target statistics once per rollout.
        popart_stats = critic.update_popart_stats(rollout.frame_returns, rollout.frame_exec_mask)
        metrics["popart_mu"] = float(popart_stats.get("popart_mu", 0.0))

    for _ in range(args.critic_num_epochs):
        for batch_inds in iter_sample_minibatches(rollout.frame_features.shape[0], args.critic_minibatch_size, rollout.frame_features.device):
            target = rollout.frame_returns[batch_inds]
            mask = rollout.frame_exec_mask[batch_inds]

            if getattr(critic, "popart_enabled", False):
                # Regress in normalized value space but keep raw-value metrics for logging.
                pred_norm = critic(
                    rollout.frame_features[batch_inds],
                    rollout.text_embeds[batch_inds],
                    normalized=True,
                )
                pred = critic.denormalize_values(pred_norm)
                target_norm = critic.normalize_targets(target)
                loss = masked_value_loss(pred_norm, target_norm, mask)
            else:
                pred = critic(rollout.frame_features[batch_inds], rollout.text_embeds[batch_inds])
                pred_norm = pred
                target_norm = target
                loss = masked_value_loss(pred, target, mask)

            optimizer.zero_grad()
            loss.backward()
            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                metrics["value_loss"] += float(loss.item())
                num_updates += 1

    if num_updates == 0:
        return metrics
    averaged = {}
    for key, value in metrics.items():
        if key in {"popart_mu"} and getattr(critic, "popart_enabled", False):
            averaged[key] = value
        else:
            averaged[key] = value / num_updates
    return averaged


def train_actor(args, runtime: FrameDPPORuntime, actor_optimizer, rollout) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    num_updates = 0

    for _ in range(args.actor_num_epochs):
        for batch_inds in iter_sample_minibatches(len(rollout.texts), args.ppo_minibatch_size, rollout.frame_features.device):
            batch_texts = [rollout.texts[idx] for idx in batch_inds.tolist()]
            loss_dict = runtime.policy.loss(
                texts=batch_texts,
                text_embeds=rollout.text_embeds[batch_inds],
                lengths_20fps=rollout.lengths_20fps[batch_inds],
                chain_prev=rollout.chain_prev[batch_inds],
                chain_next=rollout.chain_next[batch_inds],
                timesteps=rollout.timesteps[batch_inds],
                frame_advantages=rollout.frame_advantages[batch_inds],
                frame_exec_mask=rollout.frame_exec_mask[batch_inds],
                frame_logprobs_old=rollout.frame_logprobs_old[batch_inds],
            )

            actor_optimizer.zero_grad()
            loss_dict["loss"].backward()
            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(list(runtime.policy.trainable_parameters()), args.max_grad_norm)
            actor_optimizer.step()

            with torch.no_grad():
                for key, value in loss_dict.items():
                    if key == "loss":
                        continue
                    metrics[key] = metrics.get(key, 0.0) + float(value.item())
                num_updates += 1

    if num_updates == 0:
        return metrics
    return {key: value / num_updates for key, value in metrics.items()}


def save_checkpoint(
    save_dir: Path,
    filename: str,
    step: int,
    runtime: FrameDPPORuntime,
    critic,
    actor_optimizer,
    critic_optimizer,
    args,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "step": step,
        "actor_lora": lora_state_dict(runtime.policy.actor, bank_name="ft"),
        "critic": critic.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(checkpoint, save_dir / filename)


def select_best_metric(merged_eval: Dict[str, float]) -> Optional[tuple]:
    if "eval_failure/phc_return_mean" in merged_eval:
        return "eval_failure/phc_return_mean", float(merged_eval["eval_failure/phc_return_mean"])
    if "eval/phc_return_mean" in merged_eval:
        return "eval/phc_return_mean", float(merged_eval["eval/phc_return_mean"])
    return None


def maybe_init_wandb(args):
    if args.wandb_mode == "disabled" or wandb is None:
        return None
    return wandb.init(
        project=args.wandb_project,
        mode=args.wandb_mode,
        name=args.run_name,
        config=vars(args),
    )


def main():
    args = parse_args()
    set_seed(args.seed)

    save_dir = Path(args.save_root).expanduser().resolve() / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)

    if args.train_sampling_mode in {"failure_only", "mixed"} and not args.failure_cases_file:
        raise ValueError("--failure_cases_file is required when train_sampling_mode is failure_only or mixed.")

    train_pools = build_prompt_entry_pools(
        data_root=args.data_root,
        split=args.train_split,
        max_motion_frames=args.max_motion_frames,
        failure_cases_file=args.failure_cases_file or None,
    )
    eval_entries = build_prompt_entries(
        data_root=args.data_root,
        split=args.eval_split,
        max_motion_frames=args.max_motion_frames,
    )
    failure_eval_entries: Optional[Sequence] = None
    if args.failure_eval_cases_file:
        failure_eval_entries = build_prompt_entries(
            data_root=args.data_root,
            split=args.eval_split,
            max_motion_frames=args.max_motion_frames,
            failure_cases_file=args.failure_eval_cases_file,
        )

    runtime = FrameDPPORuntime(args)
    critic = FrameCritic(
        max_frames=args.max_motion_frames,
        value_target_norm=args.value_target_norm,
        popart_beta=args.popart_beta,
        popart_epsilon=args.popart_epsilon,
    ).to(runtime.device)
    actor_optimizer = AdamW(list(runtime.policy.trainable_parameters()), lr=args.lr, weight_decay=args.weight_decay)
    critic_optimizer = AdamW(critic.parameters(), lr=args.critic_lr, weight_decay=args.weight_decay)
    wandb_run = maybe_init_wandb(args)

    rng = random.Random(args.seed)
    best_eval_score = float("-inf")
    best_eval_name: Optional[str] = None
    try:
        for outer_step in range(1, args.num_outer_steps + 1):
            actor_lr = resolve_scheduled_lr(
                schedule=args.lr_schedule,
                step=outer_step,
                total_steps=args.num_outer_steps,
                base_lr=args.lr,
                min_lr=args.min_lr,
            )
            critic_lr = resolve_scheduled_lr(
                schedule=args.critic_lr_schedule,
                step=outer_step,
                total_steps=args.num_outer_steps,
                base_lr=args.critic_lr,
                min_lr=args.critic_min_lr,
            )
            set_optimizer_lr(actor_optimizer, actor_lr)
            set_optimizer_lr(critic_optimizer, critic_lr)

            batch_entries, _ = sample_prompt_batch(
                pools=train_pools,
                mode=args.train_sampling_mode,
                batch_size=args.prompt_batch_size,
                rng=rng,
                success_sampling_weight=args.success_sampling_weight,
                failure_sampling_weight=args.failure_sampling_weight,
            )
            rollout, rollout_metrics = runtime.collect_rollout(
                entries=batch_entries,
                critic=critic,
                deterministic=False,
                sampling_seed=args.seed if args.fixed_sampling_seed else None,
            )

            norm_adv = normalize_advantages(rollout.frame_advantages, rollout.frame_exec_mask)
            rollout.frame_advantages = norm_adv["advantages"]
            rollout_metrics = merge_metrics(
                rollout_metrics,
                {
                },
            )

            critic_metrics = train_critic(args, critic, critic_optimizer, rollout)
            actor_metrics = train_actor(args, runtime, actor_optimizer, rollout)
            metrics = merge_metrics(rollout_metrics, critic_metrics, actor_metrics)
            metrics = merge_metrics(
                metrics,
                {
                    "actor_lr": actor_lr,
                    "critic_lr": critic_lr,
                },
            )
            train_metrics = prefix_metrics("train/", metrics)

            if outer_step % args.log_interval == 0:
                print(
                    f"[train] step={outer_step} "
                    + " ".join(f"{k}={v:.4f}" for k, v in sorted(train_metrics.items()))
                )
            if wandb_run is not None:
                wandb.log({**train_metrics, "outer_step": outer_step}, step=outer_step)

            if outer_step % args.eval_interval == 0:
                eval_batch = select_eval_batch(eval_entries, args.eval_prompt_batch_size, offset=0)
                eval_metrics = prefix_metrics(
                    "eval/",
                    runtime.evaluate_entries(
                        entries=eval_batch,
                        deterministic=False,
                        sampling_seed=args.seed if args.fixed_sampling_seed else None,
                    ),
                )
                merged_eval = dict(eval_metrics)

                if failure_eval_entries:
                    failure_batch = select_eval_batch(failure_eval_entries, args.eval_prompt_batch_size, offset=0)
                    failure_metrics = prefix_metrics(
                        "eval_failure/",
                        runtime.evaluate_entries(
                            entries=failure_batch,
                            deterministic=False,
                            sampling_seed=args.seed if args.fixed_sampling_seed else None,
                        ),
                    )
                    merged_eval.update(failure_metrics)

                print(f"[eval] step={outer_step} " + " ".join(f"{k}={v:.4f}" for k, v in sorted(merged_eval.items())))
                if wandb_run is not None:
                    wandb.log({**merged_eval, "outer_step": outer_step}, step=outer_step)

                best_metric = select_best_metric(merged_eval)
                if best_metric is not None:
                    metric_name, metric_value = best_metric
                    if metric_value > best_eval_score:
                        best_eval_score = metric_value
                        best_eval_name = metric_name
                        save_checkpoint(
                            save_dir=save_dir,
                            filename="best_lora.pt",
                            step=outer_step,
                            runtime=runtime,
                            critic=critic,
                            actor_optimizer=actor_optimizer,
                            critic_optimizer=critic_optimizer,
                            args=args,
                        )
                        print(f"[checkpoint] step={outer_step} saved best_lora.pt ({metric_name}={metric_value:.4f})")

            if outer_step % args.save_interval == 0 or outer_step == args.num_outer_steps:
                save_checkpoint(
                    save_dir=save_dir,
                    filename="latest_lora.pt",
                    step=outer_step,
                    runtime=runtime,
                    critic=critic,
                    actor_optimizer=actor_optimizer,
                    critic_optimizer=critic_optimizer,
                    args=args,
                )
    finally:
        runtime.close()
        if wandb_run is not None:
            wandb.finish()

'''
python train/train_dppo_frame_phc.py \
  --model_path /home/gxy/hay-thesis/motion-diffusion-model-phc/save/humanml_enc_512_50steps/model000750000.pt \
  --data_root /home/gxy/hay-thesis/HumanML3D/HumanML3D \
  --run_name dppo_frame_failure_only_00016_onlyfailure \
  --train_split train \
  --eval_split test \
  --train_sampling_mode failure_only \
  --failure_cases_file /home/gxy/hay-thesis/PHC/output/eval_mdm/mdm_eval_train_full/falling_cases.txt \
  --failure_eval_cases_file /home/gxy/hay-thesis/PHC/output/eval_mdm/mdm_eval_v7_full/falling_cases.txt \
  --phc_actor_ckpt /home/gxy/hay-thesis/PHC/output/HumanoidIm/phc_kp_mcp_iccv/Humanoid.pth \
  --dense_reward_weight 0 \
  --fail_penalty -90.0 \
  --max_motion_frames 196 \
  --prompt_batch_size 32 \
  --ppo_minibatch_size 8 \
  --critic_minibatch_size 32 \
  --actor_num_epochs 2 \
  --critic_num_epochs 2 \
  --lr 1e-4 \
  --lr_schedule cosine \
  --min_lr 1e-5 \
  --critic_lr 3e-4 \
  --critic_lr_schedule cosine \
  --critic_min_lr 1e-6 \
  --clip_range 5e-2 \
  --phc_num_envs 32 \
  --phc_max_steps 420 \
  --num_outer_steps 200 \
  --ft_denoising_steps 4 \
  --value_target_norm popart \
  --popart_beta 0.005 \
  --popart_epsilon 1e-5 \
  --eval_interval 10 \
  --eval_prompt_batch_size 16 \
  --save_interval 10 \
  --log_interval 1 \
  --fixed_sampling_seed \
  --wandb_project mdm-phc-dppo-frame \
  --wandb_mode online

  #small sample debugging
  python train/train_dppo_frame_phc.py \
  --model_path /home/gxy/hay-thesis/motion-diffusion-model-phc/save/humanml_enc_512_50steps/model000750000.pt \
  --data_root /home/gxy/hay-thesis/motion-diffusion-model-phc/data-failure \
  --run_name dppo_frame_data_failure_00021_debug \
  --train_split train \
  --eval_split test \
  --train_sampling_mode uniform \
  --phc_actor_ckpt /home/gxy/hay-thesis/PHC/output/HumanoidIm/phc_kp_mcp_iccv/Humanoid.pth \
  --max_motion_frames 196 \
  --prompt_batch_size 32 \
  --ppo_minibatch_size 8 \
  --critic_minibatch_size 32 \
  --actor_num_epochs 1 \
  --critic_num_epochs 2 \
  --lr 5e-4 \
  --lr_schedule cosine \
  --min_lr 1e-5 \
  --critic_lr 1e-4 \
  --critic_lr_schedule cosine \
  --critic_min_lr 1e-6 \
  --clip_range 1e-1 \
  --phc_num_envs 32 \
  --phc_max_steps 420 \
  --num_outer_steps 200 \
  --ft_denoising_steps 4 \
  --value_target_norm popart \
  --popart_beta 0.005 \
  --popart_epsilon 1e-5 \
  --eval_interval 5 \
  --eval_prompt_batch_size 1 \
  --save_interval 10 \
  --log_interval 1 \
  --fixed_sampling_seed \
  --wandb_project mdm-phc-dppo-frame \
  --wandb_mode online \
  --dense_reward_norm \
  --fail_penalty -9.0

  # normal big batch
  python train/train_dppo_frame_phc.py \
  --model_path /home/gxy/hay-thesis/motion-diffusion-model-phc/save/humanml_enc_512_50steps/model000750000.pt \
  --data_root /home/gxy/hay-thesis/HumanML3D/HumanML3D \
  --run_name dppo_frame_failure_only_00022_lr_clipran \
  --train_split train \
  --eval_split test \
  --train_sampling_mode failure_only \
  --failure_cases_file /home/gxy/hay-thesis/PHC/output/eval_mdm/mdm_eval_train_full/falling_cases.txt \
  --failure_eval_cases_file /home/gxy/hay-thesis/PHC/output/eval_mdm/mdm_eval_v7_full/falling_cases.txt \
  --phc_actor_ckpt /home/gxy/hay-thesis/PHC/output/HumanoidIm/phc_kp_mcp_iccv/Humanoid.pth \
  --max_motion_frames 196 \
  --prompt_batch_size 256 \
  --ppo_minibatch_size 64 \
  --critic_minibatch_size 128 \
  --actor_num_epochs 1 \
  --critic_num_epochs 2 \
  --lr 1e-4 \
  --lr_schedule cosine \
  --min_lr 1e-5 \
  --critic_lr 1e-4 \
  --critic_lr_schedule cosine \
  --critic_min_lr 1e-6 \
  --clip_range 5e-2 \
  --phc_num_envs 256 \
  --phc_max_steps 420 \
  --num_outer_steps 300 \
  --ft_denoising_steps 4 \
  --value_target_norm popart \
  --popart_beta 0.005 \
  --popart_epsilon 1e-5 \
  --eval_interval 10 \
  --eval_prompt_batch_size 128 \
  --save_interval 10 \
  --log_interval 1 \
  --fixed_sampling_seed \
  --wandb_project mdm-phc-dppo-frame \
  --wandb_mode online \
  --dense_reward_norm \
  --fail_penalty -9.0

  # finetune LoRA
  python train/train_dppo_frame_phc.py \
  --model_path /home/gxy/hay-thesis/motion-diffusion-model-phc/save/stage2_repa/stage2_repa_run_002/best_model.pt \
  --data_root /home/gxy/hay-thesis/HumanML3D/HumanML3D \
  --run_name dppo_frame_failure_only_stage2_repa_0001 \
  --train_split train \
  --eval_split test \
  --train_sampling_mode failure_only \
  --failure_cases_file /home/gxy/hay-thesis/PHC/output/eval_mdm/mdm_eval_train_full/falling_cases.txt \
  --failure_eval_cases_file /home/gxy/hay-thesis/PHC/output/eval_mdm/mdm_eval_v7_full/falling_cases.txt \
  --phc_actor_ckpt /home/gxy/hay-thesis/PHC/output/HumanoidIm/phc_kp_mcp_iccv/Humanoid.pth \
  --max_motion_frames 196 \
  --prompt_batch_size 256 \
  --ppo_minibatch_size 64 \
  --critic_minibatch_size 128 \
  --actor_num_epochs 1 \
  --critic_num_epochs 2 \
  --lr 1e-4 \
  --lr_schedule cosine \
  --min_lr 1e-5 \
  --critic_lr 1e-4 \
  --critic_lr_schedule cosine \
  --critic_min_lr 1e-6 \
  --clip_range 5e-2 \
  --phc_num_envs 256 \
  --phc_max_steps 420 \
  --num_outer_steps 300 \
  --ft_denoising_steps 4 \
  --value_target_norm popart \
  --popart_beta 0.005 \
  --popart_epsilon 1e-5 \
  --eval_interval 10 \
  --eval_prompt_batch_size 128 \
  --save_interval 10 \
  --log_interval 1 \
  --fixed_sampling_seed \
  --wandb_project mdm-phc-dppo-frame \
  --wandb_mode online \
  --dense_reward_norm \
  --fail_penalty -9.0
'''
if __name__ == "__main__":
    main()
