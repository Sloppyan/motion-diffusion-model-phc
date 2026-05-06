import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

try:  # Isaac Gym must be imported before torch in this process.
    import isaacgym  # noqa: F401
except ImportError:
    isaacgym = None

import numpy as np
import torch
from torch.optim import AdamW

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.dppo_frame_rl.data import (
    build_prompt_entries,
    build_prompt_entry_pools,
    sample_prompt_batch,
    select_eval_batch,
)
from train.dppo_frame_rl.logging import merge_metrics
from train.dppo_frame_rl.models.frame_critic import FrameCritic
from train.drppo_rl import DRPPORuntime, FrameReverseNoisePolicy, compute_reverse_step_advantages
from train.drppo_rl.ppo import compute_reverse_noise_policy_loss

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
    parser.add_argument("--entropy_coef", default=0.0, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)

    parser.add_argument("--guidance_param", default=2.5, type=float)
    parser.add_argument("--num_controlled_reverse_steps", default=5, type=int)
    parser.add_argument("--sigma_embed_dim", default=128, type=int)
    parser.add_argument("--reverse_policy_hidden_dim", default=512, type=int)
    parser.add_argument("--reverse_policy_num_layers", default=3, type=int)
    parser.add_argument("--reverse_log_std_init", default=0.0, type=float)
    parser.add_argument("--reverse_noise_prior_kl_coef", default=0.0, type=float)
    parser.add_argument("--deterministic_eval_policy", action="store_true")

    parser.add_argument("--frame_gamma", default=0.995, type=float)
    parser.add_argument("--frame_lambda", default=0.95, type=float)
    parser.add_argument("--denoising_discount", default=0.95, type=float)
    parser.add_argument("--adv_norm_std_only", action="store_true")
    parser.add_argument("--dense_reward_weight", default=1.0, type=float)
    parser.set_defaults(reward_norm=False)
    parser.add_argument("--reward_norm", dest="reward_norm", action="store_true")
    parser.add_argument("--pose_reward_weight", default=0.0, type=float)
    parser.add_argument("--pose_reward_alpha", default=1.0, type=float)
    parser.add_argument("--velocity_reward_weight", default=0.0, type=float)
    parser.add_argument("--velocity_reward_alpha", default=1.0, type=float)
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
    parser.add_argument("--wandb_project", default="mdm-phc-drppo", type=str)
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


def masked_explained_variance(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    valid_pred = pred[mask]
    valid_target = target[mask]
    if valid_target.numel() == 0:
        return float("nan")
    var_target = valid_target.var(unbiased=False)
    if float(var_target.item()) == 0.0:
        return float("nan")
    residual_var = (valid_target - valid_pred).var(unbiased=False)
    return float((1.0 - residual_var / var_target).item())


def normalize_advantages(
    advantages: torch.Tensor,
    mask: torch.Tensor,
    std_only: bool = False,
) -> Dict[str, torch.Tensor]:
    valid = advantages[mask]
    if valid.numel() == 0:
        return {
            "advantages": torch.zeros_like(advantages),
            "adv_mean": advantages.new_zeros(()),
            "adv_std": advantages.new_zeros(()),
        }
    mean = valid.mean()
    std = valid.std(unbiased=False) + 1e-8
    if std_only:
        normalized = (advantages / std) * mask.float()
    else:
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


def resolve_scheduled_lr(schedule: str, step: int, total_steps: int, base_lr: float, min_lr: float) -> float:
    if schedule == "constant":
        return float(base_lr)
    return _cosine_decay(step=step, total_steps=total_steps, max_lr=base_lr, min_lr=min_lr)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def train_critic(args, critic, optimizer, rollout) -> Dict[str, float]:
    metrics = {"value_loss": 0.0, "explained_variance": float("nan")}
    if getattr(critic, "popart_enabled", False):
        metrics.update({"popart_mu": 0.0})
    num_updates = 0

    if getattr(critic, "popart_enabled", False):
        popart_stats = critic.update_popart_stats(rollout.frame_returns, rollout.frame_exec_mask)
        metrics["popart_mu"] = float(popart_stats.get("popart_mu", 0.0))

    for _ in range(args.critic_num_epochs):
        for batch_inds in iter_sample_minibatches(rollout.frame_features.shape[0], args.critic_minibatch_size, rollout.frame_features.device):
            target = rollout.frame_returns[batch_inds]
            mask = rollout.frame_exec_mask[batch_inds]

            if getattr(critic, "popart_enabled", False):
                pred_norm = critic(rollout.frame_features[batch_inds], rollout.text_embeds[batch_inds], normalized=True)
                target_norm = critic.normalize_targets(target)
                loss = masked_value_loss(pred_norm, target_norm, mask)
            else:
                pred = critic(rollout.frame_features[batch_inds], rollout.text_embeds[batch_inds])
                loss = masked_value_loss(pred, target, mask)

            optimizer.zero_grad()
            loss.backward()
            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                metrics["value_loss"] += float(loss.item())
                num_updates += 1

    with torch.no_grad():
        pred = critic(rollout.frame_features, rollout.text_embeds)
        metrics["explained_variance"] = masked_explained_variance(
            pred=pred,
            target=rollout.frame_returns,
            mask=rollout.frame_exec_mask,
        )

    if num_updates == 0:
        return metrics
    averaged = {}
    for key, value in metrics.items():
        if key in {"explained_variance"}:
            averaged[key] = value
        elif key == "popart_mu" and getattr(critic, "popart_enabled", False):
            averaged[key] = value
        else:
            averaged[key] = value / num_updates
    return averaged


def train_actor(args, reverse_noise_policy, actor_optimizer, rollout) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    num_updates = 0

    for _ in range(args.actor_num_epochs):
        for batch_inds in iter_sample_minibatches(len(rollout.texts), args.ppo_minibatch_size, rollout.frame_features.device):
            loss_dict = compute_reverse_noise_policy_loss(
                reverse_noise_policy=reverse_noise_policy,
                text_embeds=rollout.text_embeds[batch_inds],
                controlled_x_t_frames=rollout.controlled_x_t_frames[batch_inds],
                reverse_noise_actions=rollout.reverse_noise_actions[batch_inds],
                reverse_logprobs_old=rollout.reverse_logprobs_old[batch_inds],
                reverse_advantages=rollout.reverse_advantages[batch_inds],
                reverse_action_mask=rollout.reverse_action_mask[batch_inds],
                controlled_timesteps=rollout.controlled_timesteps,
                clip_range=args.clip_range,
                entropy_coef=args.entropy_coef,
                reverse_noise_prior_kl_coef=args.reverse_noise_prior_kl_coef,
            )

            actor_optimizer.zero_grad()
            loss_dict["loss"].backward()
            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(reverse_noise_policy.parameters(), args.max_grad_norm)
            actor_optimizer.step()

            with torch.no_grad():
                for key, value in loss_dict.items():
                    metric_key = "total_loss" if key == "loss" else key
                    metrics[metric_key] = metrics.get(metric_key, 0.0) + float(value.item())
                num_updates += 1

    if num_updates == 0:
        return metrics
    return {key: value / num_updates for key, value in metrics.items()}


def save_checkpoint(
    save_dir: Path,
    filename: str,
    step: int,
    reverse_noise_policy,
    critic,
    actor_optimizer,
    critic_optimizer,
    args,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "step": step,
        "reverse_noise_policy": reverse_noise_policy.state_dict(),
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
        dir=str(Path(args.save_root).expanduser().resolve()),
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

    runtime = DRPPORuntime(args)
    print(
        "[drppo] "
        f"diffusion_num_timesteps={int(runtime.diffusion.num_timesteps)} "
        f"controlled_timesteps={list(runtime.controlled_timestep_ids())}"
    )

    reverse_noise_policy = FrameReverseNoisePolicy(
        text_dim=runtime.actor.clip_dim,
        sigma_embed_dim=args.sigma_embed_dim,
        hidden_dim=args.reverse_policy_hidden_dim,
        action_dim=runtime.actor.njoints * runtime.actor.nfeats,
        num_diffusion_steps=runtime.diffusion.num_timesteps,
        num_layers=args.reverse_policy_num_layers,
        log_std_init=args.reverse_log_std_init,
    ).to(runtime.device)
    critic = FrameCritic(
        max_frames=args.max_motion_frames,
        value_target_norm=args.value_target_norm,
        popart_beta=args.popart_beta,
        popart_epsilon=args.popart_epsilon,
    ).to(runtime.device)

    actor_optimizer = AdamW(reverse_noise_policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    critic_optimizer = AdamW(critic.parameters(), lr=args.critic_lr, weight_decay=args.weight_decay)
    wandb_run = maybe_init_wandb(args)

    rng = random.Random(args.seed)
    best_eval_score = float("-inf")
    try:
        for outer_step in range(1, args.num_outer_steps + 1):
            actor_lr = resolve_scheduled_lr(args.lr_schedule, outer_step, args.num_outer_steps, args.lr, args.min_lr)
            critic_lr = resolve_scheduled_lr(
                args.critic_lr_schedule,
                outer_step,
                args.num_outer_steps,
                args.critic_lr,
                args.critic_min_lr,
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
                reverse_noise_policy=reverse_noise_policy,
                critic=critic,
                policy_deterministic=False,
                sampling_seed=args.seed if args.fixed_sampling_seed else None,
            )

            norm_adv = normalize_advantages(
                rollout.frame_advantages,
                rollout.frame_exec_mask,
                std_only=args.adv_norm_std_only,
            )
            rollout.frame_advantages = norm_adv["advantages"]
            reverse_adv = compute_reverse_step_advantages(
                frame_advantages=rollout.frame_advantages,
                frame_exec_mask=rollout.frame_exec_mask,
                num_controlled_steps=int(rollout.controlled_timesteps.shape[0]),
                gamma_denoising=args.denoising_discount,
            )
            rollout.reverse_advantages = reverse_adv["reverse_advantages"]

            valid_reverse_adv = rollout.reverse_advantages[rollout.reverse_action_mask]
            reverse_adv_mean = float(valid_reverse_adv.mean().item()) if valid_reverse_adv.numel() > 0 else 0.0
            reverse_adv_std = float(valid_reverse_adv.std(unbiased=False).item()) if valid_reverse_adv.numel() > 0 else 0.0
            rollout_metrics = merge_metrics(
                rollout_metrics,
                {
                    "frame_adv_mean": float(norm_adv["adv_mean"].item()),
                    "frame_adv_std": float(norm_adv["adv_std"].item()),
                    "reverse_adv_mean": reverse_adv_mean,
                    "reverse_adv_std": reverse_adv_std,
                },
            )

            critic_metrics = train_critic(args, critic, critic_optimizer, rollout)
            actor_metrics = train_actor(args, reverse_noise_policy, actor_optimizer, rollout)
            metrics = merge_metrics(
                rollout_metrics,
                critic_metrics,
                actor_metrics,
                {
                    "actor_lr": actor_lr,
                    "critic_lr": critic_lr,
                },
            )
            train_metrics = prefix_metrics("train/", metrics)

            if outer_step % args.log_interval == 0:
                print(f"[train] step={outer_step} " + " ".join(f"{k}={v:.4f}" for k, v in sorted(train_metrics.items())))
            if wandb_run is not None:
                wandb.log({**train_metrics, "outer_step": outer_step}, step=outer_step)

            if outer_step % args.eval_interval == 0:
                eval_batch = select_eval_batch(eval_entries, args.eval_prompt_batch_size, offset=0)
                eval_metrics = prefix_metrics(
                    "eval/",
                    runtime.evaluate_entries(
                        entries=eval_batch,
                        reverse_noise_policy=reverse_noise_policy,
                        policy_deterministic=args.deterministic_eval_policy,
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
                            reverse_noise_policy=reverse_noise_policy,
                            policy_deterministic=args.deterministic_eval_policy,
                            sampling_seed=args.seed if args.fixed_sampling_seed else None,
                        ),
                    )
                    merged_eval.update(failure_metrics)

                print(f"[eval] step={outer_step} " + " ".join(f"{k}={v:.4f}" for k, v in sorted(merged_eval.items())))
                if wandb_run is not None:
                    wandb.log({**merged_eval, "outer_step": outer_step}, step=outer_step)

                best_metric = select_best_metric(merged_eval)
                if best_metric is not None:
                    _, metric_value = best_metric
                    if metric_value > best_eval_score:
                        best_eval_score = metric_value
                        save_checkpoint(
                            save_dir=save_dir,
                            filename="best.pt",
                            step=outer_step,
                            reverse_noise_policy=reverse_noise_policy,
                            critic=critic,
                            actor_optimizer=actor_optimizer,
                            critic_optimizer=critic_optimizer,
                            args=args,
                        )
                        print(f"[checkpoint] step={outer_step} saved best.pt")

            if outer_step % args.save_interval == 0 or outer_step == args.num_outer_steps:
                save_checkpoint(
                    save_dir=save_dir,
                    filename="latest.pt",
                    step=outer_step,
                    reverse_noise_policy=reverse_noise_policy,
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
python train/train_drppo_phc.py \
  --model_path /home/gxy/hay-thesis/motion-diffusion-model-phc/save/humanml_enc_512_50steps/model000750000.pt \
  --data_root /home/gxy/hay-thesis/HumanML3D/HumanML3D \
  --run_name drppo_data_failure_00013_lr_\
  --train_split train \
  --eval_split test \
  --train_sampling_mode failure_only \
  --failure_cases_file /home/gxy/hay-thesis/PHC/output/eval_mdm/mdm_eval_train_full/falling_cases.txt \
  --failure_eval_cases_file /home/gxy/hay-thesis/PHC/output/eval_mdm/mdm_eval_v7_full/falling_cases.txt \
  --max_motion_frames 196 \
  --prompt_batch_size 256 \
  --ppo_minibatch_size 256 \
  --critic_minibatch_size 256 \
  --actor_num_epochs 2 \
  --critic_num_epochs 2 \
  --num_controlled_reverse_steps 5 \
  --sigma_embed_dim 128 \
  --reverse_policy_hidden_dim 256 \
  --reverse_policy_num_layers 5 \
  --reverse_log_std_init 0.0 \
  --reverse_noise_prior_kl_coef 0.01 \
  --denoising_discount 0.95 \
  --entropy_coef 0.0 \
  --lr_schedule cosine \
  --lr 1e-4 \
  --min_lr 1e-5 \
  --critic_lr_schedule cosine \
  --critic_lr 2e-4 \
  --critic_min_lr 1e-6 \
  --clip_range 1e-1 \
  --guidance_param 2.5 \
  --value_target_norm popart \
  --popart_beta 0.01 \
  --popart_epsilon 1e-5 \
  --phc_num_envs 256 \
  --phc_max_steps 400 \
  --num_outer_steps 100 \
  --eval_interval 10 \
  --eval_prompt_batch_size 256 \
  --save_root /home/gxy/hay-thesis/motion-diffusion-model-phc/save \
  --save_interval 10 \
  --log_interval 5 \
  --wandb_mode online \
  --fixed_sampling_seed \
  --fail_penalty -1.0 \
  --reward_norm \
  --dense_reward_weight 5.0 \
  --pose_reward_weight 1.0 \
  --pose_reward_alpha 5 \
  --velocity_reward_weight 0.0 \
  --velocity_reward_alpha 0.5 \
  --frame_gamma 0.99 \
  --frame_lambda 0.9
'''
if __name__ == "__main__":
    main()
