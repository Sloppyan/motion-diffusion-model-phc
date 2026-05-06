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
from train.dsppo_rl.noise_policy_mean import FrameNoiseMeanPolicy
from train.dsppo_rl.ppo_inversion import compute_noise_policy_inversion_loss
from train.dsppo_rl.runtime_ddim import DSPPODDIMInversionRuntime
from train.train_dsppo_phc import (
    maybe_init_wandb,
    normalize_advantages,
    prefix_metrics,
    resolve_scheduled_lr,
    save_checkpoint,
    select_best_metric,
    set_optimizer_lr,
    set_seed,
    train_critic,
)


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
    parser.add_argument("--max_grad_norm", default=1.0, type=float)

    parser.add_argument("--guidance_param", default=2.5, type=float)
    parser.add_argument("--noise_hidden_dim", default=256, type=int)
    parser.add_argument("--inversion_loss_coef", default=0.0, type=float)
    parser.add_argument("--ddim_eta", default=0.0, type=float)
    parser.add_argument("--deterministic_eval_policy", action="store_true")

    parser.add_argument("--frame_gamma", default=0.995, type=float)
    parser.add_argument("--frame_lambda", default=0.95, type=float)
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
    parser.add_argument("--wandb_project", default="mdm-phc-dsppo-ddim-inversion", type=str)
    return parser.parse_args()


def train_actor(args, noise_policy, actor_optimizer, rollout) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    num_updates = 0

    for _ in range(args.actor_num_epochs):
        permutation = torch.randperm(len(rollout.texts), device=rollout.frame_noise.device)
        for start in range(0, len(rollout.texts), args.ppo_minibatch_size):
            batch_inds = permutation[start : start + args.ppo_minibatch_size]
            loss_dict = compute_noise_policy_inversion_loss(
                noise_policy=noise_policy,
                text_embeds=rollout.text_embeds[batch_inds],
                frame_noise=rollout.frame_noise[batch_inds],
                frame_noise_base=rollout.frame_noise_base[batch_inds],
                frame_noise_gt_inv=rollout.frame_noise_gt_inv[batch_inds],
                frame_logprobs_old=rollout.frame_logprobs_old[batch_inds],
                frame_advantages=rollout.frame_advantages[batch_inds],
                frame_exec_mask=rollout.frame_exec_mask[batch_inds],
                clip_range=args.clip_range,
                inversion_loss_coef=args.inversion_loss_coef,
            )

            actor_optimizer.zero_grad()
            loss_dict["loss"].backward()
            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(noise_policy.parameters(), args.max_grad_norm)
            actor_optimizer.step()

            with torch.no_grad():
                for key, value in loss_dict.items():
                    metric_key = "total_loss" if key == "loss" else key
                    metrics[metric_key] = metrics.get(metric_key, 0.0) + float(value.item())
                num_updates += 1

    if num_updates == 0:
        return metrics
    return {key: value / num_updates for key, value in metrics.items()}


def main():
    args = parse_args()
    if float(args.ddim_eta) != 0.0:
        raise ValueError("Only deterministic DDIM is supported in this script; set --ddim_eta 0.")

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

    runtime = DSPPODDIMInversionRuntime(args)
    noise_policy = FrameNoiseMeanPolicy(
        text_dim=runtime.actor.clip_dim,
        hidden_dim=args.noise_hidden_dim,
        action_dim=runtime.actor.njoints * runtime.actor.nfeats,
        max_frames=args.max_motion_frames,
    ).to(runtime.device)
    critic = FrameCritic(
        max_frames=args.max_motion_frames,
        value_target_norm=args.value_target_norm,
        popart_beta=args.popart_beta,
        popart_epsilon=args.popart_epsilon,
    ).to(runtime.device)

    actor_optimizer = AdamW(noise_policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
                noise_policy=noise_policy,
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
            rollout_metrics = merge_metrics(
                rollout_metrics,
                {
                    "frame_adv_mean": float(norm_adv["adv_mean"].item()),
                    "frame_adv_std": float(norm_adv["adv_std"].item()),
                },
            )

            critic_metrics = train_critic(args, critic, critic_optimizer, rollout)
            actor_metrics = train_actor(args, noise_policy, actor_optimizer, rollout)
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
                wandb_run.log({**train_metrics, "outer_step": outer_step}, step=outer_step)

            if outer_step % args.eval_interval == 0:
                eval_batch = select_eval_batch(eval_entries, args.eval_prompt_batch_size, offset=0)
                eval_metrics = prefix_metrics(
                    "eval/",
                    runtime.evaluate_entries(
                        entries=eval_batch,
                        noise_policy=noise_policy,
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
                            noise_policy=noise_policy,
                            policy_deterministic=args.deterministic_eval_policy,
                            sampling_seed=args.seed if args.fixed_sampling_seed else None,
                        ),
                    )
                    merged_eval.update(failure_metrics)

                print(f"[eval] step={outer_step} " + " ".join(f"{k}={v:.4f}" for k, v in sorted(merged_eval.items())))
                if wandb_run is not None:
                    wandb_run.log({**merged_eval, "outer_step": outer_step}, step=outer_step)

                best_metric = select_best_metric(merged_eval)
                if best_metric is not None:
                    _, metric_value = best_metric
                    if metric_value > best_eval_score:
                        best_eval_score = metric_value
                        save_checkpoint(
                            save_dir=save_dir,
                            filename="best.pt",
                            step=outer_step,
                            noise_policy=noise_policy,
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
                    noise_policy=noise_policy,
                    critic=critic,
                    actor_optimizer=actor_optimizer,
                    critic_optimizer=critic_optimizer,
                    args=args,
                )
    finally:
        runtime.close()
        if wandb_run is not None:
            wandb_run.finish()

'''
python train/train_dsppo_ddim_inversion_phc.py \
  --model_path /home/gxy/hay-thesis/motion-diffusion-model-phc/save/humanml_enc_512_50steps/model000750000.pt \
  --data_root /home/gxy/hay-thesis/HumanML3D/HumanML3D \
  --run_name dsppo_data_failure_0038_DDIM_prior_kl \
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
  --inversion_loss_coef 0 \
  --noise_prior_kl_coef 1e-5 \
  --ddim_eta 0.0 \
  --pose_reward_weight 1.0 \
  --pose_reward_alpha 5 \
  --lr_schedule cosine \
  --lr 1e-4 \
  --min_lr 1e-5 \
  --critic_lr_schedule cosine \
  --critic_lr 1e-4 \
  --critic_min_lr 1e-6 \
  --clip_range 2e-1 \
  --guidance_param 2.5 \
  --noise_hidden_dim 256 \
  --value_target_norm popart \
  --popart_beta 0.005 \
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
  --deterministic_eval_policy \
  --fail_penalty -0.2 \
  --reward_norm \
  --dense_reward_weight 1.0 \
  --velocity_reward_weight 0.0 \
  --velocity_reward_alpha 0.5
'''
if __name__ == "__main__":
    main()

