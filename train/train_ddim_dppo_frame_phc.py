import json
import random
import sys
from pathlib import Path
from typing import Optional, Sequence

try:  # Isaac Gym must be imported before torch in this process.
    import isaacgym  # noqa: F401
except ImportError:
    isaacgym = None

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
from train.dppo_frame_rl.runtime_ddim import FrameDDIMDPPORuntime
from train.train_dppo_frame_phc import (
    maybe_init_wandb,
    normalize_advantages,
    parse_args,
    prefix_metrics,
    resolve_scheduled_lr,
    save_checkpoint,
    select_best_metric,
    set_optimizer_lr,
    set_seed,
    train_actor,
    train_critic,
)


def main():
    args = parse_args()
    if args.wandb_project == "mdm-phc-dppo-frame":
        args.wandb_project = "mdm-phc-ddim-dppo-frame"
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

    runtime = FrameDDIMDPPORuntime(args)
    critic = FrameCritic(
        max_frames=args.max_motion_frames,
        value_target_norm=args.value_target_norm,
        popart_beta=args.popart_beta,
        popart_epsilon=args.popart_epsilon,
    ).to(runtime.device)

    from torch.optim import AdamW

    actor_optimizer = AdamW(list(runtime.policy.trainable_parameters()), lr=args.lr, weight_decay=args.weight_decay)
    critic_optimizer = AdamW(critic.parameters(), lr=args.critic_lr, weight_decay=args.weight_decay)
    wandb_run = maybe_init_wandb(args)

    rng = random.Random(args.seed)
    best_eval_score = float("-inf")
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
            ##############################################
            # DDIM rollout is deterministic by construction.
            # The only randomness left is the initial x_T.
            ##############################################
            rollout, rollout_metrics = runtime.collect_rollout(
                entries=batch_entries,
                critic=critic,
                deterministic=True,
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
                import wandb

                wandb.log({**train_metrics, "outer_step": outer_step}, step=outer_step)

            if outer_step % args.eval_interval == 0:
                eval_batch = select_eval_batch(eval_entries, args.eval_prompt_batch_size, offset=0)
                eval_metrics = prefix_metrics(
                    "eval/",
                    runtime.evaluate_entries(
                        entries=eval_batch,
                        deterministic=True,
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
                            deterministic=True,
                            sampling_seed=args.seed if args.fixed_sampling_seed else None,
                        ),
                    )
                    merged_eval.update(failure_metrics)

                print(f"[eval] step={outer_step} " + " ".join(f"{k}={v:.4f}" for k, v in sorted(merged_eval.items())))
                if wandb_run is not None:
                    import wandb

                    wandb.log({**merged_eval, "outer_step": outer_step}, step=outer_step)

                best_metric = select_best_metric(merged_eval)
                if best_metric is not None:
                    metric_name, metric_value = best_metric
                    if metric_value > best_eval_score:
                        best_eval_score = metric_value
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
            import wandb

            wandb.finish()


if __name__ == "__main__":
    main()
