import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import isaacgym
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from model.lora_attention import iter_lora_parameters
from train.ddpo_rl.data import (
    FailureOnlyBatchSampler,
    HumanMLPromptDataset,
    MixedBatchSampler,
    PromptSampleDataset,
    WithoutReplacementBatchSampler,
    filter_prompt_samples_by_keys,
    load_failure_case_keys,
)
from train.ddpo_rl.reward import build_reward_spec
from train.ddpo_rl import build_algorithm
from train.ddpo_rl.logging import format_debug_entries, summarize_rollout
from train.ddpo_rl.runtime import DDPORuntime, init_wandb, load_policy, save_checkpoint, set_seed
from utils.ddpo_parser import parse_ddpo_args


def _build_failure_dataset(args, base_dataset):
    failure_split = args.failure_cases_split
    if failure_split == "train":
        return base_dataset
    return HumanMLPromptDataset(
        data_root=args.data_root,
        split=failure_split,
        max_motion_frames=args.max_motion_frames,
    )


def _build_train_sampler(args, dataset, rng):
    if args.train_sampling_mode == "uniform":
        sampler = WithoutReplacementBatchSampler(dataset, args.prompt_batch_size, rng, shuffle=True)
        return sampler, 0, "train"

    ##############################
    # Build the failure pool from an explicit split so failure-only recovery
    # can train on mined hard cases without forcing them to live in train.txt.
    ##############################
    failure_dataset_source = _build_failure_dataset(args, dataset)
    failure_keys = load_failure_case_keys(args.failure_cases_file)
    failure_samples = filter_prompt_samples_by_keys(failure_dataset_source.samples, failure_keys)
    if len(failure_samples) == 0:
        raise RuntimeError(
            "No failure cases from "
            f"{args.failure_cases_file} matched samples in {failure_dataset_source.split_path}."
        )

    failure_dataset = PromptSampleDataset(failure_samples)
    if args.train_sampling_mode == "failure_only":
        sampler = FailureOnlyBatchSampler(failure_dataset, args.prompt_batch_size, rng, shuffle=True)
        return sampler, len(failure_dataset), args.failure_cases_split
    if args.train_sampling_mode == "mixed":
        sampler = MixedBatchSampler(
            base_dataset=dataset,
            failure_dataset=failure_dataset,
            batch_size=args.prompt_batch_size,
            failure_batch_size=args.failure_batch_size,
            rng=rng,
            shuffle=True,
        )
        return sampler, len(failure_dataset), args.failure_cases_split
    raise ValueError(f"Unsupported train_sampling_mode: {args.train_sampling_mode}")


def _build_eval_batches(args):
    if args.eval_interval <= 0:
        return None, None, None

    eval_dataset = HumanMLPromptDataset(
        data_root=args.data_root,
        split=args.eval_split,
        max_motion_frames=args.max_motion_frames,
    )
    eval_batch_size = min(len(eval_dataset), int(args.eval_prompt_batch_size))
    eval_samples = eval_dataset.first_batch(eval_batch_size)

    eval_failure_samples = None
    if args.failure_eval_cases_file:
        failure_keys = load_failure_case_keys(args.failure_eval_cases_file)
        failure_samples = filter_prompt_samples_by_keys(eval_dataset.samples, failure_keys)
        if len(failure_samples) == 0:
            raise RuntimeError(
                f"No failure eval cases from {args.failure_eval_cases_file} matched samples in {eval_dataset.split_path}."
            )
        failure_dataset = PromptSampleDataset(failure_samples)
        failure_batch_size = min(len(failure_dataset), int(args.eval_prompt_batch_size))
        eval_failure_samples = failure_dataset.first_batch(failure_batch_size)

    eval_args = SimpleNamespace(**vars(args))
    eval_args.fixed_sampling_seed = True
    eval_args.seed = args.seed + 100000
    return eval_samples, eval_failure_samples, eval_args


def main(argv=None):
    #######
    # Parse args and build the top-level training context.
    #######
    args = parse_ddpo_args(argv)
    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    set_seed(args.seed)
    reward_spec = build_reward_spec(args)

    #######
    # Initialize logging and the prompt datasets used for train/eval.
    #######
    run = init_wandb(args)
    dataset = HumanMLPromptDataset(
        data_root=args.data_root,
        split="train",
        max_samples=args.max_train_samples,
        max_motion_frames=args.max_motion_frames,
    )
    rng = np.random.RandomState(args.seed)
    train_sampler, failure_train_size, failure_train_split = _build_train_sampler(args, dataset, rng)
    steps_per_epoch = int(
        getattr(
            train_sampler,
            "steps_per_epoch",
            max(1, int(math.ceil(len(dataset) / float(args.prompt_batch_size)))),
        )
    )
    total_epochs = max(1, int(math.ceil(args.num_outer_steps / float(steps_per_epoch))))
    print(
        json.dumps(
            {
                "train_sampling_mode": args.train_sampling_mode,
                "base_train_samples": len(dataset),
                "failure_train_samples": int(failure_train_size),
                "failure_train_split": failure_train_split,
                "steps_per_epoch": int(steps_per_epoch),
            },
            ensure_ascii=False,
        )
    )

    #######
    # Freeze a deterministic eval batch so checkpoints are compared on the same prompts.
    #######
    eval_samples, eval_failure_samples, eval_args = _build_eval_batches(args)
    if eval_failure_samples is not None:
        print(
            json.dumps(
                {
                    "failure_eval_samples": len(eval_failure_samples),
                    "failure_eval_cases_file": args.failure_eval_cases_file,
                },
                ensure_ascii=False,
            )
        )

    #######
    # Build the actor, the optional reference actor, and the shared runtime.
    #######
    model, diffusion = load_policy(args, device, trainable=True)
    reference_model = None
    if args.kl_coef > 0.0 or args.ft_denoising_steps > 0:
        reference_model, _ = load_policy(args, device, trainable=False)

    trainable_params = list(iter_lora_parameters(model))
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    runtime = DDPORuntime(
        args=args,
        device=device,
        model=model,
        diffusion=diffusion,
        reward_spec=reward_spec,
        reference_model=reference_model,
    )
    algorithm = build_algorithm(
        args.algo,
        model=model,
        diffusion=diffusion,
        optimizer=optimizer,
        trainable_params=trainable_params,
        device=device,
        args=args,
        reference_model=reference_model,
    )

    #######
    # Outer loop: sample prompts, run PHC rollouts, update the chosen RL algorithm.
    #######
    best_reward = -float("inf")
    for outer_step in range(args.num_outer_steps):
        prompt_batch = train_sampler.next_batch()
        rollout = runtime.collect_rollout_batch(
            samples=prompt_batch,
            collect_hidden=algorithm.requires_hidden_features,
        )

        metrics = summarize_rollout(rollout, reward_spec)
        metrics.update(algorithm.on_rollout_end(rollout))
        metrics.update(algorithm.update(rollout))
        metrics["algo"] = args.algo

        #######
        # Run periodic eval with fixed prompts and a fixed sampling seed.
        #######
        if eval_samples is not None and (outer_step + 1) % args.eval_interval == 0:
            with torch.no_grad():
                eval_rollout = runtime.collect_rollout_batch(
                    samples=eval_samples,
                    sampling_args=eval_args,
                    collect_hidden=False,
                )
                if eval_failure_samples is not None:
                    eval_failure_rollout = runtime.collect_rollout_batch(
                        samples=eval_failure_samples,
                        sampling_args=eval_args,
                        collect_hidden=False,
                    )
            metrics.update(summarize_rollout(eval_rollout, reward_spec, prefix="eval/"))
            if eval_failure_samples is not None:
                metrics.update(summarize_rollout(eval_failure_rollout, reward_spec, prefix="eval_failure/"))

        #######
        # Emit logs after the update so train metrics reflect the full step.
        #######
        if run is not None:
            run.log(metrics, step=outer_step)
        if outer_step % args.log_interval == 0:
            step_now = outer_step + 1
            epoch_now = train_sampler.epoch + 1
            progress_prefix = f"[step {step_now}/{args.num_outer_steps} | epoch {epoch_now}/{total_epochs}] "
            print(progress_prefix + json.dumps(metrics, ensure_ascii=False))
            if args.debug_prompt_advantages:
                entries = algorithm.debug_entries(rollout, args.debug_prompt_advantages_limit)
                if entries:
                    print(format_debug_entries(entries))

        #######
        # Save both the best checkpoint so far and periodic snapshots.
        #######
        reward_mean = metrics["reward_mean"]
        is_best = reward_mean > best_reward
        if is_best:
            best_reward = reward_mean
            save_checkpoint(
                save_dir,
                outer_step,
                model,
                optimizer,
                algorithm_state=algorithm.state_dict(),
                best=True,
            )
        if (outer_step + 1) % args.save_interval == 0:
            save_checkpoint(
                save_dir,
                outer_step,
                model,
                optimizer,
                algorithm_state=algorithm.state_dict(),
                best=False,
            )

    #######
    # Always write the final checkpoint, even if it is not the best one.
    #######
    save_checkpoint(
        save_dir,
        args.num_outer_steps - 1,
        model,
        optimizer,
        algorithm_state=algorithm.state_dict(),
        best=False,
    )
    if run is not None:
        run.finish()

'''
# small sample exp for raw-ppo
python train/train_ddpo_phc_rl.py \
  --model_path /home/gxy/hay-thesis/motion-diffusion-model-phc/save/humanml_enc_512_50steps/model000750000.pt \
  --data_root /home/gxy/hay-thesis/HumanML3D/HumanML3D \
  --run_name ddpo_phc_actor_critic_failure_only_0005_with_KL_1_ft_4 \
  --ft_denoising_steps 4 \
  --algo actor_critic \
  --kl_coef 1 \
  --reward_assignment sequence \
  --reward_mode failure \
  --reward_success_weight 2.0 \
  --reward_fail_penalty 2.0 \
  --train_sampling_mode failure_only \
  --failure_cases_file /home/gxy/hay-thesis/PHC/output/eval_mdm/mdm_eval_train_full/falling_cases.txt \
  --failure_cases_split train \
  --failure_eval_cases_file /home/gxy/hay-thesis/PHC/output/eval_mdm/mdm_eval_v7_full/falling_cases.txt \
  --max_motion_frames 196 \
  --prompt_batch_size 16 \
  --ppo_minibatch_size 8 \
  --critic_minibatch_size 128 \
  --actor_num_epochs 1 \
  --critic_num_epochs 2 \
  --lr 3e-5 \
  --critic_lr 3e-4 \
  --clip_range 1e-2 \
  --gae_gamma 0.99 \
  --gae_lambda 0.95 \
  --phc_num_envs 16 \
  --phc_max_steps 300 \
  --num_outer_steps 200 \
  --eval_interval 2 \
  --eval_prompt_batch_size 4 \
  --eval_split test \
  --save_interval 10 \
  --log_interval 1 \
  --wandb_project mdm-phc-ddpo

# small sample exp for chunk-level-ppo
python train/train_ddpo_phc_rl.py \
  --model_path /home/gxy/hay-thesis/motion-diffusion-model-phc/save/humanml_enc_512_50steps/model000750000.pt \
  --data_root /home/gxy/hay-thesis/HumanML3D/HumanML3D \
  --run_name ddpo_phc_actor_critic_failure_only_0006_chunk_level_with_KL_1 \
  --algo actor_critic \
  --kl_coef 1.0 \
  --reward_mode failure \
  --reward_success_weight 2.0 \
  --reward_fail_penalty 2.0 \
  --reward_assignment chunk \
  --reward_chunk_size 14 \
  --train_sampling_mode failure_only \
  --failure_cases_file /home/gxy/hay-thesis/PHC/output/eval_mdm/mdm_eval_train_full/falling_cases.txt \
  --failure_cases_split train \
  --failure_eval_cases_file /home/gxy/hay-thesis/PHC/output/eval_mdm/mdm_eval_v7_full/falling_cases.txt \
  --max_motion_frames 196 \
  --prompt_batch_size 16 \
  --ppo_minibatch_size 8 \
  --critic_minibatch_size 128 \
  --actor_num_epochs 1 \
  --critic_num_epochs 2 \
  --lr 3e-5 \
  --critic_lr 3e-4 \
  --clip_range 1e-2 \
  --gae_gamma 0.99 \
  --gae_lambda 0.95 \
  --phc_num_envs 16 \
  --phc_max_steps 300 \
  --num_outer_steps 200 \
  --eval_interval 2 \
  --eval_prompt_batch_size 4 \
  --eval_split test \
  --save_interval 10 \
  --log_interval 1 \
  --wandb_project mdm-phc-ddpo

  # real chunk level reward
  python train/train_ddpo_phc_rl.py \
    --chunk_early_weight 0 \
  --chunk_early_prev_weight 0 \
  --ft_denoising_steps 4 \
  --model_path /home/gxy/hay-thesis/motion-diffusion-model-phc/save/humanml_enc_512_50steps/model000750000.pt \
  --data_root /home/gxy/hay-thesis/HumanML3D/HumanML3D \
  --run_name ddpo_phc_actor_critic_failure_only_00013_chunk_reward_with_KL_0.001_only_base_reward_k_steps_4 \
  --algo actor_critic \
  --kl_coef 0.001 \
  --reward_mode failure \
  --reward_assignment chunk \
  --reward_chunk_size 14 \
  --chunk_mean_weight 20.0 \
  --chunk_q10_weight 8.0 \
  --train_sampling_mode failure_only \
  --failure_cases_file /home/gxy/hay-thesis/PHC/output/eval_mdm/mdm_eval_train_full/falling_cases.txt \
  --failure_cases_split train \
  --failure_eval_cases_file /home/gxy/hay-thesis/PHC/output/eval_mdm/mdm_eval_v7_full/falling_cases.txt \
  --max_motion_frames 196 \
  --prompt_batch_size 16 \
  --ppo_minibatch_size 8 \
  --critic_minibatch_size 128 \
  --actor_num_epochs 1 \
  --critic_num_epochs 2 \
  --lr 3e-5 \
  --critic_lr 3e-4 \
  --clip_range 2e-2 \
  --gae_gamma 0.99 \
  --gae_lambda 0.95 \
  --phc_num_envs 16 \
  --phc_max_steps 300 \
  --num_outer_steps 280 \
  --eval_interval 2 \
  --eval_prompt_batch_size 4 \
  --eval_split test \
  --save_interval 10 \
  --log_interval 1 \
  --wandb_project mdm-phc-ddpo

python train/train_ddpo_phc_rl.py \
  --model_path /home/gxy/hay-thesis/motion-diffusion-model-phc/save/humanml_enc_512_50steps/model000750000.pt \
  --data_root /home/gxy/hay-thesis/HumanML3D/HumanML3D \
  --run_name ddpo_phc_actor_critic_mixed_failure_v1 \
  --algo actor_critic \
  --reward_mode failure \
  --reward_success_weight 2.0 \
  --reward_fail_penalty 2.0 \
  --train_sampling_mode mixed \
  --failure_cases_file /home/gxy/hay-thesis/PHC/output/eval_mdm/mdm_eval_train_full/falling_cases.txt \
  --failure_cases_split train \
  --failure_batch_size 64 \
  --failure_eval_cases_file /home/gxy/hay-thesis/PHC/output/eval_mdm/mdm_eval_v7_full/falling_cases.txt \
  --max_motion_frames 196 \
  --prompt_batch_size 128 \
  --ppo_minibatch_size 32 \
  --critic_minibatch_size 1024 \
  --actor_num_epochs 1 \
  --critic_num_epochs 2 \
  --lr 3e-5 \
  --critic_lr 3e-4 \
  --clip_range 1e-2 \
  --gae_gamma 0.99 \
  --gae_lambda 0.95 \
  --phc_num_envs 128 \
  --phc_max_steps 420 \
  --num_outer_steps 160 \
  --eval_interval 5 \
  --eval_prompt_batch_size 8 \
  --eval_split test \
  --save_interval 20 \
  --log_interval 1 \
  --wandb_project mdm-phc-ddpo
'''
if __name__ == "__main__":
    main()
