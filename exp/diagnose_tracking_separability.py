import argparse
import sys
from pathlib import Path

try:  # Isaac Gym must be imported before torch in this process.
    import isaacgym  # noqa: F401
except ImportError:
    isaacgym = None

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from diagnostic_common import (
    build_runtime,
    ensure_dir,
    per_sample_metrics,
    pooled_frame_features,
    sample_and_evaluate_batch,
    save_json,
    set_seed,
)

REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.dppo_frame_rl.data import build_prompt_entry_pools, sample_prompt_batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--data_root", required=True, type=str)
    parser.add_argument("--output_root", default="output/diagnostics/separability", type=str)
    parser.add_argument("--run_name", required=True, type=str)

    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--sampling_mode", choices=["uniform", "failure_only", "mixed"], default="uniform")
    parser.add_argument("--failure_cases_file", default="", type=str)
    parser.add_argument("--num_prompts", default=8, type=int)
    parser.add_argument("--samples_per_prompt", default=8, type=int)

    parser.add_argument("--device_id", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--max_motion_frames", default=196, type=int)
    parser.add_argument("--guidance_param", default=2.5, type=float)
    parser.add_argument("--ft_denoising_steps", default=4, type=int)
    parser.add_argument("--gamma_denoising", default=0.995, type=float)
    parser.add_argument("--min_sampling_denoising_std", default=1e-2, type=float)
    parser.add_argument("--min_logprob_denoising_std", default=1e-2, type=float)
    parser.add_argument("--clip_range", default=5e-2, type=float)
    parser.add_argument("--lora_rank", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16.0, type=float)
    parser.add_argument("--lora_layer_scope", default="all", choices=["all", "last3"], type=str)
    parser.add_argument("--resume_lora_path", default="", type=str)

    parser.add_argument("--dense_reward_weight", default=1.0, type=float)
    parser.add_argument("--dense_reward_norm", action="store_true")
    parser.add_argument("--success_bonus", default=0.0, type=float)
    parser.add_argument("--fail_penalty", default=-5.0, type=float)
    parser.add_argument("--frame_gamma", default=0.995, type=float)
    parser.add_argument("--frame_lambda", default=0.95, type=float)

    parser.add_argument("--phc_num_envs", default=16, type=int)
    parser.add_argument("--phc_max_steps", default=420, type=int)
    parser.add_argument(
        "--phc_actor_ckpt",
        default="/home/gxy/hay-thesis/PHC/output/HumanoidIm/phc_kp_mcp_iccv/Humanoid.pth",
        type=str,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    rng = np.random.RandomState(args.seed)
    prompt_rng = __import__("random").Random(args.seed)
    output_dir = ensure_dir(Path(args.output_root).expanduser().resolve() / args.run_name)

    if args.sampling_mode != "uniform" and not args.failure_cases_file:
        raise ValueError("failure_cases_file is required when sampling_mode is not uniform.")

    if args.sampling_mode == "uniform":
        selected_prompts = sample_prompt_batch(
            pools=build_prompt_entry_pools(
                data_root=args.data_root,
                split=args.split,
                max_motion_frames=args.max_motion_frames,
                failure_cases_file="",
            ),
            mode="uniform",
            batch_size=args.num_prompts,
            rng=prompt_rng,
        )[0]
    else:
        selected_prompts = sample_prompt_batch(
            pools=build_prompt_entry_pools(
                data_root=args.data_root,
                split=args.split,
                max_motion_frames=args.max_motion_frames,
                failure_cases_file=args.failure_cases_file,
            ),
            mode=args.sampling_mode,
            batch_size=args.num_prompts,
            rng=prompt_rng,
            success_sampling_weight=1.0,
            failure_sampling_weight=1.0,
        )[0]

    runtime = build_runtime(args)
    all_features = []
    all_success = []
    all_terminate = []
    all_exec_ratio = []
    all_phc_return = []
    all_seq_reward = []
    all_meta = []

    try:
        ##############################################
        # Process prompts independently so each saved sample keeps
        # a clear prompt-local grouping for offline probes.
        ##############################################
        for prompt_idx, entry in enumerate(selected_prompts):
            batch_entries = [entry for _ in range(args.samples_per_prompt)]
            batch_seed = int(rng.randint(0, 2**31 - 1))
            batch = sample_and_evaluate_batch(
                runtime=runtime,
                entries=batch_entries,
                deterministic=args.deterministic,
                sampling_seed=batch_seed,
            )
            lengths_20fps = torch.tensor(
                [e.length_20fps for e in batch_entries],
                dtype=torch.long,
                device=batch.sampled["frame_features"].device,
            )
            features = pooled_frame_features(batch.sampled["frame_features"], lengths_20fps)
            metrics = per_sample_metrics(batch)

            all_features.append(features)
            all_success.append(metrics["success"])
            all_terminate.append(metrics["terminate"])
            all_exec_ratio.append(metrics["exec_ratio"])
            all_phc_return.append(metrics["phc_return_mean"])
            all_seq_reward.append(metrics["undiscounted_sequence_reward"])

            for sample_idx, episode in enumerate(batch.episodes):
                all_meta.append(
                    {
                        "prompt_index": prompt_idx,
                        "sample_index": sample_idx,
                        "sampling_seed": batch_seed,
                        "db_key": entry.db_key,
                        "caption_idx": entry.caption_idx,
                        "caption": entry.caption,
                        "length_20fps": entry.length_20fps,
                        "success": bool(episode.get("success", False)),
                        "exec_len": int(episode.get("exec_len", 0)),
                        "target_len": int(episode.get("target_len", 0)),
                    }
                )

            print(
                f"[prompt {prompt_idx + 1}/{len(selected_prompts)}] "
                f"success_rate={batch.summary_metrics['success_rate']:.4f} "
                f"phc_return_mean={batch.summary_metrics['phc_return_mean']:.4f} "
                f"exec_ratio_mean={batch.summary_metrics['exec_ratio_mean']:.4f}"
            )
    finally:
        runtime.close()

    features = np.concatenate(all_features, axis=0).astype(np.float32)
    success = np.concatenate(all_success, axis=0).astype(np.float32)
    terminate = np.concatenate(all_terminate, axis=0).astype(np.float32)
    exec_ratio = np.concatenate(all_exec_ratio, axis=0).astype(np.float32)
    phc_return = np.concatenate(all_phc_return, axis=0).astype(np.float32)
    seq_reward = np.concatenate(all_seq_reward, axis=0).astype(np.float32)

    np.save(output_dir / "features.npy", features)
    np.save(output_dir / "success.npy", success)
    np.save(output_dir / "terminate.npy", terminate)
    np.save(output_dir / "exec_ratio.npy", exec_ratio)
    np.save(output_dir / "phc_return_mean.npy", phc_return)
    np.save(output_dir / "undiscounted_sequence_reward.npy", seq_reward)
    save_json(
        output_dir / "meta.json",
        {
            "run_name": args.run_name,
            "num_prompts": args.num_prompts,
            "samples_per_prompt": args.samples_per_prompt,
            "feature_dim": int(features.shape[1]),
            "num_samples": int(features.shape[0]),
            "meta": all_meta,
        },
    )
    print(f"Saved separability diagnostics to: {output_dir}")

'''
python exp/diagnose_tracking_separability.py \
  --model_path /home/gxy/hay-thesis/motion-diffusion-model-phc/save/humanml_enc_512_50steps/model000750000.pt \
  --data_root /home/gxy/hay-thesis/HumanML3D/HumanML3D \
  --output_root /home/gxy/hay-thesis/motion-diffusion-model-phc/output/diagnostics/separability \
  --run_name sep_train_failure_probe \
  --split train \
  --sampling_mode failure_only \
  --failure_cases_file /home/gxy/hay-thesis/PHC/output/eval_mdm/mdm_eval_train_full/falling_cases.txt \
  --num_prompts 128 \
  --samples_per_prompt 8 \
  --device_id 0 \
  --seed 0 \
  --max_motion_frames 196 \
  --guidance_param 2.5 \
  --ft_denoising_steps 4 \
  --clip_range 5e-2 \
  --phc_num_envs 32 \
  --phc_max_steps 420 \
  --dense_reward_norm
'''
if __name__ == "__main__":
    main()
