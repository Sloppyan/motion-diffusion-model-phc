import argparse
import sys
from pathlib import Path

try:  # Isaac Gym must be imported before torch in this process.
    import isaacgym  # noqa: F401
except ImportError:
    isaacgym = None

import joblib
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from diagnostic_common import (
    build_runtime,
    ensure_dir,
    per_sample_metrics,
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
    parser.add_argument("--output_root", default="output/diagnostics/reachability", type=str)
    parser.add_argument("--run_name", required=True, type=str)

    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--sampling_mode", choices=["uniform", "failure_only", "mixed"], default="uniform")
    parser.add_argument("--failure_cases_file", default="", type=str)
    parser.add_argument("--num_prompts", default=8, type=int)
    parser.add_argument("--samples_per_prompt", default=16, type=int)
    parser.add_argument("--interpolation_steps", default=9, type=int)

    parser.add_argument("--device_id", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--stochastic", action="store_true")
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


def choose_pair(success: np.ndarray, phc_return: np.ndarray):
    success_idx = np.where(success > 0.5)[0]
    fail_idx = np.where(success <= 0.5)[0]
    if success_idx.size == 0 or fail_idx.size == 0:
        return None
    best_success = int(success_idx[np.argmax(phc_return[success_idx])])
    worst_failure = int(fail_idx[np.argmin(phc_return[fail_idx])])
    return worst_failure, best_success


def main():
    args = parse_args()
    set_seed(args.seed)
    deterministic = not args.stochastic

    prompt_rng = __import__("random").Random(args.seed)
    rng = np.random.RandomState(args.seed)
    output_dir = ensure_dir(Path(args.output_root).expanduser().resolve() / args.run_name)

    if args.sampling_mode != "uniform" and not args.failure_cases_file:
        raise ValueError("failure_cases_file is required when sampling_mode is not uniform.")

    selected_prompts = sample_prompt_batch(
        pools=build_prompt_entry_pools(
            data_root=args.data_root,
            split=args.split,
            max_motion_frames=args.max_motion_frames,
            failure_cases_file=args.failure_cases_file if args.sampling_mode != "uniform" else "",
        ),
        mode=args.sampling_mode,
        batch_size=args.num_prompts,
        rng=prompt_rng,
        success_sampling_weight=1.0,
        failure_sampling_weight=1.0,
    )[0]

    runtime = build_runtime(args)
    all_episodes = []
    path_metrics = []
    skipped_prompts = 0

    try:
        ##############################################
        # For each prompt:
        # 1) sample a candidate set
        # 2) choose one failure and one success exemplar
        # 3) interpolate their initial noise x_T
        # 4) re-run PHC tracking along the path
        ##############################################
        for prompt_idx, entry in enumerate(selected_prompts):
            candidate_entries = [entry for _ in range(args.samples_per_prompt)]
            candidate_seed = int(rng.randint(0, 2**31 - 1))
            candidate_batch = sample_and_evaluate_batch(
                runtime=runtime,
                entries=candidate_entries,
                deterministic=deterministic,
                sampling_seed=candidate_seed,
            )
            candidate_metrics = per_sample_metrics(candidate_batch)
            pair = choose_pair(candidate_metrics["success"], candidate_metrics["phc_return_mean"])
            if pair is None:
                skipped_prompts += 1
                print(f"[prompt {prompt_idx + 1}/{len(selected_prompts)}] skipped: no success-failure pair found")
                continue

            fail_idx, success_idx = pair
            fail_noise = candidate_batch.sampled["initial_noise"][fail_idx : fail_idx + 1]
            success_noise = candidate_batch.sampled["initial_noise"][success_idx : success_idx + 1]

            alphas = np.linspace(0.0, 1.0, num=max(2, args.interpolation_steps), dtype=np.float32)
            interp_noises = []
            interp_entries = []
            for alpha in alphas:
                noise = (1.0 - float(alpha)) * fail_noise + float(alpha) * success_noise
                interp_noises.append(noise)
                interp_entries.append(entry)
            interp_noise_batch = torch.cat(interp_noises, dim=0)

            path_batch = sample_and_evaluate_batch(
                runtime=runtime,
                entries=interp_entries,
                deterministic=deterministic,
                initial_noise=interp_noise_batch,
            )
            path_sample_metrics = per_sample_metrics(path_batch)

            for alpha_idx, (alpha, episode) in enumerate(zip(alphas.tolist(), path_batch.episodes)):
                all_episodes.append(
                    {
                        "prompt_index": prompt_idx,
                        "alpha": alpha,
                        "endpoint": (
                            "failure" if alpha_idx == 0 else "success" if alpha_idx == len(alphas) - 1 else "intermediate"
                        ),
                        "db_key": entry.db_key,
                        "caption_idx": entry.caption_idx,
                        "caption": entry.caption,
                        "episode": episode,
                    }
                )

            path_metrics.append(
                {
                    "prompt_index": prompt_idx,
                    "db_key": entry.db_key,
                    "caption_idx": entry.caption_idx,
                    "caption": entry.caption,
                    "candidate_sampling_seed": candidate_seed,
                    "failure_candidate_index": int(fail_idx),
                    "success_candidate_index": int(success_idx),
                    "alphas": alphas.tolist(),
                    "success": path_sample_metrics["success"].tolist(),
                    "exec_ratio": path_sample_metrics["exec_ratio"].tolist(),
                    "phc_return_mean": path_sample_metrics["phc_return_mean"].tolist(),
                    "undiscounted_sequence_reward": path_sample_metrics["undiscounted_sequence_reward"].tolist(),
                }
            )
            print(
                f"[prompt {prompt_idx + 1}/{len(selected_prompts)}] "
                f"path_success={path_sample_metrics['success'].tolist()} "
                f"path_return={path_sample_metrics['phc_return_mean'].tolist()}"
            )
    finally:
        runtime.close()

    joblib.dump(all_episodes, output_dir / "episodes.pkl")
    save_json(
        output_dir / "path_metrics.json",
        {
            "run_name": args.run_name,
            "num_prompts": args.num_prompts,
            "samples_per_prompt": args.samples_per_prompt,
            "interpolation_steps": args.interpolation_steps,
            "num_paths": len(path_metrics),
            "skipped_prompts": skipped_prompts,
            "paths": path_metrics,
        },
    )
    save_json(
        output_dir / "meta.json",
        {
            "run_name": args.run_name,
            "split": args.split,
            "sampling_mode": args.sampling_mode,
            "deterministic": bool(deterministic),
            "num_prompts": args.num_prompts,
            "samples_per_prompt": args.samples_per_prompt,
            "interpolation_steps": args.interpolation_steps,
        },
    )
    print(f"Saved reachability diagnostics to: {output_dir}")

'''
python exp/diagnose_tracking_reachability.py \
  --model_path /home/gxy/hay-thesis/motion-diffusion-model-phc/save/humanml_enc_512_50steps/model000750000.pt \
  --data_root /home/gxy/hay-thesis/HumanML3D/HumanML3D \
  --output_root /home/gxy/hay-thesis/motion-diffusion-model-phc/output/diagnostics/reachability \
  --run_name reach_train_failure_probe \
  --split train \
  --sampling_mode failure_only \
  --failure_cases_file /home/gxy/hay-thesis/PHC/output/eval_mdm/mdm_eval_train_full/falling_cases.txt \
  --num_prompts 64 \
  --samples_per_prompt 16 \
  --interpolation_steps 9 \
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
