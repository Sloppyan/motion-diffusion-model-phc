import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.dppo_frame_rl.data import PromptEntry, prompt_entries_to_meta
from train.dppo_frame_rl.logging import merge_metrics, summarize_episodes
from train.dppo_frame_rl.reward import FrameRewardBatch, build_frame_reward_batch
from train.dppo_frame_rl.runtime import FrameDPPORuntime


@dataclass
class DiagnosticBatch:
    entries: List[PromptEntry]
    sampled: Dict[str, torch.Tensor]
    episodes: List[Dict]
    reward_batch: FrameRewardBatch
    summary_metrics: Dict[str, float]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_runtime_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        device_id=int(args.device_id),
        max_motion_frames=int(args.max_motion_frames),
        model_path=str(args.model_path),
        guidance_param=float(args.guidance_param),
        ft_denoising_steps=int(args.ft_denoising_steps),
        min_sampling_denoising_std=float(args.min_sampling_denoising_std),
        min_logprob_denoising_std=float(args.min_logprob_denoising_std),
        lora_rank=int(args.lora_rank),
        lora_alpha=float(args.lora_alpha),
        lora_layer_scope=str(args.lora_layer_scope),
        resume_lora_path=str(args.resume_lora_path),
        gamma_denoising=float(args.gamma_denoising),
        clip_range=float(args.clip_range),
        clip_range_schedule="constant",
        clip_range_base=None,
        clip_range_rate=3.0,
        kl_coef=0.0,
        data_root=str(args.data_root),
        phc_num_envs=int(args.phc_num_envs),
        phc_max_steps=int(args.phc_max_steps),
        phc_actor_ckpt=str(args.phc_actor_ckpt),
        dense_reward_weight=float(args.dense_reward_weight),
        dense_reward_norm=bool(args.dense_reward_norm),
        success_bonus=float(args.success_bonus),
        fail_penalty=float(args.fail_penalty),
        frame_gamma=float(args.frame_gamma),
        frame_lambda=float(args.frame_lambda),
    )


def build_runtime(args) -> FrameDPPORuntime:
    runtime_args = build_runtime_args(args)
    return FrameDPPORuntime(runtime_args)


def sample_policy_batch(
    runtime: FrameDPPORuntime,
    entries: Sequence[PromptEntry],
    deterministic: bool = False,
    sampling_seed: Optional[int] = None,
    initial_noise: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    texts = [entry.caption for entry in entries]
    lengths_20fps = torch.tensor(
        [entry.length_20fps for entry in entries],
        dtype=torch.long,
        device=runtime.device,
    )

    if sampling_seed is None:
        return runtime.policy.sample(
            texts=texts,
            lengths_20fps=lengths_20fps,
            max_motion_frames=runtime.max_motion_frames,
            deterministic=deterministic,
            initial_noise=initial_noise,
        )

    ##############################################
    # Keep stochastic sampling reproducible without polluting
    # the caller's global RNG state.
    ##############################################
    devices = [] if runtime.device.index is None else [runtime.device.index]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(sampling_seed))
        return runtime.policy.sample(
            texts=texts,
            lengths_20fps=lengths_20fps,
            max_motion_frames=runtime.max_motion_frames,
            deterministic=deterministic,
            initial_noise=initial_noise,
        )


def evaluate_sampled_batch(
    runtime: FrameDPPORuntime,
    entries: Sequence[PromptEntry],
    sampled: Dict[str, torch.Tensor],
) -> DiagnosticBatch:
    lengths_20fps = torch.tensor(
        [entry.length_20fps for entry in entries],
        dtype=torch.long,
        device=runtime.device,
    )
    ref_motion_batch, lengths_30hz = runtime._sample_to_reference_batch(sampled["final_sample"], lengths_20fps)

    ##############################################
    # PHC external eval uses a rollout budget over the whole queued batch,
    # not a per-sample horizon. Diagnostic jobs often use small env counts
    # with long motions, so auto-expand the budget to cover the batch.
    ##############################################
    num_envs = max(1, int(runtime.args.phc_num_envs))
    max_len_30hz = max(lengths_30hz) if lengths_30hz else 0
    total_len_30hz = int(sum(lengths_30hz))
    required_steps = max(
        max_len_30hz,
        int(np.ceil(float(total_len_30hz) / float(num_envs))),
    ) + 8
    original_max_steps = runtime.phc_bridge.phc_max_steps
    runtime.phc_bridge.phc_max_steps = max(original_max_steps, required_steps)
    try:
        episodes = runtime.phc_bridge.evaluate_batch(
            ref_motion_batch=ref_motion_batch,
            lengths_30hz=lengths_30hz,
            meta=prompt_entries_to_meta(entries),
        )
    finally:
        runtime.phc_bridge.phc_max_steps = original_max_steps

    reward_batch = build_frame_reward_batch(
        episodes=episodes,
        max_motion_frames=runtime.max_motion_frames,
        gamma_frame=runtime.args.frame_gamma,
        dense_reward_weight=runtime.args.dense_reward_weight,
        dense_reward_norm=runtime.args.dense_reward_norm,
        success_bonus=runtime.args.success_bonus,
        fail_penalty=runtime.args.fail_penalty,
        device=runtime.device,
    )
    summary_metrics = merge_metrics(
        summarize_episodes(episodes),
        {
            "undiscounted_sequence_reward_mean": reward_batch.undiscounted_sequence_reward_mean,
            "exec_ratio_mean": reward_batch.exec_ratio_mean,
        },
    )
    return DiagnosticBatch(
        entries=list(entries),
        sampled=sampled,
        episodes=episodes,
        reward_batch=reward_batch,
        summary_metrics=summary_metrics,
    )


def sample_and_evaluate_batch(
    runtime: FrameDPPORuntime,
    entries: Sequence[PromptEntry],
    deterministic: bool = False,
    sampling_seed: Optional[int] = None,
    initial_noise: Optional[torch.Tensor] = None,
) -> DiagnosticBatch:
    sampled = sample_policy_batch(
        runtime=runtime,
        entries=entries,
        deterministic=deterministic,
        sampling_seed=sampling_seed,
        initial_noise=initial_noise,
    )
    return evaluate_sampled_batch(runtime=runtime, entries=entries, sampled=sampled)


def pooled_frame_features(frame_features: torch.Tensor, lengths_20fps: torch.Tensor) -> np.ndarray:
    max_frames = frame_features.shape[1]
    frame_index = torch.arange(max_frames, device=frame_features.device).view(1, max_frames)
    mask = frame_index < lengths_20fps.view(-1, 1)
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1).float()
    pooled = (frame_features * mask.unsqueeze(-1).float()).sum(dim=1) / denom
    return pooled.detach().cpu().numpy().astype(np.float32)


def per_sample_metrics(batch: DiagnosticBatch) -> Dict[str, np.ndarray]:
    episodes = batch.episodes
    rewards = (
        batch.reward_batch.frame_rewards * batch.reward_batch.frame_exec_mask.float()
    ).sum(dim=1).detach().cpu().numpy()
    success = np.asarray([float(bool(ep.get("success", False))) for ep in episodes], dtype=np.float32)
    exec_ratio = np.asarray(
        [float(ep.get("exec_len", 0)) / max(1.0, float(ep.get("target_len", 1))) for ep in episodes],
        dtype=np.float32,
    )
    phc_return = np.asarray([float(ep.get("return_mean", 0.0)) for ep in episodes], dtype=np.float32)
    terminate = np.asarray([float(bool(ep.get("terminate", False))) for ep in episodes], dtype=np.float32)
    return {
        "success": success,
        "terminate": terminate,
        "exec_ratio": exec_ratio,
        "phc_return_mean": phc_return,
        "undiscounted_sequence_reward": rewards.astype(np.float32),
    }


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
