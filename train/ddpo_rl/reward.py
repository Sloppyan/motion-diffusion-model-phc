import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


REWARD_PRESETS = {
    "refine": {
        "return_mean_weight": 20.0,
        "q10_weight": 8.0,
        "early_term_weight": -6.0,
        "success_weight": 1.0,
        "fail_weight": 0.0,
    },
    "failure": {
        "return_mean_weight": 20.0,
        "q10_weight": 8.0,
        "early_term_weight": -6.0,
        "success_weight": 1.0,
        "fail_weight": 1.0,
    },
}


@dataclass(frozen=True)
class EpisodeRewardTerms:
    return_mean: float
    q10_reward: float
    completion: float
    early_term: float
    success: float
    fail: float


@dataclass(frozen=True)
class RewardSpec:
    mode: str
    return_mean_weight: float
    q10_weight: float
    early_term_weight: float
    success_weight: float
    fail_weight: float
    assignment: str
    chunk_size: int
    chunk_reduce: str
    chunk_count: int
    chunk_mean_weight: float
    chunk_q10_weight: float
    chunk_success_weight: float
    chunk_fail_weight: float
    chunk_fail_prev_weight: float
    chunk_early_weight: float
    chunk_early_prev_weight: float


@dataclass(frozen=True)
class ChunkRewardTerms:
    frame_reward: np.ndarray
    frame_exec_mask: np.ndarray
    chunk_reward: np.ndarray
    base_chunk_reward: np.ndarray
    success_chunk_reward: np.ndarray
    fail_chunk_reward: np.ndarray
    early_chunk_reward: np.ndarray
    chunk_exec_mask: np.ndarray
    chunk_frame_counts: np.ndarray
    chunk_weights: np.ndarray
    chunk_ranges: List[Tuple[int, int]]
    target_len_20fps: int
    exec_len_20fps: int
    term_chunk: int
    prev_chunk: int


def _resolve_weight(override, fallback: float) -> float:
    if override is None:
        return float(fallback)
    return float(override)


def extract_episode_reward_terms(episode: dict) -> EpisodeRewardTerms:
    reward_steps_value = episode.get("reward_steps", [])
    if reward_steps_value is None:
        reward_steps = np.empty((0,), dtype=np.float32)
    else:
        reward_steps = np.asarray(reward_steps_value, dtype=np.float32).reshape(-1)

    return_mean = float(episode.get("return_mean", 0.0) or 0.0)
    q10_reward = float(np.quantile(reward_steps, 0.1)) if reward_steps.size > 0 else 0.0

    target_len = max(1, int(episode.get("target_len", 0) or 0))
    exec_len = int(episode.get("exec_len", 0) or 0)
    completion = float(np.clip(exec_len / float(target_len), 0.0, 1.0))
    early_term = max(0.0, 1.0 - completion)
    success = float(bool(episode.get("success", False)))
    fail = 1.0 - success

    return EpisodeRewardTerms(
        return_mean=return_mean,
        q10_reward=q10_reward,
        completion=completion,
        early_term=early_term,
        success=success,
        fail=fail,
    )


def build_reward_spec(args) -> RewardSpec:
    if args.reward_mode not in REWARD_PRESETS:
        raise ValueError(f"Unsupported reward mode: {args.reward_mode}")

    preset = REWARD_PRESETS[args.reward_mode]
    return RewardSpec(
        mode=args.reward_mode,
        return_mean_weight=_resolve_weight(args.reward_return_mean_weight, preset["return_mean_weight"]),
        q10_weight=_resolve_weight(args.reward_q10_weight, preset["q10_weight"]),
        early_term_weight=_resolve_weight(args.reward_early_term_weight, preset["early_term_weight"]),
        success_weight=_resolve_weight(args.reward_success_weight, preset["success_weight"]),
        fail_weight=_resolve_weight(args.reward_fail_penalty, preset["fail_weight"]),
        assignment=str(args.reward_assignment),
        chunk_size=int(args.reward_chunk_size),
        chunk_reduce=str(args.reward_chunk_reduce),
        chunk_count=max(1, int(math.ceil(int(args.max_motion_frames) / float(args.reward_chunk_size)))),
        chunk_mean_weight=_resolve_weight(args.chunk_mean_weight, _resolve_weight(args.reward_return_mean_weight, preset["return_mean_weight"])),
        chunk_q10_weight=_resolve_weight(args.chunk_q10_weight, _resolve_weight(args.reward_q10_weight, preset["q10_weight"])),
        chunk_success_weight=_resolve_weight(args.chunk_success_weight, _resolve_weight(args.reward_success_weight, preset["success_weight"])),
        chunk_fail_weight=_resolve_weight(args.chunk_fail_weight, _resolve_weight(args.reward_fail_penalty, preset["fail_weight"])),
        chunk_fail_prev_weight=_resolve_weight(args.chunk_fail_prev_weight, _resolve_weight(args.reward_fail_penalty, preset["fail_weight"])),
        chunk_early_weight=_resolve_weight(args.chunk_early_weight, abs(_resolve_weight(args.reward_early_term_weight, preset["early_term_weight"]))),
        chunk_early_prev_weight=_resolve_weight(args.chunk_early_prev_weight, abs(_resolve_weight(args.reward_early_term_weight, preset["early_term_weight"]))),
    )


def score_reward_terms(terms: EpisodeRewardTerms, spec: RewardSpec) -> Tuple[float, Dict[str, float]]:
    # Keep raw term extraction and weighted composition separate so the
    # training loop, logging, and future reward variants all share one scorer.
    return_mean_term = spec.return_mean_weight * terms.return_mean
    q10_term = spec.q10_weight * terms.q10_reward
    early_term_term = spec.early_term_weight * terms.early_term
    success_term = spec.success_weight * terms.success
    fail_term = -spec.fail_weight * terms.fail
    total_reward = return_mean_term + q10_term + early_term_term + success_term + fail_term

    return float(total_reward), {
        "return_mean_term": float(return_mean_term),
        "q10_term": float(q10_term),
        "early_term_term": float(early_term_term),
        "success_term": float(success_term),
        "fail_term": float(fail_term),
        "total_reward": float(total_reward),
    }


def score_episode_reward(episode: dict, spec: RewardSpec) -> Tuple[float, Dict[str, float]]:
    terms = extract_episode_reward_terms(episode)
    return score_reward_terms(terms, spec)


def build_chunk_ranges(max_motion_frames: int, chunk_size: int, chunk_count: Optional[int] = None) -> List[Tuple[int, int]]:
    max_motion_frames = max(1, int(max_motion_frames))
    chunk_size = max(1, int(chunk_size))
    if chunk_count is None:
        chunk_count = max(1, int(math.ceil(max_motion_frames / float(chunk_size))))

    ranges: List[Tuple[int, int]] = []
    for chunk_idx in range(int(chunk_count)):
        start = int(chunk_idx * chunk_size)
        end = int(min(max_motion_frames, start + chunk_size))
        ranges.append((start, end))
    return ranges


def build_chunk_frame_mask(lengths, max_motion_frames: int, chunk_size: int, chunk_count: int):
    lengths_np = np.asarray(lengths, dtype=np.int64).reshape(-1)
    max_motion_frames = max(1, int(max_motion_frames))
    chunk_ranges = build_chunk_ranges(max_motion_frames, chunk_size, chunk_count=chunk_count)
    frame_ids = np.arange(max_motion_frames, dtype=np.int64)[None, None, :]

    chunk_masks = []
    for start, end in chunk_ranges:
        if end <= start:
            chunk_masks.append(np.zeros((lengths_np.shape[0], max_motion_frames), dtype=bool))
            continue
        local_mask = (frame_ids >= start) & (frame_ids < end) & (frame_ids < lengths_np[:, None, None])
        chunk_masks.append(local_mask[:, 0, :])

    chunk_frame_mask = np.stack(chunk_masks, axis=1)
    chunk_frame_counts = chunk_frame_mask.sum(axis=-1).astype(np.int64)
    return chunk_ranges, chunk_frame_mask, chunk_frame_counts


def compute_exec_len_20fps(exec_len_30hz: int, target_len_20fps: int, target_len_30hz: int) -> int:
    target_len_20fps = max(1, int(target_len_20fps))
    target_len_30hz = max(1, int(target_len_30hz))
    exec_len_30hz = max(0, int(exec_len_30hz))
    return int(np.clip(np.rint(exec_len_30hz * (target_len_20fps / float(target_len_30hz))), 0, target_len_20fps))


def build_exec_frame_mask(target_len_20fps: int, exec_len_20fps: int) -> np.ndarray:
    target_len_20fps = max(1, int(target_len_20fps))
    exec_len_20fps = int(np.clip(exec_len_20fps, 0, target_len_20fps))
    mask = np.zeros((target_len_20fps,), dtype=bool)
    mask[:exec_len_20fps] = True
    return mask


def align_reward_steps_30hz_to_frames_20hz(reward_steps, exec_len_20fps: int) -> np.ndarray:
    exec_len_20fps = max(0, int(exec_len_20fps))
    reward_steps = np.asarray(reward_steps, dtype=np.float32).reshape(-1)
    if exec_len_20fps == 0 or reward_steps.size == 0:
        return np.zeros((exec_len_20fps,), dtype=np.float32)

    frame_sums = np.zeros((exec_len_20fps,), dtype=np.float32)
    frame_counts = np.zeros((exec_len_20fps,), dtype=np.int32)
    frame_idx = np.floor(np.arange(reward_steps.size, dtype=np.float32) * (exec_len_20fps / float(reward_steps.size))).astype(
        np.int64
    )
    frame_idx = np.clip(frame_idx, 0, exec_len_20fps - 1)
    np.add.at(frame_sums, frame_idx, reward_steps)
    np.add.at(frame_counts, frame_idx, 1)

    frame_reward = np.zeros((exec_len_20fps,), dtype=np.float32)
    valid_mask = frame_counts > 0
    frame_reward[valid_mask] = frame_sums[valid_mask] / frame_counts[valid_mask]
    if valid_mask.any() and not valid_mask.all():
        valid_idx = np.flatnonzero(valid_mask)
        missing_idx = np.flatnonzero(~valid_mask)
        frame_reward[missing_idx] = np.interp(
            missing_idx.astype(np.float32),
            valid_idx.astype(np.float32),
            frame_reward[valid_idx],
        )
    return frame_reward


def aggregate_frame_reward_to_chunks(
    frame_reward,
    frame_exec_mask,
    chunk_ranges: List[Tuple[int, int]],
    chunk_reduce: str = "mean",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_reward = np.asarray(frame_reward, dtype=np.float32).reshape(-1)
    frame_exec_mask = np.asarray(frame_exec_mask, dtype=bool).reshape(-1)
    if chunk_reduce != "mean":
        raise ValueError(f"Unsupported chunk_reduce: {chunk_reduce}")

    chunk_mean = np.zeros((len(chunk_ranges),), dtype=np.float32)
    chunk_q10 = np.zeros((len(chunk_ranges),), dtype=np.float32)
    chunk_exec_mask = np.zeros((len(chunk_ranges),), dtype=bool)
    for chunk_idx, (start, end) in enumerate(chunk_ranges):
        if start >= frame_reward.shape[0] or end <= start:
            continue
        local_end = min(end, frame_reward.shape[0])
        local_valid = frame_exec_mask[start:local_end]
        if not local_valid.any():
            continue
        local = frame_reward[start:local_end][local_valid]
        chunk_mean[chunk_idx] = float(local.mean())
        chunk_q10[chunk_idx] = float(np.quantile(local, 0.1))
        chunk_exec_mask[chunk_idx] = True
    return chunk_mean, chunk_q10, chunk_exec_mask


def extract_chunk_reward_terms(episode: dict, spec: RewardSpec) -> ChunkRewardTerms:
    target_len_20fps = int(
        episode.get(
            "gt_len_20fps",
            episode.get("length", max(1, int(round(float(episode.get("target_len", 1) or 1) * 20.0 / 30.0)))),
        )
        or 1
    )
    target_len_20fps = max(1, target_len_20fps)
    target_len_30hz = max(1, int(episode.get("target_len", 0) or 0) or int(round(target_len_20fps * 30.0 / 20.0)))
    exec_len_20fps = compute_exec_len_20fps(
        exec_len_30hz=int(episode.get("exec_len", 0) or 0),
        target_len_20fps=target_len_20fps,
        target_len_30hz=target_len_30hz,
    )
    frame_exec_mask = build_exec_frame_mask(target_len_20fps=target_len_20fps, exec_len_20fps=exec_len_20fps)
    frame_reward_exec = align_reward_steps_30hz_to_frames_20hz(
        episode.get("reward_steps", []),
        exec_len_20fps=exec_len_20fps,
    )
    frame_reward = np.zeros((target_len_20fps,), dtype=np.float32)
    frame_reward[:exec_len_20fps] = frame_reward_exec
    chunk_ranges = build_chunk_ranges(
        max_motion_frames=spec.chunk_count * spec.chunk_size,
        chunk_size=spec.chunk_size,
        chunk_count=spec.chunk_count,
    )
    chunk_frame_counts = np.asarray(
        [max(0, min(end, target_len_20fps) - start) if start < target_len_20fps else 0 for start, end in chunk_ranges],
        dtype=np.int64,
    )
    chunk_weights = np.zeros((len(chunk_ranges),), dtype=np.float32)
    total_frames = int(chunk_frame_counts.sum())
    if total_frames > 0:
        chunk_weights = chunk_frame_counts.astype(np.float32) / float(total_frames)

    chunk_mean, chunk_q10, chunk_exec_mask = aggregate_frame_reward_to_chunks(
        frame_reward=frame_reward,
        frame_exec_mask=frame_exec_mask,
        chunk_ranges=chunk_ranges,
        chunk_reduce=spec.chunk_reduce,
    )
    base_chunk_reward = spec.chunk_mean_weight * chunk_mean + spec.chunk_q10_weight * chunk_q10
    base_chunk_reward = np.where(chunk_exec_mask, base_chunk_reward, 0.0).astype(np.float32)

    terms = extract_episode_reward_terms(episode)
    success_chunk_reward = (spec.chunk_success_weight * terms.success * chunk_weights).astype(np.float32)
    fail_chunk_reward = np.zeros((len(chunk_ranges),), dtype=np.float32)
    early_chunk_reward = np.zeros((len(chunk_ranges),), dtype=np.float32)

    term_chunk = -1
    prev_chunk = -1
    if terms.fail > 0.0 and len(chunk_ranges) > 0:
        term_frame = max(0, exec_len_20fps - 1)
        term_chunk = min(len(chunk_ranges) - 1, int(term_frame // max(1, spec.chunk_size)))
        prev_chunk = max(term_chunk - 1, 0)
        fail_chunk_reward[term_chunk] -= spec.chunk_fail_weight * terms.fail
        fail_chunk_reward[prev_chunk] -= spec.chunk_fail_prev_weight * terms.fail
        early_chunk_reward[term_chunk] -= spec.chunk_early_weight * terms.early_term
        early_chunk_reward[prev_chunk] -= spec.chunk_early_prev_weight * terms.early_term

    chunk_reward = base_chunk_reward + success_chunk_reward + fail_chunk_reward + early_chunk_reward
    return ChunkRewardTerms(
        frame_reward=frame_reward,
        frame_exec_mask=frame_exec_mask,
        chunk_reward=chunk_reward.astype(np.float32),
        base_chunk_reward=base_chunk_reward.astype(np.float32),
        success_chunk_reward=success_chunk_reward.astype(np.float32),
        fail_chunk_reward=fail_chunk_reward.astype(np.float32),
        early_chunk_reward=early_chunk_reward.astype(np.float32),
        chunk_exec_mask=chunk_exec_mask,
        chunk_frame_counts=chunk_frame_counts,
        chunk_weights=chunk_weights,
        chunk_ranges=chunk_ranges,
        target_len_20fps=target_len_20fps,
        exec_len_20fps=exec_len_20fps,
        term_chunk=term_chunk,
        prev_chunk=prev_chunk,
    )
