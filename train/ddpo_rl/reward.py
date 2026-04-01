from dataclasses import dataclass
from typing import Dict, Tuple

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
