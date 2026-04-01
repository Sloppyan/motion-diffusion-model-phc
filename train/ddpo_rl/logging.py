from __future__ import annotations

import json
from typing import Dict

import numpy as np

from train.ddpo_rl.reward import extract_episode_reward_terms, score_reward_terms


def summarize_rollout(rollout, reward_spec, prefix: str = "") -> Dict[str, float]:
    reward_mean = float(rollout.rewards.mean().item())
    reward_std = float(rollout.rewards.std(unbiased=False).item())
    success_rate = float(np.mean([float(ep.get("success", False)) for ep in rollout.phc_episodes]))
    terminate_rate = float(np.mean([float(ep.get("terminate", False)) for ep in rollout.phc_episodes]))
    phc_return_mean = float(np.mean([float(ep.get("return_mean", 0.0)) for ep in rollout.phc_episodes]))
    phc_return_sum = float(np.mean([float(ep.get("return_sum", 0.0)) for ep in rollout.phc_episodes]))
    reward_terms = [extract_episode_reward_terms(ep) for ep in rollout.phc_episodes]
    reward_infos = [score_reward_terms(term, reward_spec)[1] for term in reward_terms]
    q10_reward_mean = float(np.mean([term.q10_reward for term in reward_terms])) if reward_terms else 0.0
    completion_mean = float(np.mean([term.completion for term in reward_terms])) if reward_terms else 0.0
    early_term_mean = float(np.mean([term.early_term for term in reward_terms])) if reward_terms else 0.0
    return {
        f"{prefix}reward_mean": reward_mean,
        f"{prefix}reward_loss": -reward_mean,
        f"{prefix}reward_std": reward_std,
        f"{prefix}reward_mode": reward_spec.mode,
        f"{prefix}reward_component_return_mean": float(np.mean([info["return_mean_term"] for info in reward_infos])) if reward_infos else 0.0,
        f"{prefix}reward_component_q10": float(np.mean([info["q10_term"] for info in reward_infos])) if reward_infos else 0.0,
        f"{prefix}reward_component_early_term": float(np.mean([info["early_term_term"] for info in reward_infos])) if reward_infos else 0.0,
        f"{prefix}reward_component_success": float(np.mean([info["success_term"] for info in reward_infos])) if reward_infos else 0.0,
        f"{prefix}reward_component_fail": float(np.mean([info["fail_term"] for info in reward_infos])) if reward_infos else 0.0,
        f"{prefix}success_rate": success_rate,
        f"{prefix}terminate_rate": terminate_rate,
        f"{prefix}phc_return_mean": phc_return_mean,
        f"{prefix}phc_return_loss": -phc_return_mean,
        f"{prefix}phc_return_sum": phc_return_sum,
        f"{prefix}q10_reward_mean": q10_reward_mean,
        f"{prefix}completion_mean": completion_mean,
        f"{prefix}early_term_mean": early_term_mean,
    }


def format_debug_entries(entries) -> str:
    return "adv_debug=" + json.dumps(entries, ensure_ascii=False)
