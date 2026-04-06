from typing import Dict, Iterable

import numpy as np
import torch


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    valid = mask.bool()
    if not bool(valid.any().item()):
        return 0.0
    return float(values[valid].mean().item())


def merge_metrics(*metric_dicts: Iterable[Dict[str, float]]) -> Dict[str, float]:
    merged: Dict[str, float] = {}
    for item in metric_dicts:
        merged.update(item)
    return merged


def summarize_episodes(episodes) -> Dict[str, float]:
    if not episodes:
        return {
            "success_rate": 0.0,
            "phc_return_mean": 0.0,
            "exec_ratio_mean": 0.0,
        }
    reward_means = np.asarray([float(ep.get("return_mean", 0.0)) for ep in episodes], dtype=np.float32)
    exec_ratio = np.asarray(
        [
            float(ep.get("exec_len", 0)) / max(1.0, float(ep.get("target_len", 1)))
            for ep in episodes
        ],
        dtype=np.float32,
    )
    successes = np.asarray([bool(ep.get("success", False)) for ep in episodes], dtype=np.float32)
    return {
        "success_rate": float(successes.mean()),
        "phc_return_mean": float(reward_means.mean()),
        "exec_ratio_mean": float(exec_ratio.mean()),
    }
