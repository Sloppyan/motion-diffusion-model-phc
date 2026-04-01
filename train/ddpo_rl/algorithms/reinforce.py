from __future__ import annotations

from collections import deque
from typing import Dict, List

import torch

from train.ddpo_rl.algorithms.common import ppo_policy_update
from train.ddpo_rl.interfaces import RLAlgorithm
from train.ddpo_rl.stat_tracking import PerPromptStatTracker


class ReinforceAlgorithm(RLAlgorithm):
    requires_hidden_features = False

    def __init__(
        self,
        model,
        diffusion,
        optimizer,
        trainable_params,
        device,
        args,
        reference_model=None,
    ):
        self.model = model
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.trainable_params = list(trainable_params)
        self.device = device
        self.args = args
        self.reference_model = reference_model
        self.stat_tracker = PerPromptStatTracker(
            buffer_size=args.prompt_stat_buffer_size,
            min_count=args.prompt_stat_min_count,
            mode=args.prompt_baseline_mode,
            ema_alpha=args.prompt_ema_alpha,
        )
        self._last_prompt_stats: List[Dict] = []

    def on_rollout_end(self, rollout) -> Dict[str, float]:
        advantages = self.stat_tracker.update(rollout.texts, rollout.rewards.cpu().numpy())
        rollout.advantages = torch.as_tensor(advantages, dtype=torch.float32)
        rollout.step_advantages = rollout.advantages.unsqueeze(1).expand(-1, rollout.num_steps).contiguous()
        self._last_prompt_stats = [self.stat_tracker.prompt_stats(prompt) for prompt in rollout.texts]
        return {
            "adv_mean": float(rollout.advantages.mean().item()),
            "adv_std": float(rollout.advantages.std(unbiased=False).item()),
        }

    def update(self, rollout) -> Dict[str, float]:
        metrics = ppo_policy_update(
            model=self.model,
            diffusion=self.diffusion,
            optimizer=self.optimizer,
            trainable_params=self.trainable_params,
            rollout=rollout,
            step_advantages=rollout.step_advantages,
            device=self.device,
            args=self.args,
            reference_model=self.reference_model,
        )
        return {
            "policy_loss": metrics["loss"],
            "approx_kl": metrics["approx_kl"],
            "clipfrac": metrics["clipfrac"],
            "ratio_mean": metrics["ratio_mean"],
            "entropy": metrics["entropy"],
            "reg_loss": metrics["reg_loss"],
        }

    def debug_entries(self, rollout, limit: int) -> List[Dict]:
        entries = []
        for idx in range(min(int(limit), rollout.batch_size)):
            prompt_stats = self._last_prompt_stats[idx] if idx < len(self._last_prompt_stats) else {}
            entries.append(
                {
                    "prompt": rollout.texts[idx],
                    "reward": round(float(rollout.rewards[idx].item()), 6),
                    "adv": round(float(rollout.advantages[idx].item()), 6),
                    "count": int(prompt_stats.get("count", 0)),
                    "ready": bool(prompt_stats.get("ready", False)),
                }
            )
        return entries

    def state_dict(self) -> Dict:
        return {
            "prompt_stats": self.stat_tracker.stats,
        }
