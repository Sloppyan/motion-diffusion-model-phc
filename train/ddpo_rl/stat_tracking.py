from collections import deque

import numpy as np


class PerPromptStatTracker:
    def __init__(
        self,
        buffer_size: int,
        min_count: int,
        mode: str = "zscore",
        ema_alpha: float = 0.05,
    ):
        self.buffer_size = int(buffer_size)
        self.min_count = int(min_count)
        self.mode = str(mode)
        self.ema_alpha = float(ema_alpha)
        if self.mode not in {"zscore", "ema", "history_zscore"}:
            raise ValueError(f"Unsupported prompt baseline mode: {self.mode}")
        if not (0.0 < self.ema_alpha <= 1.0):
            raise ValueError(f"prompt EMA alpha must be in (0, 1], got {self.ema_alpha}")
        self.stats = {}

    def _ensure_zscore_state(self, prompt):
        if prompt not in self.stats:
            self.stats[prompt] = deque(maxlen=self.buffer_size)
        return self.stats[prompt]

    def _ensure_ema_state(self, prompt):
        if prompt not in self.stats:
            self.stats[prompt] = {"count": 0, "ema": 0.0}
        return self.stats[prompt]

    def update(self, prompts, rewards):
        prompts = np.asarray(prompts)
        rewards = np.asarray(rewards, dtype=np.float32)
        unique_prompts = np.unique(prompts)
        advantages = np.empty_like(rewards)

        batch_mean = float(np.mean(rewards))
        batch_std = float(np.std(rewards) + 1e-6)

        for prompt in unique_prompts:
            prompt_rewards = rewards[prompts == prompt]
            if self.mode == "zscore":
                values = self._ensure_zscore_state(prompt)
                values.extend(prompt_rewards.tolist())

                if len(values) < self.min_count:
                    mean = batch_mean
                    std = batch_std
                else:
                    mean = float(np.mean(values))
                    std = float(np.std(values) + 1e-6)
                advantages[prompts == prompt] = (prompt_rewards - mean) / std
            elif self.mode == "history_zscore":
                values = self._ensure_zscore_state(prompt)
                prev_count = len(values)

                if prev_count < self.min_count:
                    advantages[prompts == prompt] = (prompt_rewards - batch_mean) / batch_std
                else:
                    hist = np.asarray(values, dtype=np.float32)
                    mean = float(np.mean(hist))
                    std = float(np.std(hist) + 1e-6)
                    advantages[prompts == prompt] = (prompt_rewards - mean) / std

                values.extend(prompt_rewards.tolist())
            else:
                state = self._ensure_ema_state(prompt)
                prev_count = int(state["count"])
                prev_ema = float(state["ema"])

                if prev_count < self.min_count:
                    advantages[prompts == prompt] = (prompt_rewards - batch_mean) / batch_std
                else:
                    advantages[prompts == prompt] = prompt_rewards - prev_ema

                batch_reward_mean = float(np.mean(prompt_rewards))
                if prev_count == 0:
                    state["ema"] = batch_reward_mean
                else:
                    state["ema"] = (1.0 - self.ema_alpha) * prev_ema + self.ema_alpha * batch_reward_mean
                state["count"] = prev_count + int(prompt_rewards.shape[0])

        return advantages

    def prompt_stats(self, prompt):
        values = self.stats.get(prompt)
        if self.mode in {"zscore", "history_zscore"}:
            if values is None or len(values) == 0:
                return {
                    "count": 0,
                    "mean": 0.0,
                    "std": 0.0,
                    "ready": False,
                    "ema": 0.0,
                    "mode": self.mode,
                }
            return {
                "count": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "ready": len(values) >= self.min_count,
                "ema": 0.0,
                "mode": self.mode,
            }

        if values is None:
            return {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "ready": False,
                "ema": 0.0,
                "mode": self.mode,
            }
        count = int(values["count"])
        ema = float(values["ema"])
        return {
            "count": count,
            "mean": ema,
            "std": 0.0,
            "ready": count >= self.min_count,
            "ema": ema,
            "mode": self.mode,
        }
