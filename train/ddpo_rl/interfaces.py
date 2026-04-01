from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List


class RLAlgorithm(ABC):
    requires_hidden_features: bool = False

    def on_rollout_end(self, rollout) -> Dict[str, float]:
        return {}

    @abstractmethod
    def update(self, rollout) -> Dict[str, float]:
        raise NotImplementedError

    def debug_entries(self, rollout, limit: int) -> List[Dict]:
        return []

    def state_dict(self) -> Dict:
        return {}
