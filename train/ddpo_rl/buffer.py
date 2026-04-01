from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class DDPORolloutBatch:
    texts: List[str]
    tokens: List[str]
    db_keys: List[str]
    caption_indices: List[int]
    text_embeds: torch.Tensor
    lengths: torch.Tensor
    mask: torch.Tensor
    scales: torch.Tensor
    x_t: torch.Tensor
    x_prev: torch.Tensor
    timesteps: torch.Tensor
    logp_old: torch.Tensor
    final_sample: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    phc_episodes: List[Dict]
    hidden_features: Optional[torch.Tensor] = None
    step_rewards: Optional[torch.Tensor] = None
    returns: Optional[torch.Tensor] = None
    values: Optional[torch.Tensor] = None
    td_deltas: Optional[torch.Tensor] = None
    step_advantages: Optional[torch.Tensor] = None

    @property
    def batch_size(self) -> int:
        return int(self.lengths.shape[0])

    @property
    def num_steps(self) -> int:
        return int(self.timesteps.shape[1])


def build_rollout_batch(
    texts: List[str],
    tokens: List[str],
    db_keys: List[str],
    caption_indices: List[int],
    text_embeds: torch.Tensor,
    lengths: torch.Tensor,
    mask: torch.Tensor,
    scales: torch.Tensor,
    sample: torch.Tensor,
    trajectory: List[Dict[str, torch.Tensor]],
    rewards: torch.Tensor,
    episodes: List[Dict],
) -> DDPORolloutBatch:
    x_t = torch.stack([step["x_t"] for step in trajectory], dim=1).float()
    x_prev = torch.stack([step["x_prev"] for step in trajectory], dim=1).float()
    timesteps = torch.stack([step["t"] for step in trajectory], dim=1).long()
    logp_old = torch.stack([step["logp"] for step in trajectory], dim=1).float()
    hidden_features = None
    if len(trajectory) > 0 and "hidden" in trajectory[0]:
        hidden_features = torch.stack([step["hidden"] for step in trajectory], dim=1)

    return DDPORolloutBatch(
        texts=texts,
        tokens=tokens,
        db_keys=db_keys,
        caption_indices=caption_indices,
        text_embeds=text_embeds.cpu().float(),
        lengths=lengths.cpu().long(),
        mask=mask.cpu().bool(),
        scales=scales.cpu().float(),
        x_t=x_t.cpu(),
        x_prev=x_prev.cpu(),
        timesteps=timesteps.cpu(),
        logp_old=logp_old.cpu(),
        final_sample=sample.cpu().float(),
        rewards=rewards.cpu().float(),
        advantages=torch.zeros_like(rewards).cpu().float(),
        phc_episodes=episodes,
        hidden_features=hidden_features.cpu() if hidden_features is not None else None,
    )
