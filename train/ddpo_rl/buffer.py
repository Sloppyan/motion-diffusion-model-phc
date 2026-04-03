from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from diffusion.gaussian_diffusion import calc_chunk_log_prob_from_stats


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
    chunk_ranges: Optional[List[tuple]] = None
    chunk_frame_mask: Optional[torch.Tensor] = None
    chunk_exec_mask: Optional[torch.Tensor] = None
    chunk_frame_counts: Optional[torch.Tensor] = None
    chunk_weights: Optional[torch.Tensor] = None
    chunk_logp_old: Optional[torch.Tensor] = None
    chunk_rewards: Optional[torch.Tensor] = None
    chunk_values: Optional[torch.Tensor] = None
    chunk_value_targets: Optional[torch.Tensor] = None
    chunk_advantages: Optional[torch.Tensor] = None

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
    chunk_ranges: Optional[List[tuple]] = None,
    chunk_frame_mask: Optional[torch.Tensor] = None,
    chunk_exec_mask: Optional[torch.Tensor] = None,
    chunk_frame_counts: Optional[torch.Tensor] = None,
    chunk_weights: Optional[torch.Tensor] = None,
    chunk_rewards: Optional[torch.Tensor] = None,
) -> DDPORolloutBatch:
    x_t = torch.stack([step["x_t"] for step in trajectory], dim=1).float()
    x_prev = torch.stack([step["x_prev"] for step in trajectory], dim=1).float()
    timesteps = torch.stack([step["t"] for step in trajectory], dim=1).long()
    logp_old = torch.stack([step["logp"] for step in trajectory], dim=1).float()
    mean_old = torch.stack([step["mean"] for step in trajectory], dim=1).float()
    log_variance_old = torch.stack([step["log_variance"] for step in trajectory], dim=1).float()
    hidden_features = None
    if len(trajectory) > 0 and "hidden" in trajectory[0]:
        hidden_features = torch.stack([step["hidden"] for step in trajectory], dim=1)

    chunk_logp_old = None
    if chunk_frame_mask is not None:
        chunk_frame_mask = chunk_frame_mask.bool()
        step_chunk_frame_mask = chunk_frame_mask.unsqueeze(1).expand(-1, x_prev.shape[1], -1, -1)
        chunk_logp_old = calc_chunk_log_prob_from_stats(
            x_prev=x_prev,
            mean=mean_old,
            log_variance=log_variance_old,
            chunk_frame_mask=step_chunk_frame_mask,
        ).cpu().float()

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
        chunk_ranges=chunk_ranges,
        chunk_frame_mask=chunk_frame_mask.cpu() if chunk_frame_mask is not None else None,
        chunk_exec_mask=chunk_exec_mask.cpu() if chunk_exec_mask is not None else None,
        chunk_frame_counts=chunk_frame_counts.cpu().long() if chunk_frame_counts is not None else None,
        chunk_weights=chunk_weights.cpu().float() if chunk_weights is not None else None,
        chunk_logp_old=chunk_logp_old,
        chunk_rewards=chunk_rewards.cpu().float() if chunk_rewards is not None else None,
    )
