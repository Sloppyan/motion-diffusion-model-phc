from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class FrameRollout:
    texts: List[str]
    text_embeds: torch.Tensor
    lengths_20fps: torch.Tensor
    frame_features: torch.Tensor
    frame_rewards: torch.Tensor
    frame_exec_mask: torch.Tensor
    frame_dones: torch.Tensor
    frame_values: Optional[torch.Tensor]
    frame_advantages: Optional[torch.Tensor]
    frame_returns: Optional[torch.Tensor]
    chain_prev: torch.Tensor
    chain_next: torch.Tensor
    final_sample: torch.Tensor
    timesteps: torch.Tensor
    frame_logprobs_old: torch.Tensor
    success: torch.Tensor
    terminate: torch.Tensor
    episodes: List[Dict]
    trainable_step_frac: float


def compute_frame_returns_and_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    exec_mask: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Dict[str, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros((rewards.shape[0],), device=rewards.device, dtype=rewards.dtype)
    for frame_idx in reversed(range(rewards.shape[1])):
        mask_t = exec_mask[:, frame_idx].float()
        if frame_idx == rewards.shape[1] - 1:
            next_values = torch.zeros_like(lastgaelam)
        else:
            next_values = values[:, frame_idx + 1] * exec_mask[:, frame_idx + 1].float()
        nonterminal = (1.0 - dones[:, frame_idx].float()) * mask_t
        delta = (rewards[:, frame_idx] + gamma * next_values * nonterminal - values[:, frame_idx]) * mask_t
        lastgaelam = delta + gamma * gae_lambda * nonterminal * lastgaelam
        advantages[:, frame_idx] = lastgaelam
        lastgaelam = lastgaelam * mask_t

    returns = advantages + values
    returns = returns * exec_mask.float()
    advantages = advantages * exec_mask.float()
    return {
        "advantages": advantages,
        "returns": returns,
    }
