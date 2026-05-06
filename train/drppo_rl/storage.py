from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class DRPPORollout:
    texts: List[str]
    text_embeds: torch.Tensor
    lengths_20fps: torch.Tensor
    frame_features: torch.Tensor
    frame_rewards: torch.Tensor
    frame_exec_mask: torch.Tensor
    frame_dones: torch.Tensor
    frame_values: torch.Tensor
    frame_advantages: torch.Tensor
    frame_returns: torch.Tensor
    final_sample: torch.Tensor
    success: torch.Tensor
    terminate: torch.Tensor
    episodes: List[Dict]
    controlled_timesteps: torch.Tensor
    controlled_x_t_frames: torch.Tensor
    reverse_noise_actions: torch.Tensor
    reverse_logprobs_old: torch.Tensor
    reverse_action_mask: torch.Tensor
    reverse_advantages: Optional[torch.Tensor]


def compute_reverse_step_advantages(
    frame_advantages: torch.Tensor,
    frame_exec_mask: torch.Tensor,
    num_controlled_steps: int,
    gamma_denoising: float,
) -> Dict[str, torch.Tensor]:
    if num_controlled_steps <= 0:
        empty = frame_advantages.new_empty((frame_advantages.shape[0], 0, frame_advantages.shape[1]))
        return {
            "discounts": frame_advantages.new_empty((0,)),
            "reverse_advantages": empty,
        }

    discounts = torch.tensor(
        [float(gamma_denoising) ** (num_controlled_steps - idx - 1) for idx in range(num_controlled_steps)],
        dtype=frame_advantages.dtype,
        device=frame_advantages.device,
    )
    reverse_advantages = frame_advantages.unsqueeze(1) * discounts.view(1, num_controlled_steps, 1)
    reverse_advantages = reverse_advantages * frame_exec_mask.unsqueeze(1).float()
    return {
        "discounts": discounts,
        "reverse_advantages": reverse_advantages,
    }
