from __future__ import annotations

import torch


def build_step_update_mask(rollout, ft_denoising_steps: int) -> torch.Tensor:
    timesteps = rollout.timesteps.long()
    step_mask = timesteps != 0
    if int(ft_denoising_steps) > 0:
        step_mask = step_mask & (timesteps <= int(ft_denoising_steps))
    return step_mask


def build_terminal_reward_sequence(rollout) -> torch.Tensor:
    step_rewards = torch.zeros((rollout.batch_size, rollout.num_steps), dtype=torch.float32)
    step_rewards[:, -1] = rollout.rewards.float()
    rollout.step_rewards = step_rewards.cpu()
    return rollout.step_rewards


def build_chunk_reward_sequence(rollout) -> torch.Tensor:
    if rollout.chunk_rewards is None:
        raise ValueError("Rollout does not contain chunk_rewards required by chunk assignment.")
    step_chunk_rewards = torch.zeros(
        (rollout.batch_size, rollout.num_steps, rollout.chunk_rewards.shape[1]),
        dtype=torch.float32,
    )
    step_chunk_rewards[:, -1, :] = rollout.chunk_rewards.float()
    rollout.step_rewards = step_chunk_rewards.cpu()
    return rollout.step_rewards


def compute_gae_from_rewards_and_values(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
):
    if rewards.shape != values.shape:
        raise ValueError(f"rewards and values must have the same shape, got {rewards.shape} vs {values.shape}.")

    rewards = rewards.float()
    values = values.float()
    next_values = torch.zeros_like(values)
    next_values[:, :-1] = values[:, 1:]

    deltas = rewards + float(gamma) * next_values - values
    advantages = torch.zeros_like(deltas)
    gae = torch.zeros((values.shape[0],), dtype=values.dtype, device=values.device)

    ##############################
    # Run the backward GAE recursion on diffusion steps. Rewards are zero
    # except for the last step, so credit flows from the terminal PHC reward
    # through the value bootstrap instead of being copied uniformly.
    ##############################
    for step_idx in range(values.shape[1] - 1, -1, -1):
        gae = deltas[:, step_idx] + float(gamma) * float(gae_lambda) * gae
        advantages[:, step_idx] = gae

    value_targets = advantages + values
    return advantages, value_targets, deltas


def compute_chunk_gae_from_rewards_and_values(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
):
    if rewards.shape != values.shape:
        raise ValueError(f"chunk rewards and values must have the same shape, got {rewards.shape} vs {values.shape}.")
    rewards = rewards.float()
    values = values.float()
    next_values = torch.zeros_like(values)
    next_values[:, :-1, :] = values[:, 1:, :]

    deltas = rewards + float(gamma) * next_values - values
    advantages = torch.zeros_like(deltas)
    gae = torch.zeros((values.shape[0], values.shape[2]), dtype=values.dtype, device=values.device)

    ##############################################
    # Run one backward GAE recursion per chunk. The time axis is still the
    # diffusion step axis, while chunk indices remain independent columns.
    ##############################################
    for step_idx in range(values.shape[1] - 1, -1, -1):
        gae = deltas[:, step_idx, :] + float(gamma) * float(gae_lambda) * gae
        advantages[:, step_idx, :] = gae

    value_targets = advantages + values
    return advantages, value_targets, deltas


def flatten_hidden_features(rollout):
    if rollout.hidden_features is None:
        raise ValueError("Rollout does not contain hidden_features required by actor_critic.")

    hidden = rollout.hidden_features.float()
    batch_size, num_steps, num_frames, hidden_dim = hidden.shape
    frame_mask = rollout.mask.squeeze(1).squeeze(1).bool()
    step_mask = frame_mask.unsqueeze(1).expand(batch_size, num_steps, num_frames)

    return (
        hidden.reshape(batch_size * num_steps, num_frames, hidden_dim),
        step_mask.reshape(batch_size * num_steps, num_frames),
    )


def reshape_step_vector(rollout, values: torch.Tensor) -> torch.Tensor:
    return values.reshape(rollout.batch_size, rollout.num_steps)


def reshape_step_chunk_tensor(rollout, values: torch.Tensor) -> torch.Tensor:
    if rollout.chunk_rewards is None:
        raise ValueError("Rollout does not contain chunk_rewards required by chunk assignment.")
    return values.reshape(rollout.batch_size, rollout.num_steps, rollout.chunk_rewards.shape[1])


def flatten_hidden_features_with_chunk_mask(rollout):
    hidden_flat, frame_mask_flat = flatten_hidden_features(rollout)
    if rollout.chunk_exec_mask is None:
        raise ValueError("Rollout does not contain chunk_exec_mask required by chunk assignment.")
    chunk_exec_mask = rollout.chunk_exec_mask.bool()
    step_chunk_mask = chunk_exec_mask.unsqueeze(1).expand(rollout.batch_size, rollout.num_steps, chunk_exec_mask.shape[1])
    return hidden_flat, frame_mask_flat, step_chunk_mask.reshape(rollout.batch_size * rollout.num_steps, chunk_exec_mask.shape[1])
