from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch


@dataclass
class FrameRewardBatch:
    frame_rewards: torch.Tensor
    frame_exec_mask: torch.Tensor
    frame_dones: torch.Tensor
    success: torch.Tensor
    terminate: torch.Tensor
    undiscounted_sequence_reward_mean: float
    exec_ratio_mean: float


def _align_reward_steps_to_frames(
    reward_steps: np.ndarray,
    exec_len_30hz: int,
    target_len_20fps: int,
) -> np.ndarray:
    exec_len_20fps = max(1, min(target_len_20fps, int(round(exec_len_30hz * 20.0 / 30.0))))
    if reward_steps.size == 0:
        return np.zeros((exec_len_20fps,), dtype=np.float32)
    if reward_steps.size == exec_len_20fps:
        return reward_steps.astype(np.float32, copy=True)

    old_axis = np.linspace(0.0, 1.0, num=reward_steps.shape[0], dtype=np.float32)
    new_axis = np.linspace(0.0, 1.0, num=exec_len_20fps, dtype=np.float32)
    return np.interp(new_axis, old_axis, reward_steps.astype(np.float32)).astype(np.float32)


def build_frame_reward_batch(
    episodes: List[Dict],
    max_motion_frames: int,
    gamma_frame: float,
    dense_reward_weight: float,
    dense_reward_norm: bool,
    success_bonus: float,
    fail_penalty: float,
    device: torch.device,
) -> FrameRewardBatch:
    batch_size = len(episodes)
    frame_rewards = torch.zeros((batch_size, max_motion_frames), dtype=torch.float32, device=device)
    frame_exec_mask = torch.zeros((batch_size, max_motion_frames), dtype=torch.bool, device=device)
    frame_dones = torch.zeros((batch_size, max_motion_frames), dtype=torch.bool, device=device)
    success = torch.zeros((batch_size,), dtype=torch.float32, device=device)
    terminate = torch.zeros((batch_size,), dtype=torch.float32, device=device)

    exec_ratios: List[float] = []
    undiscounted_sequence_rewards: List[float] = []

    for idx, episode in enumerate(episodes):
        target_len_20fps = max(1, min(int(episode["length"]), max_motion_frames))
        exec_len_30hz = int(episode["exec_len"])
        dense = _align_reward_steps_to_frames(
            reward_steps=np.asarray(episode["reward_steps"], dtype=np.float32),
            exec_len_30hz=exec_len_30hz,
            target_len_20fps=target_len_20fps,
        )
        exec_len_20fps = int(dense.shape[0])
        if exec_len_20fps > 0:
            ##############################################
            # Dense tracking reward can be scaled down or disabled entirely.
            # When the weight is zero, only terminal sparse reward remains.
            ##############################################
            dense_t = torch.from_numpy(dense).to(device)
            if dense_reward_norm:
                dense_t = dense_t / float(target_len_20fps)
            weighted_dense = dense_t * float(dense_reward_weight)
            frame_rewards[idx, :exec_len_20fps] = weighted_dense
            frame_exec_mask[idx, :exec_len_20fps] = True
            frame_dones[idx, exec_len_20fps - 1] = True
        else:
            undiscounted_sequence_rewards.append(0.0)

        if bool(episode["terminate"]):
            frame_rewards[idx, max(0, exec_len_20fps - 1)] += float(fail_penalty)
            terminate[idx] = 1.0
        else:
            frame_rewards[idx, max(0, exec_len_20fps - 1)] += float(success_bonus)
            success[idx] = 1.0

        if exec_len_20fps > 0:
            valid_rewards = frame_rewards[idx, :exec_len_20fps]
            undiscounted_sequence_rewards.append(float(valid_rewards.sum().item()))

        exec_ratios.append(float(exec_len_20fps) / float(target_len_20fps))

    # Reward discounting belongs to return / GAE on the frame axis.
    # Keep frame_rewards as immediate rewards here to avoid double discounting.
    del gamma_frame

    return FrameRewardBatch(
        frame_rewards=frame_rewards,
        frame_exec_mask=frame_exec_mask,
        frame_dones=frame_dones,
        success=success,
        terminate=terminate,
        undiscounted_sequence_reward_mean=(
            float(np.mean(undiscounted_sequence_rewards)) if undiscounted_sequence_rewards else 0.0
        ),
        exec_ratio_mean=float(np.mean(exec_ratios)) if exec_ratios else 0.0,
    )
