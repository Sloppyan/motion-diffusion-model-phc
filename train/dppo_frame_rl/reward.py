from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch


@dataclass
class FrameRewardBatch:
    frame_rewards: torch.Tensor
    frame_exec_mask: torch.Tensor
    frame_dones: torch.Tensor
    success: torch.Tensor
    terminate: torch.Tensor
    dense_reward_mean: float
    imitation_pose_reward_mean: float
    pose_mse_mean: float
    imitation_velocity_reward_mean: float
    terminal_reward_mean: float
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
    reward_norm: bool,
    success_bonus: float,
    fail_penalty: float,
    device: torch.device,
    generated_joints_20fps: Optional[torch.Tensor] = None,
    gt_joints_20fps: Optional[torch.Tensor] = None,
    pose_reward_weight: float = 0.0,
    pose_reward_alpha: float = 1.0,
    pose_reward_joint_ids: Optional[Sequence[int]] = None,
    velocity_reward_weight: float = 0.0,
    velocity_reward_alpha: float = 1.0,
    velocity_reward_joint_ids: Optional[Sequence[int]] = None,
) -> FrameRewardBatch:
    batch_size = len(episodes)
    frame_rewards = torch.zeros((batch_size, max_motion_frames), dtype=torch.float32, device=device)
    frame_exec_mask = torch.zeros((batch_size, max_motion_frames), dtype=torch.bool, device=device)
    frame_dones = torch.zeros((batch_size, max_motion_frames), dtype=torch.bool, device=device)
    success = torch.zeros((batch_size,), dtype=torch.float32, device=device)
    terminate = torch.zeros((batch_size,), dtype=torch.float32, device=device)

    exec_ratios: List[float] = []
    undiscounted_sequence_rewards: List[float] = []
    dense_reward_sum = 0.0
    pose_reward_sum = 0.0
    pose_mse_sum = 0.0
    pose_mse_count = 0
    velocity_reward_sum = 0.0
    terminal_reward_sum = 0.0
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
            if reward_norm:
                dense_t = dense_t / float(target_len_20fps)
            weighted_dense = dense_t * float(dense_reward_weight)
            frame_rewards[idx, :exec_len_20fps] = weighted_dense
            dense_reward_sum += float(weighted_dense.sum().item())
            frame_exec_mask[idx, :exec_len_20fps] = True
            frame_dones[idx, exec_len_20fps - 1] = True

            ##############################################
            # Add a GT pose imitation reward on a small joint subset.
            # This discourages trivial "stand still" solutions while
            # keeping PHC tracking reward as the main optimization target.
            ##############################################
            if (
                generated_joints_20fps is not None
                and gt_joints_20fps is not None
            ):
                if pose_reward_joint_ids is None:
                    joint_ids = torch.arange(generated_joints_20fps.shape[2], dtype=torch.long, device=device)
                else:
                    joint_ids = torch.as_tensor(pose_reward_joint_ids, dtype=torch.long, device=device)
                pred_joints = generated_joints_20fps[idx, :exec_len_20fps].index_select(1, joint_ids)
                gt_joints = gt_joints_20fps[idx, :exec_len_20fps].index_select(1, joint_ids)
                pose_dist = ((pred_joints - gt_joints) ** 2).sum(dim=-1).mean(dim=-1)
                pose_mse_sum += float(pose_dist.sum().item())
                pose_mse_count += int(exec_len_20fps)
                if pose_reward_weight > 0.0:
                    pose_reward = torch.exp(-float(pose_reward_alpha) * pose_dist)
                    ##############################################
                    # Keep pose reward on the same length-normalized scale
                    # as dense reward when reward_norm is enabled.
                    ##############################################
                    if reward_norm:
                        pose_reward = pose_reward / float(target_len_20fps)
                    weighted_pose = float(pose_reward_weight) * pose_reward
                    frame_rewards[idx, :exec_len_20fps] += weighted_pose
                    pose_reward_sum += float(weighted_pose.sum().item())

            ##############################################
            # Add a GT velocity imitation reward on the same joint subset.
            # This penalizes static shortcuts by matching frame-to-frame
            # motion, while keeping the implementation aligned with pose reward.
            ##############################################
            if (
                velocity_reward_weight > 0.0
                and generated_joints_20fps is not None
                and gt_joints_20fps is not None
                and exec_len_20fps > 1
            ):
                if velocity_reward_joint_ids is None:
                    joint_ids = torch.arange(generated_joints_20fps.shape[2], dtype=torch.long, device=device)
                else:
                    joint_ids = torch.as_tensor(velocity_reward_joint_ids, dtype=torch.long, device=device)
                pred_joints = generated_joints_20fps[idx, :exec_len_20fps].index_select(1, joint_ids)
                gt_joints = gt_joints_20fps[idx, :exec_len_20fps].index_select(1, joint_ids)
                pred_vel = pred_joints[1:] - pred_joints[:-1]
                gt_vel = gt_joints[1:] - gt_joints[:-1]
                velocity_dist = ((pred_vel - gt_vel) ** 2).sum(dim=-1).mean(dim=-1)
                velocity_reward = torch.exp(-float(velocity_reward_alpha) * velocity_dist)
                ##############################################
                # Velocity reward is defined on frame differences, so its
                # natural normalization length is (L_gt - 1).
                ##############################################
                if reward_norm:
                    velocity_reward = velocity_reward / float(max(1, target_len_20fps - 1))
                weighted_velocity = float(velocity_reward_weight) * velocity_reward
                frame_rewards[idx, 1:exec_len_20fps] += weighted_velocity
                velocity_reward_sum += float(weighted_velocity.sum().item())
        else:
            undiscounted_sequence_rewards.append(0.0)

        if bool(episode["terminate"]):
            frame_rewards[idx, max(0, exec_len_20fps - 1)] += float(fail_penalty)
            terminate[idx] = 1.0
            terminal_reward_sum += float(fail_penalty)
        else:
            frame_rewards[idx, max(0, exec_len_20fps - 1)] += float(success_bonus)
            success[idx] = 1.0
            terminal_reward_sum += float(success_bonus)

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
        dense_reward_mean=(dense_reward_sum / float(batch_size)) if batch_size > 0 else 0.0,
        imitation_pose_reward_mean=(pose_reward_sum / float(batch_size)) if batch_size > 0 else 0.0,
        pose_mse_mean=(pose_mse_sum / float(pose_mse_count)) if pose_mse_count > 0 else 0.0,
        imitation_velocity_reward_mean=(velocity_reward_sum / float(batch_size)) if batch_size > 0 else 0.0,
        terminal_reward_mean=(terminal_reward_sum / float(batch_size)) if batch_size > 0 else 0.0,
        undiscounted_sequence_reward_mean=(
            float(np.mean(undiscounted_sequence_rewards)) if undiscounted_sequence_rewards else 0.0
        ),
        exec_ratio_mean=float(np.mean(exec_ratios)) if exec_ratios else 0.0,
    )
