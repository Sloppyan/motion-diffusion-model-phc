from __future__ import annotations

from dataclasses import dataclass

import torch

from mdm_core.data_loaders import humanml_utils
from mdm_core.data_loaders.humanml.scripts.motion_process import recover_from_ric


@dataclass(frozen=True)
class FootGroundGuidanceConfig:
    last_steps: int = 10
    tau: float = 0.005
    contact_k: float = 50.0
    lambda_pen: float = 1.0
    lambda_float: float = 1.0
    lambda_skate: float = 2.0
    step_size: float = 1e-4
    grad_clip: float = 0.0


class FootGroundGuidance:
    def __init__(self, mean, std, config: FootGroundGuidanceConfig):
        if config.last_steps < 0:
            raise ValueError("last_steps must be non-negative.")
        if config.step_size < 0:
            raise ValueError("step_size must be non-negative.")

        mean_tensor = torch.as_tensor(mean, dtype=torch.float32).view(1, 1, 1, -1)
        std_tensor = torch.as_tensor(std, dtype=torch.float32).view(1, 1, 1, -1)
        if mean_tensor.shape != std_tensor.shape:
            raise ValueError("mean/std shapes must match.")
        if mean_tensor.shape[-1] != 263:
            raise ValueError("Foot-ground guidance currently supports HumanML only.")

        self.config = config
        self.mean = mean_tensor
        self.std = std_tensor
        self.joints_num = humanml_utils.NUM_HML_JOINTS
        self.foot_joint_indices = tuple(
            humanml_utils.HML_JOINT_NAMES.index(name)
            for name in ("left_foot", "right_foot")
        )
        self.eps = 1e-6
        self.mm_scale = 1000.0

    def __call__(self, pred_xstart, t=None, model_kwargs=None):
        if self.config.step_size == 0 or self.config.last_steps == 0:
            return pred_xstart
        if t is None or int(t.min().item()) >= self.config.last_steps:
            return pred_xstart

        with torch.enable_grad():
            guided = pred_xstart.detach().requires_grad_(True)
            xyz_joints = self._pred_to_xyz(guided)
            lengths = self._resolve_lengths(model_kwargs, guided.shape[0], guided.shape[-1], guided.device)
            loss = self._physical_loss(xyz_joints, lengths)
            if not torch.isfinite(loss):
                return pred_xstart

            grad = torch.autograd.grad(loss, guided, allow_unused=False)[0]
            grad = self._sanitize_grad(grad)
            grad = self._clip_grad(grad)
            guided = guided - self.config.step_size * grad
        return guided.detach()

    def _resolve_lengths(self, model_kwargs, batch_size, num_frames, device):
        lengths = None
        if model_kwargs is not None:
            y_kwargs = model_kwargs.get("y")
            if isinstance(y_kwargs, dict):
                lengths = y_kwargs.get("lengths")
        if lengths is None:
            return torch.full((batch_size,), num_frames, device=device, dtype=torch.long)
        return torch.as_tensor(lengths, device=device, dtype=torch.long).clamp_(min=1, max=num_frames)

    def _pred_to_xyz(self, pred_xstart):
        if pred_xstart.shape[1] != self.mean.shape[-1] or pred_xstart.shape[2] != 1:
            raise ValueError("Foot-ground guidance expects HumanML predictions shaped as [B, 263, 1, T].")

        # Convert normalized HumanML vectors back to xyz joints using the exact
        # same denorm + recover path already used by sample/generate.py.
        motion = pred_xstart.permute(0, 2, 3, 1)
        motion = motion * self.std.to(device=motion.device, dtype=motion.dtype)
        motion = motion + self.mean.to(device=motion.device, dtype=motion.dtype)
        xyz = recover_from_ric(motion, self.joints_num)
        return xyz.squeeze(1).permute(0, 2, 3, 1).contiguous()

    def _physical_loss(self, xyz_joints, lengths):
        frame_mask = self._frame_mask(lengths, xyz_joints.shape[-1], xyz_joints.device, xyz_joints.dtype)
        frame_count = frame_mask.sum().clamp_min(1.0)

        joint_heights = xyz_joints[:, :, 1, :]
        lowest_heights = joint_heights.amin(dim=1)

        pen_loss = torch.relu(-lowest_heights)
        pen_loss = (pen_loss * frame_mask).sum() / frame_count

        float_loss = torch.relu(lowest_heights - self.config.tau)
        float_loss = (float_loss * frame_mask).sum() / frame_count

        foot_joints = xyz_joints[:, self.foot_joint_indices]
        foot_heights = foot_joints[:, :, 1, :]
        foot_contact = torch.sigmoid(self.config.contact_k * (self.config.tau - foot_heights))

        transition_mask = (frame_mask[:, :-1] * frame_mask[:, 1:]).unsqueeze(1)
        contact_pairs = foot_contact[:, :, :-1] * foot_contact[:, :, 1:]
        contact_pairs = contact_pairs * transition_mask

        foot_offsets = foot_joints[:, :, [0, 2], 1:] - foot_joints[:, :, [0, 2], :-1]
        foot_sliding = torch.norm(foot_offsets, dim=2)
        skate_num = (contact_pairs * foot_sliding).sum()
        skate_den = contact_pairs.sum().clamp_min(self.eps)
        skate_loss = skate_num / skate_den

        pen_loss = pen_loss * self.mm_scale
        float_loss = float_loss * self.mm_scale
        skate_loss = skate_loss * self.mm_scale

        return (
            self.config.lambda_pen * pen_loss
            + self.config.lambda_float * float_loss
            + self.config.lambda_skate * skate_loss
        )

    def _frame_mask(self, lengths, num_frames, device, dtype):
        arange = torch.arange(num_frames, device=device)
        return (arange.unsqueeze(0) < lengths.unsqueeze(1)).to(dtype=dtype)

    def _clip_grad(self, grad):
        if self.config.grad_clip <= 0:
            return grad
        flat_grad = grad.flatten(1)
        grad_norm = torch.norm(flat_grad, dim=1, keepdim=True).clamp_min(self.eps)
        scale = torch.clamp(self.config.grad_clip / grad_norm, max=1.0)
        return grad * scale.view(-1, 1, 1, 1)

    def _sanitize_grad(self, grad):
        finite_mask = torch.isfinite(grad)
        if finite_mask.all():
            return grad
        return grad.masked_fill(~finite_mask, 0.0)
