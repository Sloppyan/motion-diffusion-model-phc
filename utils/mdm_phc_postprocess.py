import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from mdm_core.data_loaders.humanml.scripts.motion_process import recover_from_ric


def load_humanml_stats(data_root: str) -> Tuple[np.ndarray, np.ndarray]:
    root = Path(data_root)
    mean = np.load(root / "Mean.npy").astype(np.float32)
    std = np.load(root / "Std.npy").astype(np.float32)
    return mean, std


class HumanML3DPostprocessor:
    def __init__(self, model, data_root: str, offset_height: float = 0.92):
        self.model = model
        self.offset_height = float(offset_height)
        mean, std = load_humanml_stats(data_root)
        self.mean = torch.from_numpy(mean).float()
        self.std = torch.from_numpy(std).float()

    def _inv_transform(self, sample: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(sample.device, dtype=sample.dtype)
        std = self.std.to(sample.device, dtype=sample.dtype)
        return sample * std + mean

    def sample_to_joints24_20fps(
        self,
        sample: torch.Tensor,
        lengths: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.model.data_rep != "hml_vec":
            raise NotImplementedError("Only HumanML3D hml_vec output is supported.")

        n_joints = 22 if sample.shape[1] == 263 else 21
        sample_hml = self._inv_transform(sample.permute(0, 2, 3, 1))
        sample_22 = recover_from_ric(sample_hml, n_joints)
        sample_22 = sample_22.view(-1, *sample_22.shape[2:]).permute(0, 2, 3, 1)

        rot2xyz_pose_rep = "xyz"
        rot2xyz_mask = mask.reshape(sample.shape[0], sample.shape[-1]).bool()
        sample_xyz = self.model.rot2xyz(
            x=sample_22,
            mask=rot2xyz_mask,
            pose_rep=rot2xyz_pose_rep,
            glob=True,
            translation=True,
            jointstype="smpl",
            vertstrans=True,
            betas=None,
            beta=0,
            glob_rot=None,
            get_rotations_back=False,
        )
        sample_xyz = sample_xyz.permute(0, 3, 1, 2)
        return smpl22_to_smpl24(sample_xyz)

    def joints24_20fps_to_phc_ref(self, joints24_20fps: torch.Tensor, lengths: torch.Tensor):
        motions = joints24_20fps.detach().cpu().numpy()
        lengths_np = lengths.detach().cpu().numpy().astype(np.int64)
        refs: List[np.ndarray] = []
        ref_lengths: List[int] = []
        for motion, length in zip(motions, lengths_np):
            motion = motion[: max(1, int(length))]
            ref30 = postprocess_motion_for_phc(motion, offset_height=self.offset_height)
            refs.append(ref30)
            ref_lengths.append(ref30.shape[0])
        padded = pad_motion_batch(refs)
        return padded, np.asarray(ref_lengths, dtype=np.int64)


def smpl22_to_smpl24(joints22_20fps: torch.Tensor, hand_len: float = 0.08824) -> torch.Tensor:
    if joints22_20fps.shape[2] != 22:
        raise ValueError(f"Expected 22 joints, got {joints22_20fps.shape}")

    eps = 1e-8
    left_dir = joints22_20fps[:, :, -2] - joints22_20fps[:, :, -4]
    right_dir = joints22_20fps[:, :, -1] - joints22_20fps[:, :, -3]
    left_dir = left_dir / left_dir.norm(dim=-1, keepdim=True).clamp_min(eps)
    right_dir = right_dir / right_dir.norm(dim=-1, keepdim=True).clamp_min(eps)

    left_hand = joints22_20fps[:, :, -2] + left_dir * float(hand_len)
    right_hand = joints22_20fps[:, :, -1] + right_dir * float(hand_len)
    return torch.cat([joints22_20fps, left_hand[:, :, None], right_hand[:, :, None]], dim=2)


def postprocess_motion_for_phc(joints24_20fps: np.ndarray, offset_height: float = 0.92) -> np.ndarray:
    if joints24_20fps.ndim != 3 or joints24_20fps.shape[1:] != (24, 3):
        raise ValueError(f"Expected [T,24,3], got {joints24_20fps.shape}")

    rot = rotation_x(-math.pi / 2)
    rot2 = rot.dot(rot)
    motion = np.matmul(joints24_20fps.astype(np.float32), rot2.astype(np.float32))

    offset = -float(offset_height) - motion[0:1, 0:1, 1]
    motion[..., 1] += offset
    motion[..., [0, 2]] -= motion[:1, :1, [0, 2]]
    return fps_20_to_30(motion)


def rotation_x(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


def fps_20_to_30(joints24_20fps: np.ndarray) -> np.ndarray:
    t = joints24_20fps.shape[0]
    if t < 2:
        return joints24_20fps.copy()
    target_t = int(round(t * 1.5))
    old_t = np.arange(t, dtype=np.float32)
    new_t = np.linspace(0, t - 1, target_t, dtype=np.float32)
    out = np.empty((target_t, joints24_20fps.shape[1], 3), dtype=np.float32)
    for joint_idx in range(joints24_20fps.shape[1]):
        for axis in range(3):
            out[:, joint_idx, axis] = np.interp(new_t, old_t, joints24_20fps[:, joint_idx, axis])
    return out


def pad_motion_batch(motions: Sequence[np.ndarray]) -> np.ndarray:
    if len(motions) == 0:
        return np.zeros((0, 1, 24, 3), dtype=np.float32)
    max_t = max(max(1, motion.shape[0]) for motion in motions)
    out = np.empty((len(motions), max_t, 24, 3), dtype=np.float32)
    for idx, motion in enumerate(motions):
        t = motion.shape[0]
        out[idx, :t] = motion
        out[idx, t:] = motion[t - 1 : t]
    return out
