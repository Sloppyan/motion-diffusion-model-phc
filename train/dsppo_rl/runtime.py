import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as sRot

if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.humanml_utils import HML_JOINT_NAMES
from data_loaders.tensors import lengths_to_mask
from train.dppo_frame_rl.data import PromptEntry, prompt_entries_to_meta
from train.dppo_frame_rl.diffusion.policy_wrappers import build_sampling_actor
from train.dppo_frame_rl.logging import merge_metrics, summarize_episodes
from train.dppo_frame_rl.phc_bridge import PHCExternalEvalBridge
from train.dppo_frame_rl.reward import build_frame_reward_batch
from train.dppo_frame_rl.storage import compute_frame_returns_and_advantages
from utils.model_util import create_model_and_diffusion, load_model_wo_clip


@dataclass
class DSPPORollout:
    texts: List[str]
    text_embeds: torch.Tensor
    lengths_20fps: torch.Tensor
    frame_noise: torch.Tensor
    frame_logprobs_old: torch.Tensor
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


class DSPPORuntime:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.device_id}")
        self.max_motion_frames = int(args.max_motion_frames)
        self.deterministic_denoising = bool(args.deterministic_denoising)
        self.stochastic_first_k_steps = max(0, int(getattr(args, "stochastic_first_k_steps", 0)))

        actor, diffusion = self._build_actor_and_diffusion(args.model_path)
        self.actor = actor
        self.diffusion = diffusion
        self.guidance_param = float(args.guidance_param)
        self.sampling_actor = build_sampling_actor(self.actor, self.guidance_param).to(self.device)
        self.sampling_actor.eval()

        data_root = Path(args.data_root).expanduser().resolve()
        self._motion_root = data_root / "new_joint_vecs"
        self._mean = torch.from_numpy(np.load(data_root / "Mean.npy")).float().to(self.device)
        self._std = torch.from_numpy(np.load(data_root / "Std.npy")).float().to(self.device)
        self._motion_cache: Dict[str, np.ndarray] = {}
        self._imitation_joint_ids = tuple(
            HML_JOINT_NAMES.index(name)
            for name in ("pelvis", "head", "left_ankle", "right_ankle", "left_wrist", "right_wrist")
        )

        self.phc_bridge = PHCExternalEvalBridge(
            phc_num_envs=args.phc_num_envs,
            phc_max_steps=args.phc_max_steps,
            device_id=args.device_id,
            actor_ckpt=args.phc_actor_ckpt,
        )

    def _build_actor_and_diffusion(self, model_path: str):
        model_path = Path(model_path).expanduser().resolve()
        args_path = model_path.parent / "args.json"
        if not args_path.is_file():
            raise FileNotFoundError(f"MDM args.json not found: {args_path}")
        with args_path.open("r", encoding="utf-8") as handle:
            model_args = json.load(handle)
        arg_namespace = SimpleNamespace(**model_args)
        dummy_data = SimpleNamespace(dataset=SimpleNamespace(num_actions=1))

        actor, diffusion = create_model_and_diffusion(arg_namespace, dummy_data)
        state_dict = torch.load(model_path, map_location="cpu")
        load_model_wo_clip(actor, state_dict)
        actor.to(self.device)
        actor.eval()
        for parameter in actor.parameters():
            parameter.requires_grad = False
        return actor, diffusion

    def _build_model_kwargs(
        self,
        texts: List[str],
        lengths_20fps: torch.Tensor,
        max_motion_frames: int,
        text_embeds: Optional[torch.Tensor] = None,
    ) -> Dict:
        lengths_20fps = lengths_20fps.to(self.device, dtype=torch.long)
        mask = lengths_to_mask(lengths_20fps, max_motion_frames).unsqueeze(1).unsqueeze(1)
        if text_embeds is None:
            text_embeds = self.sampling_actor.encode_text(texts)
        text_embeds = text_embeds.detach().clone().to(self.device)
        model_kwargs = {
            "y": {
                "text": list(texts),
                "text_embed": text_embeds,
                "lengths": lengths_20fps,
                "mask": mask,
            }
        }
        if self.guidance_param != 1.0:
            model_kwargs["y"]["scale"] = torch.full(
                (len(texts),),
                self.guidance_param,
                device=self.device,
                dtype=torch.float32,
            )
        return model_kwargs

    def _sample_clean_motion(
        self,
        x_t: torch.Tensor,
        model_kwargs: Dict,
        deterministic_denoising: bool,
        reverse_noise_seed: Optional[int] = None,
    ) -> torch.Tensor:
        ##############################################
        # Run the frozen reverse diffusion chain from x_T to x_0.
        # Deterministic mode removes the extra Gaussian at each reverse step.
        # If stochastic_first_k_steps > 0, only the first k reverse steps
        # inject noise and the rest become deterministic.
        ##############################################
        batch_size = x_t.shape[0]
        devices = [] if self.device.index is None else [self.device.index]
        num_timesteps = int(self.diffusion.num_timesteps)
        first_stochastic_step = max(1, num_timesteps - self.stochastic_first_k_steps)

        def _run_chain() -> torch.Tensor:
            with torch.no_grad():
                sample = x_t
                for step in reversed(range(self.diffusion.num_timesteps)):
                    t = torch.full((batch_size,), step, dtype=torch.long, device=self.device)
                    out = self.diffusion.p_mean_variance(
                        self.sampling_actor,
                        sample,
                        t,
                        clip_denoised=False,
                        model_kwargs=model_kwargs,
                    )
                    if self.stochastic_first_k_steps > 0:
                        should_sample_noise = step >= first_stochastic_step
                    else:
                        should_sample_noise = not deterministic_denoising

                    if should_sample_noise:
                        noise = torch.randn_like(sample)
                        nonzero_mask = (t != 0).float().view(-1, *([1] * (sample.dim() - 1)))
                        x_prev = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
                    else:
                        x_prev = out["mean"]
                    sample = x_prev
            return sample.detach()

        if reverse_noise_seed is None:
            return _run_chain()
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(reverse_noise_seed))
            return _run_chain()

    @staticmethod
    def _fps_20_to_30(joints: np.ndarray) -> np.ndarray:
        t = joints.shape[0]
        if t < 2:
            return joints.copy()
        target_t = int(round(t * 1.5))
        old_t = np.arange(t, dtype=np.float32)
        new_t = np.linspace(0, t - 1, target_t, dtype=np.float32)
        out = np.empty((target_t, joints.shape[1], 3), dtype=np.float32)
        for j in range(joints.shape[1]):
            out[:, j, 0] = np.interp(new_t, old_t, joints[:, j, 0])
            out[:, j, 1] = np.interp(new_t, old_t, joints[:, j, 1])
            out[:, j, 2] = np.interp(new_t, old_t, joints[:, j, 2])
        return out

    @staticmethod
    def _smpl22_to_smpl24(joints: np.ndarray, hand_len: float = 0.08824) -> np.ndarray:
        left_wrist = joints[:, 20, :]
        right_wrist = joints[:, 21, :]
        left_elbow = joints[:, 18, :]
        right_elbow = joints[:, 19, :]

        eps = 1e-8
        left_dir = left_wrist - left_elbow
        right_dir = right_wrist - right_elbow
        left_dir = left_dir / np.maximum(np.linalg.norm(left_dir, axis=-1, keepdims=True), eps)
        right_dir = right_dir / np.maximum(np.linalg.norm(right_dir, axis=-1, keepdims=True), eps)

        left_hand = left_wrist + left_dir * float(hand_len)
        right_hand = right_wrist + right_dir * float(hand_len)
        return np.concatenate([joints, left_hand[:, None, :], right_hand[:, None, :]], axis=1)

    @staticmethod
    def _postprocess_mdm_motion(joints: np.ndarray, offset_height: float = 0.92) -> np.ndarray:
        rot = sRot.from_euler("xyz", np.array([-np.pi / 2, 0, 0]), degrees=False).as_matrix()
        joints = np.matmul(joints, rot.dot(rot))
        offset = -offset_height - joints[0:1, 0:1, 1]
        joints[..., 1] += offset
        joints[..., [0, 2]] -= joints[:1, :1, [0, 2]]
        return DSPPORuntime._fps_20_to_30(joints.astype(np.float32))

    def _sample_to_reference_batch(
        self,
        final_sample: torch.Tensor,
        lengths_20fps: torch.Tensor,
    ) -> Tuple[np.ndarray, List[int]]:
        joints_22_np = self._sample_to_joint_batch(final_sample)

        motions: List[np.ndarray] = []
        lengths_30hz: List[int] = []
        for sample_idx, length in enumerate(lengths_20fps.tolist()):
            joints = joints_22_np[sample_idx, : int(length)]
            joints = self._smpl22_to_smpl24(joints)
            joints = self._postprocess_mdm_motion(joints)
            motions.append(joints)
            lengths_30hz.append(int(joints.shape[0]))

        max_len_30hz = max(lengths_30hz)
        batch = np.zeros((len(motions), max_len_30hz, 24, 3), dtype=np.float32)
        for sample_idx, motion in enumerate(motions):
            t = motion.shape[0]
            batch[sample_idx, :t] = motion
            if t < max_len_30hz and t > 0:
                batch[sample_idx, t:] = motion[t - 1]
        return batch, lengths_30hz

    def _sample_to_joint_batch(self, final_sample: torch.Tensor) -> np.ndarray:
        frame_features = final_sample.squeeze(2).permute(0, 2, 1).contiguous()
        denorm = frame_features * self._std.view(1, 1, -1) + self._mean.view(1, 1, -1)
        joints_22 = recover_from_ric(denorm, 22)
        return joints_22.detach().cpu().numpy().astype(np.float32)

    def _load_motion_vec(self, db_key: str) -> np.ndarray:
        motion = self._motion_cache.get(db_key)
        if motion is None:
            motion_path = self._motion_root / f"{db_key}.npy"
            if not motion_path.is_file():
                raise FileNotFoundError(f"GT motion file not found: {motion_path}")
            motion = np.load(motion_path, mmap_mode="r")
            self._motion_cache[db_key] = motion
        return motion

    def _load_gt_joint_batch(self, entries: Sequence[PromptEntry]) -> torch.Tensor:
        ##############################################
        # Recover GT joints from HumanML vectors in the same 20fps joint
        # space used by the generator output before PHC post-processing.
        ##############################################
        gt_vec = torch.zeros(
            (len(entries), self.max_motion_frames, self.actor.njoints * self.actor.nfeats),
            dtype=torch.float32,
            device=self.device,
        )
        for sample_idx, entry in enumerate(entries):
            motion = self._load_motion_vec(entry.db_key)
            start = int(max(0, min(entry.start_20fps, motion.shape[0] - 1)))
            end = int(min(motion.shape[0], start + entry.length_20fps))
            segment = np.asarray(motion[start:end], dtype=np.float32).copy()
            seg_len = min(int(segment.shape[0]), self.max_motion_frames)
            if seg_len > 0:
                gt_vec[sample_idx, :seg_len] = torch.from_numpy(segment[:seg_len]).to(self.device)
        return recover_from_ric(gt_vec, 22).detach()

    def _evaluate_reference_batch(
        self,
        ref_motion_batch: np.ndarray,
        lengths_30hz: List[int],
        entries: Sequence[PromptEntry],
    ) -> List[Dict]:
        ##############################################
        # PHC external eval uses one rollout budget for the whole queued batch.
        # Expand the budget on the fly so small-env debug runs do not stop early.
        ##############################################
        num_envs = max(1, int(self.args.phc_num_envs))
        max_len_30hz = max(lengths_30hz) if lengths_30hz else 0
        total_len_30hz = int(sum(lengths_30hz))
        required_steps = max(
            max_len_30hz,
            int(np.ceil(float(total_len_30hz) / float(num_envs))),
        ) + 8
        original_max_steps = self.phc_bridge.phc_max_steps
        self.phc_bridge.phc_max_steps = max(original_max_steps, required_steps)
        try:
            return self.phc_bridge.evaluate_batch(
                ref_motion_batch=ref_motion_batch,
                lengths_30hz=lengths_30hz,
                meta=prompt_entries_to_meta(entries),
            )
        finally:
            self.phc_bridge.phc_max_steps = original_max_steps

    def _generate_batch(
        self,
        entries: Sequence[PromptEntry],
        noise_policy: nn.Module,
        policy_deterministic: bool = False,
        sampling_seed: Optional[int] = None,
        initial_noise_seed: Optional[int] = None,
        reverse_noise_seed: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        texts = [entry.caption for entry in entries]
        lengths_20fps = torch.tensor([entry.length_20fps for entry in entries], dtype=torch.long, device=self.device)
        frame_indices = torch.arange(self.max_motion_frames, dtype=torch.long, device=self.device)
        resolved_initial_seed = initial_noise_seed if initial_noise_seed is not None else sampling_seed
        resolved_reverse_seed = reverse_noise_seed if reverse_noise_seed is not None else sampling_seed

        def _sample_frame_noise(text_embeds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            devices = [] if self.device.index is None else [self.device.index]
            if resolved_initial_seed is None:
                return noise_policy.sample(
                    text_embeds=text_embeds,
                    frame_indices=frame_indices,
                    deterministic=policy_deterministic,
                )
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(resolved_initial_seed))
                return noise_policy.sample(
                    text_embeds=text_embeds,
                    frame_indices=frame_indices,
                    deterministic=policy_deterministic,
                )

        def _sample_once() -> Dict[str, torch.Tensor]:
            with torch.no_grad():
                text_embeds = self.actor.encode_text(texts)
                ##############################################
                # Split RNG control for x_T sampling and reverse noise.
                # This lets diagnostics hold one source fixed and vary the other.
                ##############################################
                frame_noise, frame_logprobs_old = _sample_frame_noise(text_embeds)
                model_kwargs = self._build_model_kwargs(
                    texts=texts,
                    lengths_20fps=lengths_20fps,
                    max_motion_frames=self.max_motion_frames,
                    text_embeds=text_embeds,
                )
                x_t = frame_noise.permute(0, 2, 1).unsqueeze(2).contiguous()
                final_sample = self._sample_clean_motion(
                    x_t=x_t,
                    model_kwargs=model_kwargs,
                    deterministic_denoising=self.deterministic_denoising,
                    reverse_noise_seed=resolved_reverse_seed,
                )
            return {
                "texts": texts,
                "text_embeds": text_embeds.detach(),
                "lengths_20fps": lengths_20fps,
                "frame_noise": frame_noise.detach(),
                "frame_logprobs_old": frame_logprobs_old.detach(),
                "final_sample": final_sample.detach(),
                "frame_features": final_sample.squeeze(2).permute(0, 2, 1).contiguous().detach(),
            }

        return _sample_once()

    def collect_rollout(
        self,
        entries: Sequence[PromptEntry],
        noise_policy: nn.Module,
        critic: nn.Module,
        policy_deterministic: bool = False,
        sampling_seed: Optional[int] = None,
    ) -> Tuple[DSPPORollout, Dict[str, float]]:
        sampled = self._generate_batch(
            entries=entries,
            noise_policy=noise_policy,
            policy_deterministic=policy_deterministic,
            sampling_seed=sampling_seed,
        )
        generated_joints = torch.from_numpy(self._sample_to_joint_batch(sampled["final_sample"])).to(self.device)
        gt_joints = self._load_gt_joint_batch(entries)

        ref_motion_batch, lengths_30hz = self._sample_to_reference_batch(sampled["final_sample"], sampled["lengths_20fps"])
        episodes = self._evaluate_reference_batch(
            ref_motion_batch=ref_motion_batch,
            lengths_30hz=lengths_30hz,
            entries=entries,
        )

        reward_batch = build_frame_reward_batch(
            episodes=episodes,
            max_motion_frames=self.max_motion_frames,
            gamma_frame=self.args.frame_gamma,
            dense_reward_weight=self.args.dense_reward_weight,
            reward_norm=self.args.reward_norm,
            success_bonus=self.args.success_bonus,
            fail_penalty=self.args.fail_penalty,
            device=self.device,
            generated_joints_20fps=generated_joints,
            gt_joints_20fps=gt_joints,
            pose_reward_weight=getattr(self.args, "pose_reward_weight", 0.0),
            pose_reward_alpha=getattr(self.args, "pose_reward_alpha", 1.0),
            pose_reward_joint_ids=self._imitation_joint_ids,
            velocity_reward_weight=getattr(self.args, "velocity_reward_weight", 0.0),
            velocity_reward_alpha=getattr(self.args, "velocity_reward_alpha", 1.0),
            velocity_reward_joint_ids=self._imitation_joint_ids,
        )

        with torch.no_grad():
            frame_values = critic(sampled["frame_features"], sampled["text_embeds"])
        gae = compute_frame_returns_and_advantages(
            rewards=reward_batch.frame_rewards,
            values=frame_values,
            dones=reward_batch.frame_dones,
            exec_mask=reward_batch.frame_exec_mask,
            gamma=self.args.frame_gamma,
            gae_lambda=self.args.frame_lambda,
        )

        rollout = DSPPORollout(
            texts=sampled["texts"],
            text_embeds=sampled["text_embeds"],
            lengths_20fps=sampled["lengths_20fps"],
            frame_noise=sampled["frame_noise"],
            frame_logprobs_old=sampled["frame_logprobs_old"],
            frame_features=sampled["frame_features"],
            frame_rewards=reward_batch.frame_rewards,
            frame_exec_mask=reward_batch.frame_exec_mask,
            frame_dones=reward_batch.frame_dones,
            frame_values=frame_values,
            frame_advantages=gae["advantages"],
            frame_returns=gae["returns"],
            final_sample=sampled["final_sample"],
            success=reward_batch.success,
            terminate=reward_batch.terminate,
            episodes=episodes,
        )
        metrics = merge_metrics(
            summarize_episodes(episodes),
            {
                "dense_reward_mean": reward_batch.dense_reward_mean,
                "imitation_pose_reward_mean": reward_batch.imitation_pose_reward_mean,
                "pose_mse_mean": reward_batch.pose_mse_mean,
                "imitation_velocity_reward_mean": reward_batch.imitation_velocity_reward_mean,
                "terminal_reward_mean": reward_batch.terminal_reward_mean,
                "undiscounted_sequence_reward_mean": reward_batch.undiscounted_sequence_reward_mean,
                "exec_ratio_mean": reward_batch.exec_ratio_mean,
            },
        )
        return rollout, metrics

    def evaluate_entries(
        self,
        entries: Sequence[PromptEntry],
        noise_policy: nn.Module,
        policy_deterministic: bool = False,
        sampling_seed: Optional[int] = None,
    ) -> Dict[str, float]:
        sampled = self._generate_batch(
            entries=entries,
            noise_policy=noise_policy,
            policy_deterministic=policy_deterministic,
            sampling_seed=sampling_seed,
        )
        generated_joints = torch.from_numpy(self._sample_to_joint_batch(sampled["final_sample"])).to(self.device)
        gt_joints = self._load_gt_joint_batch(entries)
        ref_motion_batch, lengths_30hz = self._sample_to_reference_batch(sampled["final_sample"], sampled["lengths_20fps"])
        episodes = self._evaluate_reference_batch(
            ref_motion_batch=ref_motion_batch,
            lengths_30hz=lengths_30hz,
            entries=entries,
        )
        reward_batch = build_frame_reward_batch(
            episodes=episodes,
            max_motion_frames=self.max_motion_frames,
            gamma_frame=self.args.frame_gamma,
            dense_reward_weight=self.args.dense_reward_weight,
            reward_norm=self.args.reward_norm,
            success_bonus=self.args.success_bonus,
            fail_penalty=self.args.fail_penalty,
            device=self.device,
            generated_joints_20fps=generated_joints,
            gt_joints_20fps=gt_joints,
            pose_reward_weight=getattr(self.args, "pose_reward_weight", 0.0),
            pose_reward_alpha=getattr(self.args, "pose_reward_alpha", 1.0),
            pose_reward_joint_ids=self._imitation_joint_ids,
            velocity_reward_weight=getattr(self.args, "velocity_reward_weight", 0.0),
            velocity_reward_alpha=getattr(self.args, "velocity_reward_alpha", 1.0),
            velocity_reward_joint_ids=self._imitation_joint_ids,
        )
        return merge_metrics(
            summarize_episodes(episodes),
            {
                "dense_reward_mean": reward_batch.dense_reward_mean,
                "imitation_pose_reward_mean": reward_batch.imitation_pose_reward_mean,
                "pose_mse_mean": reward_batch.pose_mse_mean,
                "imitation_velocity_reward_mean": reward_batch.imitation_velocity_reward_mean,
                "terminal_reward_mean": reward_batch.terminal_reward_mean,
                "undiscounted_sequence_reward_mean": reward_batch.undiscounted_sequence_reward_mean,
                "exec_ratio_mean": reward_batch.exec_ratio_mean,
            },
        )

    def close(self) -> None:
        self.phc_bridge.close()
