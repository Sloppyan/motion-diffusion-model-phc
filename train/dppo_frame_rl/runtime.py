import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot

if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

from data_loaders.humanml.scripts.motion_process import recover_from_ric
from train.dppo_frame_rl.data import PromptEntry, prompt_entries_to_meta
from train.dppo_frame_rl.diffusion import FramePPODiffusion
from train.dppo_frame_rl.logging import merge_metrics, summarize_episodes
from train.dppo_frame_rl.phc_bridge import PHCExternalEvalBridge
from train.dppo_frame_rl.reward import build_frame_reward_batch
from train.dppo_frame_rl.storage import FrameRollout, compute_frame_returns_and_advantages
from utils.model_util import create_model_and_diffusion, load_model_wo_clip


class FrameDPPORuntime:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.device_id}")
        self.max_motion_frames = int(args.max_motion_frames)

        actor, diffusion = self._build_actor_and_diffusion(args.model_path)
        self.policy = FramePPODiffusion(
            actor=actor,
            diffusion=diffusion,
            guidance_param=args.guidance_param,
            ft_denoising_steps=args.ft_denoising_steps,
            min_sampling_denoising_std=args.min_sampling_denoising_std,
            min_logprob_denoising_std=args.min_logprob_denoising_std,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_layer_scope=args.lora_layer_scope,
            resume_lora_path=args.resume_lora_path,
            gamma_denoising=args.gamma_denoising,
            clip_ploss_coef=args.clip_range,
            clip_ploss_schedule=args.clip_range_schedule,
            clip_ploss_coef_base=args.clip_range_base,
            clip_ploss_coef_rate=args.clip_range_rate,
            kl_coef=args.kl_coef,
            norm_adv=False,
        )

        data_root = Path(args.data_root).expanduser().resolve()
        self._mean = torch.from_numpy(np.load(data_root / "Mean.npy")).float().to(self.device)
        self._std = torch.from_numpy(np.load(data_root / "Std.npy")).float().to(self.device)

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
        return actor, diffusion

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
        return FrameDPPORuntime._fps_20_to_30(joints.astype(np.float32))

    def _sample_to_reference_batch(
        self,
        final_sample: torch.Tensor,
        lengths_20fps: torch.Tensor,
    ) -> Tuple[np.ndarray, List[int]]:
        frame_features = final_sample.squeeze(2).permute(0, 2, 1).contiguous()
        denorm = frame_features * self._std.view(1, 1, -1) + self._mean.view(1, 1, -1)
        joints_22 = recover_from_ric(denorm, 22)
        joints_22_np = joints_22.detach().cpu().numpy()

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

    def collect_rollout(
        self,
        entries: Sequence[PromptEntry],
        critic,
        deterministic: bool = False,
        sampling_seed: Optional[int] = None,
    ) -> Tuple[FrameRollout, Dict[str, float]]:
        texts = [entry.caption for entry in entries]
        lengths_20fps = torch.tensor([entry.length_20fps for entry in entries], dtype=torch.long, device=self.device)

        if sampling_seed is None:
            sampled = self.policy.sample(
                texts=texts,
                lengths_20fps=lengths_20fps,
                max_motion_frames=self.max_motion_frames,
                deterministic=deterministic,
            )
        else:
            with torch.random.fork_rng(devices=[self.device.index]):
                torch.manual_seed(int(sampling_seed))
                sampled = self.policy.sample(
                    texts=texts,
                    lengths_20fps=lengths_20fps,
                    max_motion_frames=self.max_motion_frames,
                    deterministic=deterministic,
                )

        ref_motion_batch, lengths_30hz = self._sample_to_reference_batch(sampled["final_sample"], lengths_20fps)
        episodes = self.phc_bridge.evaluate_batch(
            ref_motion_batch=ref_motion_batch,
            lengths_30hz=lengths_30hz,
            meta=prompt_entries_to_meta(entries),
        )

        reward_batch = build_frame_reward_batch(
            episodes=episodes,
            max_motion_frames=self.max_motion_frames,
            gamma_frame=self.args.frame_gamma,
            dense_reward_weight=self.args.dense_reward_weight,
            dense_reward_norm=self.args.dense_reward_norm,
            success_bonus=self.args.success_bonus,
            fail_penalty=self.args.fail_penalty,
            device=self.device,
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

        rollout = FrameRollout(
            texts=texts,
            text_embeds=sampled["text_embeds"],
            lengths_20fps=lengths_20fps,
            frame_features=sampled["frame_features"],
            frame_rewards=reward_batch.frame_rewards,
            frame_exec_mask=reward_batch.frame_exec_mask,
            frame_dones=reward_batch.frame_dones,
            frame_values=frame_values,
            frame_advantages=gae["advantages"],
            frame_returns=gae["returns"],
            chain_prev=sampled["chain_prev"],
            chain_next=sampled["chain_next"],
            final_sample=sampled["final_sample"],
            timesteps=sampled["timesteps"],
            frame_logprobs_old=sampled["frame_logprobs_old"],
            success=reward_batch.success,
            terminate=reward_batch.terminate,
            episodes=episodes,
            trainable_step_frac=sampled["trainable_step_frac"],
        )

        metrics = merge_metrics(
            summarize_episodes(episodes),
            {
                "dense_reward_mean": reward_batch.dense_reward_mean,
                "imitation_pose_reward_mean": reward_batch.imitation_pose_reward_mean,
                "imitation_velocity_reward_mean": reward_batch.imitation_velocity_reward_mean,
                "terminal_reward_mean": reward_batch.terminal_reward_mean,
                "failure_penalty_mean": reward_batch.failure_penalty_mean,
                "undiscounted_sequence_reward_mean": reward_batch.undiscounted_sequence_reward_mean,
                "exec_ratio_mean": reward_batch.exec_ratio_mean,
            },
        )
        return rollout, metrics

    def evaluate_entries(
        self,
        entries: Sequence[PromptEntry],
        deterministic: bool = False,
        sampling_seed: Optional[int] = None,
    ) -> Dict[str, float]:
        dummy_critic = lambda frame_features, text_embeds: torch.zeros(
            frame_features.shape[:2],
            dtype=frame_features.dtype,
            device=frame_features.device,
        )
        _, metrics = self.collect_rollout(
            entries=entries,
            critic=dummy_critic,
            deterministic=deterministic,
            sampling_seed=sampling_seed,
        )
        return metrics

    def close(self) -> None:
        self.phc_bridge.close()
