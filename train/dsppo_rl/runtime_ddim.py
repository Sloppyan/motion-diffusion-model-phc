from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from train.dppo_frame_rl.logging import merge_metrics, summarize_episodes
from train.dppo_frame_rl.reward import build_frame_reward_batch
from train.dppo_frame_rl.storage import compute_frame_returns_and_advantages
from train.dsppo_rl.runtime import DSPPORollout, DSPPORuntime


@dataclass
class DSPPODDIMInversionRollout(DSPPORollout):
    frame_noise_base: torch.Tensor
    frame_noise_gt_inv: torch.Tensor


class DSPPODDIMInversionRuntime(DSPPORuntime):
    def __init__(self, args):
        if not hasattr(args, "deterministic_denoising"):
            args.deterministic_denoising = True
        if not hasattr(args, "stochastic_first_k_steps"):
            args.stochastic_first_k_steps = 0
        super().__init__(args)
        self.ddim_eta = float(args.ddim_eta)
        if self.ddim_eta != 0.0:
            raise ValueError("This runtime only supports deterministic DDIM (ddim_eta must be 0.0).")
        self._gt_inversion_cache: Dict[Tuple[str, int, int, int], torch.Tensor] = {}

    def _sample_clean_motion(
        self,
        x_t: torch.Tensor,
        model_kwargs: Dict,
        deterministic_denoising: bool,
        reverse_noise_seed: Optional[int] = None,
    ) -> torch.Tensor:
        del deterministic_denoising
        del reverse_noise_seed
        with torch.no_grad():
            return self.diffusion.ddim_sample_loop(
                self.sampling_actor,
                shape=tuple(x_t.shape),
                noise=x_t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                device=self.device,
                progress=False,
                eta=self.ddim_eta,
            ).detach()

    def _generate_batch(
        self,
        entries: Sequence,
        noise_policy: nn.Module,
        policy_deterministic: bool = False,
        sampling_seed: Optional[int] = None,
        initial_noise_seed: Optional[int] = None,
        reverse_noise_seed: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        del reverse_noise_seed
        texts = [entry.caption for entry in entries]
        lengths_20fps = torch.tensor([entry.length_20fps for entry in entries], dtype=torch.long, device=self.device)
        frame_indices = torch.arange(self.max_motion_frames, dtype=torch.long, device=self.device)
        resolved_initial_seed = initial_noise_seed if initial_noise_seed is not None else sampling_seed

        def _sample_frame_noise(text_embeds: torch.Tensor):
            devices = [] if self.device.index is None else [self.device.index]
            if resolved_initial_seed is None:
                return noise_policy.sample(
                    text_embeds=text_embeds,
                    frame_indices=frame_indices,
                    deterministic=policy_deterministic,
                    return_base_noise=True,
                )
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(resolved_initial_seed))
                return noise_policy.sample(
                    text_embeds=text_embeds,
                    frame_indices=frame_indices,
                    deterministic=policy_deterministic,
                    return_base_noise=True,
                )

        with torch.no_grad():
            text_embeds = self.actor.encode_text(texts)
            frame_noise, frame_logprobs_old, frame_noise_base = _sample_frame_noise(text_embeds)
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
                deterministic_denoising=True,
            )

        return {
            "texts": texts,
            "text_embeds": text_embeds.detach(),
            "lengths_20fps": lengths_20fps,
            "frame_noise": frame_noise.detach(),
            "frame_noise_base": frame_noise_base.detach(),
            "frame_logprobs_old": frame_logprobs_old.detach(),
            "final_sample": final_sample.detach(),
            "frame_features": final_sample.squeeze(2).permute(0, 2, 1).contiguous().detach(),
        }

    def _gt_cache_key(self, entry) -> Tuple[str, int, int, int]:
        return (entry.db_key, int(entry.caption_idx), int(entry.start_20fps), int(entry.length_20fps))

    def _build_gt_x0_batch(self, entries: Sequence) -> torch.Tensor:
        feature_dim = self.actor.njoints * self.actor.nfeats
        gt_norm = torch.zeros(
            (len(entries), self.max_motion_frames, feature_dim),
            dtype=torch.float32,
            device=self.device,
        )
        mean = self._mean.view(1, 1, -1)
        std = self._std.view(1, 1, -1)

        ##############################################
        # Build normalized x_0 in the exact feature space used
        # by the diffusion model before sampling/inversion.
        ##############################################
        for sample_idx, entry in enumerate(entries):
            motion = self._load_motion_vec(entry.db_key)
            start = int(max(0, min(entry.start_20fps, motion.shape[0] - 1)))
            end = int(min(motion.shape[0], start + entry.length_20fps))
            segment = np.asarray(motion[start:end], dtype=np.float32).copy()
            seg_len = min(int(segment.shape[0]), self.max_motion_frames)
            if seg_len > 0:
                gt_norm[sample_idx, :seg_len] = torch.from_numpy(segment[:seg_len]).to(self.device)

        gt_norm = (gt_norm - mean) / std
        return gt_norm.permute(0, 2, 1).unsqueeze(2).contiguous()

    def _invert_gt_batch(self, entries: Sequence) -> torch.Tensor:
        texts = [entry.caption for entry in entries]
        lengths_20fps = torch.tensor([entry.length_20fps for entry in entries], dtype=torch.long, device=self.device)
        model_kwargs = self._build_model_kwargs(
            texts=texts,
            lengths_20fps=lengths_20fps,
            max_motion_frames=self.max_motion_frames,
        )
        sample = self._build_gt_x0_batch(entries)
        with torch.no_grad():
            ##############################################
            # Follow the exact DDIM reverse-ODE schedule used by rollout.
            # Starting from x_0^{gt}, march forward deterministically to x_T.
            ##############################################
            for step in range(int(self.diffusion.num_timesteps)):
                t = torch.full((sample.shape[0],), step, dtype=torch.long, device=self.device)
                out = self.diffusion.ddim_reverse_sample(
                    self.sampling_actor,
                    sample,
                    t,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    eta=self.ddim_eta,
                )
                sample = out["sample"]
        return sample.squeeze(2).permute(0, 2, 1).contiguous().detach()

    def _load_gt_inversion_noise_batch(self, entries: Sequence) -> torch.Tensor:
        feature_dim = self.actor.njoints * self.actor.nfeats
        batch = torch.zeros(
            (len(entries), self.max_motion_frames, feature_dim),
            dtype=torch.float32,
            device=self.device,
        )

        missing_entries = []
        missing_indices = []
        for sample_idx, entry in enumerate(entries):
            cached = self._gt_inversion_cache.get(self._gt_cache_key(entry))
            if cached is None:
                missing_entries.append(entry)
                missing_indices.append(sample_idx)
            else:
                batch[sample_idx] = cached.to(self.device)

        if missing_entries:
            inverted = self._invert_gt_batch(missing_entries)
            for local_idx, sample_idx in enumerate(missing_indices):
                cached = inverted[local_idx].detach().cpu()
                self._gt_inversion_cache[self._gt_cache_key(entries[sample_idx])] = cached
                batch[sample_idx] = cached.to(self.device)

        return batch

    def collect_rollout(
        self,
        entries: Sequence,
        noise_policy: nn.Module,
        critic: nn.Module,
        policy_deterministic: bool = False,
        sampling_seed: Optional[int] = None,
    ):
        sampled = self._generate_batch(
            entries=entries,
            noise_policy=noise_policy,
            policy_deterministic=policy_deterministic,
            sampling_seed=sampling_seed,
        )
        frame_noise_gt_inv = self._load_gt_inversion_noise_batch(entries)
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

        rollout = DSPPODDIMInversionRollout(
            texts=sampled["texts"],
            text_embeds=sampled["text_embeds"],
            lengths_20fps=sampled["lengths_20fps"],
            frame_noise=sampled["frame_noise"],
            frame_noise_base=sampled["frame_noise_base"],
            frame_noise_gt_inv=frame_noise_gt_inv,
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

