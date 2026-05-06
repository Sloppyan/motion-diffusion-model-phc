from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from train.dppo_frame_rl.logging import merge_metrics, summarize_episodes
from train.dppo_frame_rl.reward import build_frame_reward_batch
from train.dppo_frame_rl.storage import compute_frame_returns_and_advantages
from train.drppo_rl.storage import DRPPORollout
from train.dsppo_rl.runtime import DSPPORuntime


class DRPPORuntime(DSPPORuntime):
    def __init__(self, args):
        if not hasattr(args, "deterministic_denoising"):
            args.deterministic_denoising = False
        if not hasattr(args, "stochastic_first_k_steps"):
            args.stochastic_first_k_steps = 0
        super().__init__(args)
        self.num_controlled_reverse_steps = int(args.num_controlled_reverse_steps)

    def _controlled_timesteps_tensor(self) -> torch.Tensor:
        max_controlled = max(0, int(self.diffusion.num_timesteps) - 1)
        num_controlled = min(int(self.num_controlled_reverse_steps), max_controlled)
        timesteps = list(range(int(self.diffusion.num_timesteps) - 1, int(self.diffusion.num_timesteps) - 1 - num_controlled, -1))
        return torch.tensor(timesteps, dtype=torch.long, device=self.device)

    def controlled_timestep_ids(self) -> Tuple[int, ...]:
        return tuple(int(step) for step in self._controlled_timesteps_tensor().detach().cpu().tolist())

    def _generate_batch(
        self,
        entries,
        reverse_noise_policy: nn.Module,
        policy_deterministic: bool = False,
        sampling_seed: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        texts = [entry.caption for entry in entries]
        lengths_20fps = torch.tensor([entry.length_20fps for entry in entries], dtype=torch.long, device=self.device)

        def _sample_once() -> Dict[str, torch.Tensor]:
            ##############################################
            # Sample x_T once, then run reverse diffusion.
            # Only controlled steps use the learned policy noise.
            # All other reverse steps are deterministic mean updates.
            ##############################################
            with torch.no_grad():
                text_embeds = self.actor.encode_text(texts)
                model_kwargs = self._build_model_kwargs(
                    texts=texts,
                    lengths_20fps=lengths_20fps,
                    max_motion_frames=self.max_motion_frames,
                    text_embeds=text_embeds,
                )
                shape = (
                    len(texts),
                    self.actor.njoints,
                    self.actor.nfeats,
                    self.max_motion_frames,
                )
                x_t = torch.randn(shape, device=self.device)
                controlled_timesteps = self._controlled_timesteps_tensor()
                controlled_set = {int(step.item()) for step in controlled_timesteps}

                controlled_x_t_frames = []
                reverse_noise_actions = []
                reverse_logprobs_old = []

                for step in reversed(range(self.diffusion.num_timesteps)):
                    t = torch.full((len(texts),), step, dtype=torch.long, device=self.device)
                    out = self.diffusion.p_mean_variance(
                        self.sampling_actor,
                        x_t,
                        t,
                        clip_denoised=False,
                        model_kwargs=model_kwargs,
                    )
                    if step in controlled_set:
                        x_t_frames = x_t.squeeze(2).permute(0, 2, 1).contiguous()
                        eps_actions, step_logprobs = reverse_noise_policy.sample(
                            x_t_frames=x_t_frames,
                            text_embeds=text_embeds,
                            step_ids=step,
                            deterministic=policy_deterministic,
                        )
                        eps_tensor = eps_actions.permute(0, 2, 1).unsqueeze(2).contiguous()
                        nonzero_mask = (t != 0).float().view(-1, *([1] * (x_t.dim() - 1)))
                        step_std = torch.exp(0.5 * out["log_variance"])
                        x_prev = out["mean"] + nonzero_mask * step_std * eps_tensor

                        controlled_x_t_frames.append(x_t_frames.detach())
                        reverse_noise_actions.append(eps_actions.detach())
                        reverse_logprobs_old.append(step_logprobs.detach())
                    else:
                        x_prev = out["mean"]
                    x_t = x_prev

            final_sample = x_t.detach()
            if controlled_x_t_frames:
                controlled_x_t_frames_t = torch.stack(controlled_x_t_frames, dim=1)
                reverse_noise_actions_t = torch.stack(reverse_noise_actions, dim=1)
                reverse_logprobs_old_t = torch.stack(reverse_logprobs_old, dim=1)
            else:
                empty_shape = (len(texts), 0, self.max_motion_frames, self.actor.njoints * self.actor.nfeats)
                controlled_x_t_frames_t = final_sample.new_empty(empty_shape)
                reverse_noise_actions_t = final_sample.new_empty(empty_shape)
                reverse_logprobs_old_t = final_sample.new_empty((len(texts), 0, self.max_motion_frames))

            return {
                "texts": texts,
                "text_embeds": text_embeds.detach(),
                "lengths_20fps": lengths_20fps,
                "final_sample": final_sample,
                "frame_features": final_sample.squeeze(2).permute(0, 2, 1).contiguous().detach(),
                "controlled_timesteps": controlled_timesteps.detach(),
                "controlled_x_t_frames": controlled_x_t_frames_t,
                "reverse_noise_actions": reverse_noise_actions_t,
                "reverse_logprobs_old": reverse_logprobs_old_t,
            }

        if sampling_seed is None:
            return _sample_once()
        devices = [] if self.device.index is None else [self.device.index]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(sampling_seed))
            return _sample_once()

    def collect_rollout(
        self,
        entries,
        reverse_noise_policy: nn.Module,
        critic: nn.Module,
        policy_deterministic: bool = False,
        sampling_seed: Optional[int] = None,
    ) -> Tuple[DRPPORollout, Dict[str, float]]:
        sampled = self._generate_batch(
            entries=entries,
            reverse_noise_policy=reverse_noise_policy,
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

        reverse_action_mask = reward_batch.frame_exec_mask.unsqueeze(1).expand(
            -1,
            sampled["controlled_timesteps"].shape[0],
            -1,
        ).contiguous()

        rollout = DRPPORollout(
            texts=sampled["texts"],
            text_embeds=sampled["text_embeds"],
            lengths_20fps=sampled["lengths_20fps"],
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
            controlled_timesteps=sampled["controlled_timesteps"],
            controlled_x_t_frames=sampled["controlled_x_t_frames"],
            reverse_noise_actions=sampled["reverse_noise_actions"],
            reverse_logprobs_old=sampled["reverse_logprobs_old"],
            reverse_action_mask=reverse_action_mask,
            reverse_advantages=None,
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
                "controlled_reverse_steps": float(sampled["controlled_timesteps"].shape[0]),
            },
        )
        return rollout, metrics

    def evaluate_entries(
        self,
        entries,
        reverse_noise_policy: nn.Module,
        policy_deterministic: bool = False,
        sampling_seed: Optional[int] = None,
    ) -> Dict[str, float]:
        sampled = self._generate_batch(
            entries=entries,
            reverse_noise_policy=reverse_noise_policy,
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
                "controlled_reverse_steps": float(sampled["controlled_timesteps"].shape[0]),
            },
        )
