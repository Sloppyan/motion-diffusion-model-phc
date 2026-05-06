import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from train.dppo_frame_rl.diffusion.diffusion_ddim_ppo import FrameDDIMPPODiffusion
from train.dppo_frame_rl.logging import merge_metrics, summarize_episodes
from train.dppo_frame_rl.phc_bridge import PHCExternalEvalBridge
from train.dppo_frame_rl.reward import build_frame_reward_batch
from train.dppo_frame_rl.runtime import FrameDPPORuntime
from train.dppo_frame_rl.storage import FrameRollout, compute_frame_returns_and_advantages
from train.dppo_frame_rl.data import PromptEntry, prompt_entries_to_meta
from utils.model_util import create_model_and_diffusion, load_model_wo_clip


class FrameDDIMDPPORuntime(FrameDPPORuntime):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.device_id}")
        self.max_motion_frames = int(args.max_motion_frames)

        actor, diffusion = self._build_actor_and_diffusion(args.model_path)
        self.policy = FrameDDIMPPODiffusion(
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

    def _evaluate_reference_batch(
        self,
        ref_motion_batch: np.ndarray,
        lengths_30hz: Sequence[int],
        entries: Sequence[PromptEntry],
    ):
        ##############################################
        # Match the dynamic PHC rollout budget used by DSPPO
        # so single-env smoke tests do not stop early.
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

    def collect_rollout(
        self,
        entries: Sequence[PromptEntry],
        critic,
        deterministic: bool = True,
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
                "pose_mse_mean": reward_batch.pose_mse_mean,
                "imitation_velocity_reward_mean": reward_batch.imitation_velocity_reward_mean,
                "terminal_reward_mean": reward_batch.terminal_reward_mean,
                "undiscounted_sequence_reward_mean": reward_batch.undiscounted_sequence_reward_mean,
                "exec_ratio_mean": reward_batch.exec_ratio_mean,
            },
        )
        return rollout, metrics
