"""
Deterministic DDIM variant of frame-level DPPO.

Key design choice:
- Rollout uses deterministic DDIM transitions (eta = 0).
- PPO still needs a per-step action density, so we treat each deterministic
  DDIM transition as the mean of a fixed-variance Gaussian surrogate policy.

This removes reverse-step sampling noise from rollout while keeping the
existing PPO update interface intact.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .diffusion_ppo import FramePPODiffusion
from .diffusion_vpg import _frame_entropy, _frame_log_prob


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(np.array(arr))
    res = arr.to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


class FrameDDIMPPODiffusion(FramePPODiffusion):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ddim_eta = 0.0
        self.ddim_logprob_std = float(self.min_logprob_denoising_std)

    def _ddim_step(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        model_kwargs: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ##############################################
        # Deterministic DDIM transition.
        # We also return a fixed std tensor so PPO can score the
        # realized deterministic action under a surrogate Gaussian.
        ##############################################
        out = self.diffusion.p_mean_variance(
            model,
            x_t,
            t,
            clip_denoised=False,
            model_kwargs=model_kwargs,
        )
        eps = self.diffusion._predict_eps_from_xstart(x_t, t, out["pred_xstart"])
        alpha_bar_prev = _extract_into_tensor(self.diffusion.alphas_cumprod_prev, t, x_t.shape)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(torch.clamp(1.0 - alpha_bar_prev, min=0.0)) * eps
        )
        logprob_std = torch.full_like(mean_pred, self.ddim_logprob_std)
        return mean_pred, mean_pred, logprob_std

    def sample(
        self,
        texts: List[str],
        lengths_20fps: torch.Tensor,
        max_motion_frames: int,
        deterministic: bool = False,
        initial_noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del deterministic
        batch_size = len(texts)
        shape = (
            batch_size,
            self.actor.njoints,
            self.actor.nfeats,
            max_motion_frames,
        )
        model_kwargs = self._build_model_kwargs(texts, lengths_20fps, max_motion_frames)
        if initial_noise is None:
            initial_noise_t = torch.randn(shape, device=self.device)
        else:
            initial_noise_t = initial_noise.detach().clone().to(self.device)
            if tuple(initial_noise_t.shape) != shape:
                raise ValueError(
                    f"initial_noise shape mismatch: expected {shape}, got {tuple(initial_noise_t.shape)}"
                )
        x_t = initial_noise_t.clone()

        chain_prev: List[torch.Tensor] = []
        chain_next: List[torch.Tensor] = []
        old_frame_logprobs: List[torch.Tensor] = []
        stored_timesteps: List[torch.Tensor] = []

        with torch.no_grad():
            for step in reversed(range(self.num_timesteps)):
                t = torch.full((batch_size,), step, dtype=torch.long, device=self.device)
                x_prev, mean_pred, logprob_std = self._ddim_step(
                    model=self.hybrid_actor,
                    x_t=x_t,
                    t=t,
                    model_kwargs=model_kwargs,
                )
                if 0 < step <= self.trainable_steps:
                    chain_prev.append(x_t.detach())
                    chain_next.append(x_prev.detach())
                    old_frame_logprobs.append(_frame_log_prob(x_prev, mean_pred, logprob_std).detach())
                    stored_timesteps.append(t.detach())
                x_t = x_prev

        final_sample = x_t.detach()
        text_embeds = model_kwargs["y"]["text_embed"].detach()
        if chain_prev:
            chain_prev_tensor = torch.stack(chain_prev, dim=1)
            chain_next_tensor = torch.stack(chain_next, dim=1)
            timesteps_tensor = torch.stack(stored_timesteps, dim=1)
            frame_logprobs_old_tensor = torch.stack(old_frame_logprobs, dim=1)
        else:
            chain_prev_tensor = final_sample.new_empty((batch_size, 0) + final_sample.shape[1:])
            chain_next_tensor = final_sample.new_empty((batch_size, 0) + final_sample.shape[1:])
            timesteps_tensor = torch.empty((batch_size, 0), dtype=torch.long, device=self.device)
            frame_logprobs_old_tensor = final_sample.new_empty((batch_size, 0, max_motion_frames))

        return {
            "initial_noise": initial_noise_t.detach(),
            "final_sample": final_sample,
            "frame_features": final_sample.squeeze(2).permute(0, 2, 1).contiguous(),
            "text_embeds": text_embeds,
            "chain_prev": chain_prev_tensor,
            "chain_next": chain_next_tensor,
            "timesteps": timesteps_tensor,
            "frame_logprobs_old": frame_logprobs_old_tensor,
            "trainable_step_frac": (
                float(self.trainable_steps) / float(max(1, self.num_timesteps - 1))
            ),
        }

    def evaluate_frame_logprobs(
        self,
        texts: List[str],
        text_embeds: torch.Tensor,
        lengths_20fps: torch.Tensor,
        chain_prev: torch.Tensor,
        chain_next: torch.Tensor,
        timesteps: torch.Tensor,
        use_base_policy: bool = False,
        return_entropy: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        model_kwargs = self._build_model_kwargs(
            texts=texts,
            lengths_20fps=lengths_20fps,
            max_motion_frames=chain_prev.shape[-1] if chain_prev.shape[1] else chain_next.shape[-1],
            text_embeds=text_embeds,
        )
        model = self.actor_base_sampling if use_base_policy else self.actor_ft_sampling

        frame_logprobs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        for step_idx in range(chain_prev.shape[1]):
            x_t = chain_prev[:, step_idx]
            x_prev = chain_next[:, step_idx]
            t = timesteps[:, step_idx]
            _, mean_pred, std = self._ddim_step(
                model=model,
                x_t=x_t,
                t=t,
                model_kwargs=model_kwargs,
            )
            frame_logprobs.append(_frame_log_prob(x_prev, mean_pred, std))
            if return_entropy:
                entropies.append(_frame_entropy(std))

        if frame_logprobs:
            logprobs_tensor = torch.stack(frame_logprobs, dim=1)
        else:
            batch_size = chain_prev.shape[0]
            num_frames = chain_prev.shape[-1] if chain_prev.dim() == 5 else chain_next.shape[-1]
            logprobs_tensor = chain_prev.new_empty((batch_size, 0, num_frames))
        if not return_entropy:
            return logprobs_tensor, None
        if entropies:
            entropy_tensor = torch.stack(entropies, dim=1)
        else:
            batch_size = chain_prev.shape[0]
            num_frames = chain_prev.shape[-1] if chain_prev.dim() == 5 else chain_next.shape[-1]
            entropy_tensor = chain_prev.new_empty((batch_size, 0, num_frames))
        return logprobs_tensor, entropy_tensor
