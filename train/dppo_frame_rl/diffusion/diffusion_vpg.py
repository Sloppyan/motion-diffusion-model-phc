"""
Frame-level VPG utilities for MDM, following the DPPO split:

- outer value / GAE live on the frame axis
- denoising steps only provide logprobs
- only the denoising suffix is fine-tuned when ft_denoising_steps > 0
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from data_loaders.tensors import lengths_to_mask
from model.lora_attention import (
    add_lora_bank_to_mdm,
    inject_lora_into_mdm,
    iter_lora_parameters,
    load_lora_state_dict,
    mark_only_lora_trainable,
)

from .policy_wrappers import (
    HybridSuffixPolicy,
    build_sampling_actor,
    resolve_trainable_step_count,
)


def _frame_log_prob(sample: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    log_prob = Normal(mean, std).log_prob(sample)
    return log_prob.sum(dim=(1, 2))


def _frame_entropy(std: torch.Tensor) -> torch.Tensor:
    variance = std.pow(2)
    entropy = 0.5 * (1.0 + torch.log(2.0 * torch.tensor(np.pi, device=std.device)) + torch.log(variance))
    return entropy.sum(dim=(1, 2))


class FrameVPGDiffusion:
    def __init__(
        self,
        actor: nn.Module,
        diffusion,
        guidance_param: float = 2.5,
        ft_denoising_steps: int = 0,
        min_sampling_denoising_std: float = 1e-2,
        min_logprob_denoising_std: float = 1e-2,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_layer_scope: str = "all",
        resume_lora_path: str = "",
    ):
        self.diffusion = diffusion
        self.device = next(actor.parameters()).device
        self.guidance_param = float(guidance_param)
        self.min_sampling_denoising_std = float(min_sampling_denoising_std)
        self.min_logprob_denoising_std = float(min_logprob_denoising_std)

        ##############################################
        # Keep one shared MDM backbone and attach two LoRA banks:
        # - base: frozen reference policy
        # - ft:   trainable policy bank
        ##############################################
        inject_lora_into_mdm(
            actor,
            rank=lora_rank,
            alpha=lora_alpha,
            layer_scope=lora_layer_scope,
            bank_name="base",
        )
        if resume_lora_path:
            payload = torch.load(resume_lora_path, map_location="cpu")
            if isinstance(payload, dict):
                state_dict = payload.get("lora") or payload.get("actor_lora") or payload
            else:
                state_dict = payload
            load_lora_state_dict(actor, state_dict, bank_name="base")

        add_lora_bank_to_mdm(actor, bank_name="ft", init_from_bank="base")

        self.actor = actor
        self.actor_ft = actor
        self.actor.eval()
        self.num_trainable_lora_params = mark_only_lora_trainable(self.actor, bank_name="ft")

        self.num_timesteps = int(self.diffusion.num_timesteps)
        self.trainable_steps = resolve_trainable_step_count(self.num_timesteps, ft_denoising_steps)

        self.actor_base_sampling = build_sampling_actor(self.actor, self.guidance_param, bank_name="base").to(self.device)
        self.actor_ft_sampling = build_sampling_actor(self.actor, self.guidance_param, bank_name="ft").to(self.device)
        self.hybrid_actor = HybridSuffixPolicy(
            base_model=self.actor_base_sampling,
            ft_model=self.actor_ft_sampling,
            trainable_steps=self.trainable_steps,
        ).to(self.device)
        self.hybrid_actor.eval()

    def trainable_parameters(self):
        yield from iter_lora_parameters(self.actor, bank_name="ft")

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
            text_embeds = self.actor_ft_sampling.encode_text(texts)
        # CFG sampling deep-copies y; both freshly encoded tensors and sliced
        # cached embeddings should be re-materialized as leaf tensors first.
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

    def _sample_step(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        model_kwargs: Dict,
        deterministic: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.diffusion.p_mean_variance(
            model,
            x_t,
            t,
            clip_denoised=False,
            model_kwargs=model_kwargs,
        )
        raw_std = torch.exp(0.5 * out["log_variance"])
        sample_std = raw_std.clamp(min=self.min_sampling_denoising_std)
        logprob_std = raw_std.clamp(min=self.min_logprob_denoising_std)

        noise = torch.zeros_like(x_t) if deterministic else torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (x_t.dim() - 1)))
        x_prev = out["mean"] + nonzero_mask * sample_std * noise
        return x_prev, out["mean"], logprob_std

    def sample(
        self,
        texts: List[str],
        lengths_20fps: torch.Tensor,
        max_motion_frames: int,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        batch_size = len(texts)
        shape = (
            batch_size,
            self.actor.njoints,
            self.actor.nfeats,
            max_motion_frames,
        )
        model_kwargs = self._build_model_kwargs(texts, lengths_20fps, max_motion_frames)
        x_t = torch.randn(shape, device=self.device)

        chain_prev: List[torch.Tensor] = []
        chain_next: List[torch.Tensor] = []
        old_frame_logprobs: List[torch.Tensor] = []
        stored_timesteps: List[torch.Tensor] = []

        with torch.no_grad():
            for step in reversed(range(self.num_timesteps)):
                t = torch.full((batch_size,), step, dtype=torch.long, device=self.device)
                x_prev, mean, logprob_std = self._sample_step(
                    model=self.hybrid_actor,
                    x_t=x_t,
                    t=t,
                    model_kwargs=model_kwargs,
                    deterministic=deterministic,
                )
                if 0 < step <= self.trainable_steps:
                    chain_prev.append(x_t.detach())
                    chain_next.append(x_prev.detach())
                    old_frame_logprobs.append(_frame_log_prob(x_prev, mean, logprob_std).detach())
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
            out = self.diffusion.p_mean_variance(
                model,
                x_t,
                t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )
            std = torch.exp(0.5 * out["log_variance"]).clamp(min=self.min_logprob_denoising_std)
            frame_logprobs.append(_frame_log_prob(x_prev, out["mean"], std))
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
