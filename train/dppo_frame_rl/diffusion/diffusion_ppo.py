import math
from typing import Dict, Optional

import torch

from .diffusion_vpg import FrameVPGDiffusion


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float()
    denom = mask_f.sum().clamp(min=1.0)
    return (values * mask_f).sum() / denom


class FramePPODiffusion(FrameVPGDiffusion):
    def __init__(
        self,
        gamma_denoising: float,
        clip_ploss_coef: float,
        clip_ploss_schedule: str = "constant",
        clip_ploss_coef_base: Optional[float] = None,
        clip_ploss_coef_rate: float = 3.0,
        kl_coef: float = 0.0,
        norm_adv: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gamma_denoising = float(gamma_denoising)
        self.clip_ploss_coef = float(clip_ploss_coef)
        self.clip_ploss_schedule = str(clip_ploss_schedule)
        self.clip_ploss_coef_base = (
            float(clip_ploss_coef) if clip_ploss_coef_base is None else float(clip_ploss_coef_base)
        )
        self.clip_ploss_coef_rate = float(clip_ploss_coef_rate)
        self.kl_coef = float(kl_coef)
        self.norm_adv = bool(norm_adv)

        if self.clip_ploss_schedule not in {"constant", "exponential"}:
            raise ValueError(f"Unsupported clip_ploss_schedule: {self.clip_ploss_schedule}")
        if self.clip_ploss_coef_base <= 0.0 or self.clip_ploss_coef <= 0.0:
            raise ValueError("clip ranges must be positive")
        if self.clip_ploss_schedule == "exponential" and self.clip_ploss_coef_base > self.clip_ploss_coef:
            raise ValueError("clip_ploss_coef_base must be <= clip_ploss_coef for exponential schedule")

    def denoising_discounts(self, num_steps: int, device: torch.device) -> torch.Tensor:
        return torch.tensor(
            [self.gamma_denoising ** (num_steps - idx - 1) for idx in range(num_steps)],
            dtype=torch.float32,
            device=device,
        )

    def denoising_clip_ranges(self, num_steps: int, device: torch.device) -> torch.Tensor:
        if num_steps <= 0:
            return torch.empty((0,), dtype=torch.float32, device=device)
        if (
            self.clip_ploss_schedule == "constant"
            or num_steps == 1
            or abs(self.clip_ploss_coef - self.clip_ploss_coef_base) < 1e-12
        ):
            return torch.full((num_steps,), self.clip_ploss_coef, dtype=torch.float32, device=device)

        ##############################################
        # Match DPPO's denoising-step-dependent clipping schedule:
        # earlier trainable denoising steps get a larger clip range,
        # while later steps use a tighter clip range.
        ##############################################
        progress = torch.linspace(1.0, 0.0, num_steps, dtype=torch.float32, device=device)
        if abs(self.clip_ploss_coef_rate) < 1e-12:
            interp = progress
        else:
            denom = math.exp(self.clip_ploss_coef_rate) - 1.0
            interp = (torch.exp(self.clip_ploss_coef_rate * progress) - 1.0) / denom
        return self.clip_ploss_coef_base + (self.clip_ploss_coef - self.clip_ploss_coef_base) * interp

    def loss(
        self,
        texts,
        text_embeds: torch.Tensor,
        lengths_20fps: torch.Tensor,
        chain_prev: torch.Tensor,
        chain_next: torch.Tensor,
        timesteps: torch.Tensor,
        frame_advantages: torch.Tensor,
        frame_exec_mask: torch.Tensor,
        frame_logprobs_old: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        new_logprobs, entropies = self.evaluate_frame_logprobs(
            texts=texts,
            text_embeds=text_embeds,
            lengths_20fps=lengths_20fps,
            chain_prev=chain_prev,
            chain_next=chain_next,
            timesteps=timesteps,
            use_base_policy=False,
            return_entropy=True,
        )
        if new_logprobs.shape[1] == 0:
            zero = chain_prev.sum() * 0.0
            return {
                "loss": zero,
                "policy_loss": zero,
                "reg_loss": zero,
                "entropy": zero,
                "approx_kl": zero,
                "clipfrac": zero,
                "ratio_mean": zero,
            }

        advantages = frame_advantages
        if self.norm_adv:
            valid_adv = advantages[frame_exec_mask]
            if valid_adv.numel() > 1:
                advantages = (advantages - valid_adv.mean()) / (valid_adv.std(unbiased=False) + 1e-8)

        discounts = self.denoising_discounts(new_logprobs.shape[1], new_logprobs.device).view(1, -1, 1)
        weighted_advantages = advantages.unsqueeze(1) * discounts
        active_mask = frame_exec_mask.unsqueeze(1).expand_as(new_logprobs)
        clip_ranges = self.denoising_clip_ranges(new_logprobs.shape[1], new_logprobs.device).view(1, -1, 1)

        logratio = new_logprobs - frame_logprobs_old
        ratio = torch.exp(logratio)

        pg_loss1 = -weighted_advantages * ratio
        pg_loss2 = -weighted_advantages * torch.clamp(
            ratio,
            1.0 - clip_ranges,
            1.0 + clip_ranges,
        )
        policy_loss = _masked_mean(torch.maximum(pg_loss1, pg_loss2), active_mask)

        reg_loss = policy_loss.new_zeros(())
        if self.kl_coef > 0.0:
            base_logprobs, _ = self.evaluate_frame_logprobs(
                texts=texts,
                text_embeds=text_embeds,
                lengths_20fps=lengths_20fps,
                chain_prev=chain_prev,
                chain_next=chain_next,
                timesteps=timesteps,
                use_base_policy=True,
                return_entropy=False,
            )
            reg_loss = _masked_mean(0.5 * (new_logprobs - base_logprobs).pow(2), active_mask)

        total_loss = policy_loss + self.kl_coef * reg_loss

        with torch.no_grad():
            entropy = _masked_mean(entropies, active_mask)
            approx_kl = _masked_mean((ratio - 1.0) - logratio, active_mask)
            clipfrac = _masked_mean(((ratio - 1.0).abs() > clip_ranges).float(), active_mask)
            ratio_mean = _masked_mean(ratio, active_mask)

        metrics = {
            "loss": total_loss,
            "policy_loss": policy_loss,
            "reg_loss": reg_loss,
            "entropy": entropy,
            "approx_kl": approx_kl,
            "clipfrac": clipfrac,
            "ratio_mean": ratio_mean,
        }

        ##############################################
        # Log step-wise clipping diagnostics so clip fraction can be tuned
        # to the DPPO target band (roughly 10%-20%) per denoising step.
        ##############################################
        with torch.no_grad():
            step_ids = timesteps[0].tolist() if timesteps.shape[0] > 0 else []
            for step_idx, timestep in enumerate(step_ids):
                step_mask = frame_exec_mask
                step_clip = clip_ranges[:, step_idx].expand_as(ratio[:, step_idx])
                step_clipfrac = _masked_mean(((ratio[:, step_idx] - 1.0).abs() > step_clip).float(), step_mask)
                metrics[f"clipfrac_t{int(timestep)}"] = step_clipfrac
                metrics[f"clip_range_t{int(timestep)}"] = clip_ranges[0, step_idx, 0]

        return metrics
