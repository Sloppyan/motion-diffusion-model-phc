from typing import Dict

import torch

from train.dppo_frame_rl.logging import masked_mean


def _masked_tensor_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float()
    denom = mask_f.sum().clamp(min=1.0)
    return (values * mask_f).sum() / denom


def compute_noise_policy_inversion_loss(
    noise_policy,
    text_embeds: torch.Tensor,
    frame_noise: torch.Tensor,
    frame_noise_base: torch.Tensor,
    frame_noise_gt_inv: torch.Tensor,
    frame_logprobs_old: torch.Tensor,
    frame_advantages: torch.Tensor,
    frame_exec_mask: torch.Tensor,
    clip_range: float,
    inversion_loss_coef: float,
    noise_prior_kl_coef: float,
) -> Dict[str, torch.Tensor]:
    num_frames = frame_noise.shape[1]
    frame_indices = torch.arange(num_frames, device=frame_noise.device, dtype=torch.long)

    new_logprobs = noise_policy.log_prob(frame_noise, text_embeds, frame_indices)
    logratio = new_logprobs - frame_logprobs_old
    ratio = torch.exp(logratio)

    unclipped = ratio * frame_advantages
    clipped = torch.clamp(ratio, 1.0 - float(clip_range), 1.0 + float(clip_range)) * frame_advantages
    policy_loss = -_masked_tensor_mean(torch.minimum(unclipped, clipped), frame_exec_mask)

    ##############################################
    # Reuse the sampled standard-Gaussian residual from rollout
    # and let the new mean define a differentiable reparameterized
    # noise prediction for inversion supervision.
    ##############################################
    current_mean = noise_policy.forward(text_embeds, frame_indices)
    reparam_noise = current_mean + frame_noise_base
    inversion_error = torch.norm(reparam_noise - frame_noise_gt_inv, p=2, dim=-1)
    inversion_loss = _masked_tensor_mean(inversion_error, frame_exec_mask)
    weighted_inversion_loss = float(inversion_loss_coef) * inversion_loss

    ##############################################
    # For mean-only unit-std policy N(mu, I), the KL
    # to the standard Gaussian prior N(0, I) reduces
    # to 0.5 * ||mu||^2 over the action dimensions.
    ##############################################
    prior_kl = _masked_tensor_mean(0.5 * current_mean.pow(2).mean(dim=-1), frame_exec_mask)
    weighted_prior_kl = float(noise_prior_kl_coef) * prior_kl

    entropy = _masked_tensor_mean(noise_policy.entropy(text_embeds, frame_indices), frame_exec_mask)
    total_loss = policy_loss + weighted_inversion_loss + weighted_prior_kl

    with torch.no_grad():
        approx_kl = torch.as_tensor(
            masked_mean((ratio - 1.0) - logratio, frame_exec_mask),
            device=frame_noise.device,
            dtype=policy_loss.dtype,
        )
        clipfrac = torch.as_tensor(
            masked_mean((torch.abs(ratio - 1.0) > float(clip_range)).float(), frame_exec_mask),
            device=frame_noise.device,
            dtype=policy_loss.dtype,
        )
        ratio_mean = torch.as_tensor(
            masked_mean(ratio, frame_exec_mask),
            device=frame_noise.device,
            dtype=policy_loss.dtype,
        )
        sample_inversion_l2 = _masked_tensor_mean(
            torch.norm(frame_noise - frame_noise_gt_inv, p=2, dim=-1),
            frame_exec_mask,
        )
        weighted_sample_inversion_l2 = float(inversion_loss_coef) * sample_inversion_l2

    return {
        "loss": total_loss,
        "policy_loss": policy_loss,
        "inversion_loss": weighted_inversion_loss,
        "sample_inversion_l2": weighted_sample_inversion_l2,
        "prior_kl": weighted_prior_kl,
        "entropy": entropy,
        "approx_kl": approx_kl,
        "clipfrac": clipfrac,
        "ratio_mean": ratio_mean,
    }
