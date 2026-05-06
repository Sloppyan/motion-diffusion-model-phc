from typing import Dict

import torch

from train.dppo_frame_rl.logging import masked_mean


def _masked_tensor_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float()
    denom = mask_f.sum().clamp(min=1.0)
    return (values * mask_f).sum() / denom


def compute_reverse_noise_policy_loss(
    reverse_noise_policy,
    text_embeds: torch.Tensor,
    controlled_x_t_frames: torch.Tensor,
    reverse_noise_actions: torch.Tensor,
    reverse_logprobs_old: torch.Tensor,
    reverse_advantages: torch.Tensor,
    reverse_action_mask: torch.Tensor,
    controlled_timesteps: torch.Tensor,
    clip_range: float,
    entropy_coef: float,
    reverse_noise_prior_kl_coef: float,
) -> Dict[str, torch.Tensor]:
    new_logprobs = []
    entropies = []
    prior_kls = []
    for step_idx in range(controlled_x_t_frames.shape[1]):
        step_id = int(controlled_timesteps[step_idx].item())
        step_logprob = reverse_noise_policy.log_prob(
            actions=reverse_noise_actions[:, step_idx],
            x_t_frames=controlled_x_t_frames[:, step_idx],
            text_embeds=text_embeds,
            step_ids=step_id,
        )
        step_entropy = reverse_noise_policy.entropy(
            x_t_frames=controlled_x_t_frames[:, step_idx],
            text_embeds=text_embeds,
            step_ids=step_id,
        )
        step_prior_kl = reverse_noise_policy.prior_kl(
            x_t_frames=controlled_x_t_frames[:, step_idx],
            text_embeds=text_embeds,
            step_ids=step_id,
        )
        new_logprobs.append(step_logprob)
        entropies.append(step_entropy)
        prior_kls.append(step_prior_kl)

    new_logprobs_t = torch.stack(new_logprobs, dim=1)
    entropies_t = torch.stack(entropies, dim=1)
    prior_kls_t = torch.stack(prior_kls, dim=1)

    logratio = new_logprobs_t - reverse_logprobs_old
    ratio = torch.exp(logratio)
    unclipped = ratio * reverse_advantages
    clipped = torch.clamp(ratio, 1.0 - float(clip_range), 1.0 + float(clip_range)) * reverse_advantages
    policy_loss = -_masked_tensor_mean(torch.minimum(unclipped, clipped), reverse_action_mask)

    entropy = _masked_tensor_mean(entropies_t, reverse_action_mask)
    prior_kl = _masked_tensor_mean(prior_kls_t, reverse_action_mask)
    total_loss = policy_loss + float(reverse_noise_prior_kl_coef) * prior_kl - float(entropy_coef) * entropy

    with torch.no_grad():
        approx_kl = torch.as_tensor(
            masked_mean((ratio - 1.0) - logratio, reverse_action_mask),
            device=controlled_x_t_frames.device,
            dtype=policy_loss.dtype,
        )
        clipfrac = torch.as_tensor(
            masked_mean((torch.abs(ratio - 1.0) > float(clip_range)).float(), reverse_action_mask),
            device=controlled_x_t_frames.device,
            dtype=policy_loss.dtype,
        )
        ratio_mean = torch.as_tensor(
            masked_mean(ratio, reverse_action_mask),
            device=controlled_x_t_frames.device,
            dtype=policy_loss.dtype,
        )

    return {
        "loss": total_loss,
        "policy_loss": policy_loss,
        "prior_kl": prior_kl,
        "entropy": entropy,
        "approx_kl": approx_kl,
        "clipfrac": clipfrac,
        "ratio_mean": ratio_mean,
    }
