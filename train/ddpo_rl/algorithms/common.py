from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch

from model.cfg_sampler import ClassifierFreeSampleModel
from train.ddpo_rl.policy_wrappers import HybridDenoisingPolicy

_LOG_TWO_PI = float(np.log(2.0 * np.pi))


def build_sampling_model(model, guidance_param: float):
    if guidance_param == 1.0:
        return model
    return ClassifierFreeSampleModel(model)


def build_rollout_sampling_model(model, reference_model, guidance_param: float, ft_denoising_steps: int):
    sampling_model = build_sampling_model(model, guidance_param)
    if int(ft_denoising_steps) <= 0:
        return sampling_model
    if reference_model is None:
        raise ValueError("ft_denoising_steps requires a frozen reference model for rollout sampling.")
    reference_sampling_model = build_sampling_model(reference_model, guidance_param)
    return HybridDenoisingPolicy(
        trainable_model=sampling_model,
        frozen_model=reference_sampling_model,
        ft_denoising_steps=int(ft_denoising_steps),
    )


def _build_base_y(rollout, batch_idx, text_embed_full, device, guidance_param: float):
    base_y = {
        "text": [rollout.texts[int(i)] for i in batch_idx.tolist()],
        "lengths": rollout.lengths[batch_idx].to(device),
        "mask": rollout.mask[batch_idx].to(device),
        "text_embed": text_embed_full[batch_idx],
    }
    if guidance_param != 1.0:
        base_y["scale"] = rollout.scales[batch_idx].to(device)
    return base_y


def _slice_base_y(base_y, active_mask):
    active_list = active_mask.detach().cpu().tolist()
    sliced = {}
    for key, value in base_y.items():
        if torch.is_tensor(value):
            sliced[key] = value[active_mask]
        elif isinstance(value, list):
            sliced[key] = [item for item, keep in zip(value, active_list) if keep]
        else:
            sliced[key] = value
    return sliced


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    denom = mask.sum().clamp(min=1.0)
    return (values * mask).sum() / denom


def ppo_policy_update(
    model,
    diffusion,
    optimizer,
    trainable_params: Sequence[torch.nn.Parameter],
    rollout,
    step_advantages: torch.Tensor,
    device,
    args,
    reference_model=None,
    denoised_fn=None,
    cond_fn=None,
) -> Dict[str, float]:
    model.eval()
    sampling_model = build_sampling_model(model, args.guidance_param)
    reference_sampling_model = None
    if reference_model is not None and args.kl_coef > 0.0:
        reference_model.eval()
        reference_sampling_model = build_sampling_model(reference_model, args.guidance_param)

    if getattr(rollout, "text_embeds", None) is not None:
        text_embed_full = rollout.text_embeds.to(device)
    else:
        with torch.no_grad():
            text_embed_full = model.encode_text(rollout.texts).detach()

    metrics = {
        "loss": [],
        "approx_kl": [],
        "clipfrac": [],
        "ratio_mean": [],
        "entropy": [],
        "reg_loss": [],
    }
    batch_size = rollout.batch_size
    step_advantages = step_advantages.cpu().float()
    use_chunk_assignment = getattr(args, "reward_assignment", "sequence") == "chunk"
    ft_denoising_steps = int(getattr(args, "ft_denoising_steps", 0))
    trainable_t_min = []
    trainable_t_max = []
    trainable_step_count = 0
    trainable_step_total = 0

    for _ in range(args.actor_num_epochs):
        order = torch.randperm(batch_size)
        for start in range(0, batch_size, args.ppo_minibatch_size):
            batch_idx = order[start : start + args.ppo_minibatch_size]
            base_y = _build_base_y(rollout, batch_idx, text_embed_full, device, args.guidance_param)

            optimizer.zero_grad(set_to_none=True)
            minibatch_has_update = False

            ##############################################
            # Iterate over diffusion steps inside one optimizer step so the
            # policy sees the full rollout supervision before updating.
            ##############################################
            for step_idx in range(rollout.num_steps):
                timesteps = rollout.timesteps[batch_idx, step_idx].to(device)
                # The last reverse step is deterministic (x_{-1} := mean at t=0),
                # and optional suffix fine-tuning only keeps the last K
                # stochastic denoising steps trainable.
                active_mask = timesteps != 0
                if ft_denoising_steps > 0:
                    active_mask = active_mask & (timesteps <= ft_denoising_steps)
                trainable_step_total += int(timesteps.numel())
                trainable_step_count += int(active_mask.sum().item())
                if not active_mask.any():
                    continue
                trainable_t_min.append(int(timesteps[active_mask].min().item()))
                trainable_t_max.append(int(timesteps[active_mask].max().item()))

                step_y = _slice_base_y(base_y, active_mask)
                mb_advantages = step_advantages[batch_idx, step_idx].to(device)[active_mask]
                mb_advantages = torch.clamp(mb_advantages, -args.adv_clip_max, args.adv_clip_max)
                x_t = rollout.x_t[batch_idx, step_idx].to(device)[active_mask]
                x_prev = rollout.x_prev[batch_idx, step_idx].to(device)[active_mask]
                timesteps = timesteps[active_mask]
                reg_loss = torch.zeros((), device=device)

                ##############################################
                # Sequence assignment keeps the original block-level PPO loss.
                # Chunk assignment reuses the same samples but aggregates
                # logprob, ratio, GAE, and critic targets on chunk masks.
                ##############################################
                if use_chunk_assignment:
                    if rollout.chunk_logp_old is None or rollout.chunk_frame_mask is None or rollout.chunk_exec_mask is None:
                        raise ValueError("Chunk assignment requires chunk_logp_old, chunk_frame_mask, and chunk_exec_mask in rollout.")
                    mb_advantages = mb_advantages
                    chunk_frame_mask = rollout.chunk_frame_mask[batch_idx].to(device)[active_mask]
                    valid_chunk_mask = rollout.chunk_exec_mask[batch_idx].to(device)[active_mask]
                    old_logp = rollout.chunk_logp_old[batch_idx, step_idx].to(device)[active_mask]
                    out = diffusion.calc_chunk_action_logprob(
                        sampling_model,
                        x_t,
                        x_prev,
                        timesteps,
                        chunk_frame_mask=chunk_frame_mask,
                        clip_denoised=False,
                        denoised_fn=denoised_fn,
                        cond_fn=cond_fn,
                        model_kwargs={"y": step_y},
                    )
                    logp = out["chunk_logp"]
                    ratio = torch.exp(logp - old_logp)
                    clipped_ratio = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range)
                    unclipped = -mb_advantages * ratio
                    clipped = -mb_advantages * clipped_ratio
                    policy_loss = _masked_mean(torch.maximum(unclipped, clipped), valid_chunk_mask)

                    if reference_sampling_model is not None:
                        with torch.no_grad():
                            ref_out = diffusion.calc_chunk_action_logprob(
                                reference_sampling_model,
                                x_t,
                                x_prev,
                                timesteps,
                                chunk_frame_mask=chunk_frame_mask,
                                clip_denoised=False,
                                denoised_fn=denoised_fn,
                                cond_fn=cond_fn,
                                model_kwargs={"y": step_y},
                            )
                        reg_loss = 0.5 * _masked_mean((logp - ref_out["chunk_logp"]) ** 2, valid_chunk_mask)

                    entropy = 0.5 * (1.0 + _LOG_TWO_PI + out["log_variance"])
                    entropy_frame = entropy.sum(dim=1).sum(dim=1)
                    entropy_chunk = (entropy_frame.unsqueeze(1) * chunk_frame_mask.float()).sum(dim=-1)

                    metrics["loss"].append(float(policy_loss.detach().cpu().item()))
                    metrics["approx_kl"].append(float((0.5 * _masked_mean((logp - old_logp) ** 2, valid_chunk_mask)).detach().cpu().item()))
                    metrics["clipfrac"].append(
                        float(_masked_mean((torch.abs(ratio - 1.0) > args.clip_range).float(), valid_chunk_mask).detach().cpu().item())
                    )
                    metrics["ratio_mean"].append(float(_masked_mean(ratio, valid_chunk_mask).detach().cpu().item()))
                    metrics["entropy"].append(float(_masked_mean(entropy_chunk, valid_chunk_mask).detach().cpu().item()))
                else:
                    old_logp = rollout.logp_old[batch_idx, step_idx].to(device)[active_mask]
                    out = diffusion.calc_action_logprob(
                        sampling_model,
                        x_t,
                        x_prev,
                        timesteps,
                        clip_denoised=False,
                        denoised_fn=denoised_fn,
                        cond_fn=cond_fn,
                        model_kwargs={"y": step_y},
                    )
                    logp = out["logp"]
                    ratio = torch.exp(logp - old_logp)
                    unclipped = -mb_advantages * ratio
                    clipped = -mb_advantages * torch.clamp(
                        ratio,
                        1.0 - args.clip_range,
                        1.0 + args.clip_range,
                    )
                    policy_loss = torch.maximum(unclipped, clipped).mean()

                    if reference_sampling_model is not None:
                        with torch.no_grad():
                            ref_out = diffusion.calc_action_logprob(
                                reference_sampling_model,
                                x_t,
                                x_prev,
                                timesteps,
                                clip_denoised=False,
                                denoised_fn=denoised_fn,
                                cond_fn=cond_fn,
                                model_kwargs={"y": step_y},
                            )
                        reg_loss = 0.5 * ((logp - ref_out["logp"]) ** 2).mean()

                    entropy = 0.5 * (
                        1.0 + _LOG_TWO_PI + out["log_variance"]
                    )

                    metrics["loss"].append(float(policy_loss.detach().cpu().item()))
                    metrics["approx_kl"].append(float((0.5 * ((logp - old_logp) ** 2).mean()).detach().cpu().item()))
                    metrics["clipfrac"].append(float((torch.abs(ratio - 1.0) > args.clip_range).float().mean().detach().cpu().item()))
                    metrics["ratio_mean"].append(float(ratio.mean().detach().cpu().item()))
                    metrics["entropy"].append(float(entropy.sum(dim=tuple(range(1, entropy.ndim))).mean().detach().cpu().item()))

                total_loss = policy_loss + args.kl_coef * reg_loss
                total_loss.backward()
                minibatch_has_update = True
                metrics["reg_loss"].append(float(reg_loss.detach().cpu().item()))

            if not minibatch_has_update:
                optimizer.zero_grad(set_to_none=True)
                continue
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    summary = {key: float(np.mean(value)) if value else 0.0 for key, value in metrics.items()}
    summary["trainable_step_frac"] = float(trainable_step_count / max(1, trainable_step_total))
    summary["trainable_t_min"] = float(np.min(trainable_t_min)) if trainable_t_min else 0.0
    summary["trainable_t_max"] = float(np.max(trainable_t_max)) if trainable_t_max else 0.0
    return summary
