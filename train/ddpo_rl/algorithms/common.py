from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch

from model.cfg_sampler import ClassifierFreeSampleModel

_LOG_TWO_PI = float(np.log(2.0 * np.pi))


def build_sampling_model(model, guidance_param: float):
    if guidance_param == 1.0:
        return model
    return ClassifierFreeSampleModel(model)


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

    for _ in range(args.actor_num_epochs):
        order = torch.randperm(batch_size)
        for start in range(0, batch_size, args.ppo_minibatch_size):
            batch_idx = order[start : start + args.ppo_minibatch_size]
            base_y = _build_base_y(rollout, batch_idx, text_embed_full, device, args.guidance_param)

            optimizer.zero_grad(set_to_none=True)
            minibatch_has_update = False

            # Iterate over diffusion steps inside one optimizer step so the
            # policy sees the full episode-level supervision before updating.
            for step_idx in range(rollout.num_steps):
                timesteps = rollout.timesteps[batch_idx, step_idx].to(device)
                # The last reverse step is deterministic (x_{-1} := mean at t=0),
                # so exclude it from PPO likelihood-ratio updates.
                active_mask = timesteps != 0
                if not active_mask.any():
                    continue

                step_y = _slice_base_y(base_y, active_mask)
                mb_advantages = step_advantages[batch_idx, step_idx].to(device)[active_mask]
                mb_advantages = torch.clamp(mb_advantages, -args.adv_clip_max, args.adv_clip_max)
                x_t = rollout.x_t[batch_idx, step_idx].to(device)[active_mask]
                x_prev = rollout.x_prev[batch_idx, step_idx].to(device)[active_mask]
                timesteps = timesteps[active_mask]
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

                reg_loss = torch.zeros((), device=device)
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

                total_loss = policy_loss + args.kl_coef * reg_loss
                total_loss.backward()
                minibatch_has_update = True

                entropy = 0.5 * (
                    1.0 + _LOG_TWO_PI + out["log_variance"]
                )
                metrics["loss"].append(float(policy_loss.detach().cpu().item()))
                metrics["approx_kl"].append(float((0.5 * ((logp - old_logp) ** 2).mean()).detach().cpu().item()))
                metrics["clipfrac"].append(float((torch.abs(ratio - 1.0) > args.clip_range).float().mean().detach().cpu().item()))
                metrics["ratio_mean"].append(float(ratio.mean().detach().cpu().item()))
                metrics["entropy"].append(float(entropy.sum(dim=tuple(range(1, entropy.ndim))).mean().detach().cpu().item()))
                metrics["reg_loss"].append(float(reg_loss.detach().cpu().item()))

            if not minibatch_has_update:
                optimizer.zero_grad(set_to_none=True)
                continue
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    return {key: float(np.mean(value)) if value else 0.0 for key, value in metrics.items()}
