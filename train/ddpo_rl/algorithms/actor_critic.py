from __future__ import annotations

import math
from typing import Dict

import numpy as np
import torch

from train.ddpo_rl.algorithms.common import ppo_policy_update
from train.ddpo_rl.interfaces import RLAlgorithm
from train.ddpo_rl.models import SequenceValueCritic
from train.ddpo_rl.storage import (
    build_step_update_mask,
    build_chunk_reward_sequence,
    build_terminal_reward_sequence,
    compute_chunk_gae_from_rewards_and_values,
    compute_gae_from_rewards_and_values,
    flatten_hidden_features,
    flatten_hidden_features_with_chunk_mask,
    reshape_step_vector,
    reshape_step_chunk_tensor,
)


class ActorCriticAlgorithm(RLAlgorithm):
    requires_hidden_features = True

    def __init__(
        self,
        model,
        diffusion,
        optimizer,
        trainable_params,
        device,
        args,
        reference_model=None,
    ):
        self.model = model
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.trainable_params = list(trainable_params)
        self.device = device
        self.args = args
        self.reference_model = reference_model
        self.use_chunk_assignment = getattr(args, "reward_assignment", "sequence") == "chunk"
        self.critic = SequenceValueCritic(
            input_dim=int(model.latent_dim),
            hidden_dim=args.critic_hidden_dim,
            dropout=args.critic_dropout,
            output_dim=(max(1, int(math.ceil(args.max_motion_frames / float(args.reward_chunk_size)))) if self.use_chunk_assignment else 1),
        ).to(device)
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=args.critic_lr,
            weight_decay=args.critic_weight_decay,
        )

    def on_rollout_end(self, rollout) -> Dict[str, float]:
        if self.use_chunk_assignment:
            build_chunk_reward_sequence(rollout)
        else:
            build_terminal_reward_sequence(rollout)
        return {}

    def _critic_batch_iterator(self, rollout, value_targets: torch.Tensor, train_mask: torch.Tensor):
        if self.use_chunk_assignment:
            hidden_flat, mask_flat, chunk_mask_flat = flatten_hidden_features_with_chunk_mask(rollout)
            returns_flat = value_targets.reshape(-1, value_targets.shape[-1]).float()
            train_mask_flat = train_mask.reshape(-1, train_mask.shape[-1]).float()
        else:
            hidden_flat, mask_flat = flatten_hidden_features(rollout)
            chunk_mask_flat = None
            returns_flat = value_targets.reshape(-1).float()
            train_mask_flat = train_mask.reshape(-1).float()
        total = hidden_flat.shape[0]
        order = torch.randperm(total)
        for start in range(0, total, self.args.critic_minibatch_size):
            idx = order[start : start + self.args.critic_minibatch_size]
            if train_mask_flat[idx].sum().item() <= 0:
                continue
            batch = (
                hidden_flat[idx].to(self.device),
                mask_flat[idx].to(self.device),
                returns_flat[idx].to(self.device),
                train_mask_flat[idx].to(self.device),
            )
            if chunk_mask_flat is not None:
                batch = batch + (chunk_mask_flat[idx].to(self.device),)
            yield batch

    def _train_critic(self, rollout, value_targets: torch.Tensor, train_mask: torch.Tensor) -> Dict[str, float]:
        self.critic.train()
        losses = []

        ##############################
        # Regress the critic to the bootstrapped value targets built from the
        # pre-update value sequence and terminal-only reward sequence.
        ##############################
        for _ in range(self.args.critic_num_epochs):
            for critic_batch in self._critic_batch_iterator(rollout, value_targets, train_mask):
                if self.use_chunk_assignment:
                    hidden_mb, mask_mb, returns_mb, train_mask_mb, chunk_mask_mb = critic_batch
                else:
                    hidden_mb, mask_mb, returns_mb, train_mask_mb = critic_batch
                    chunk_mask_mb = None
                self.critic_optimizer.zero_grad(set_to_none=True)
                pred = self.critic(hidden_mb, mask_mb)
                if self.use_chunk_assignment:
                    loss_mask = train_mask_mb * chunk_mask_mb.float()
                    loss = 0.5 * (((pred - returns_mb) ** 2) * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
                else:
                    loss = 0.5 * (((pred - returns_mb) ** 2) * train_mask_mb).sum() / train_mask_mb.sum().clamp(min=1.0)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.max_grad_norm)
                self.critic_optimizer.step()
                losses.append(float(loss.detach().cpu().item()))

        return {
            "value_loss": float(np.mean(losses)) if losses else 0.0,
        }

    def _predict_values(self, rollout) -> torch.Tensor:
        self.critic.eval()
        hidden_flat, mask_flat = flatten_hidden_features(rollout)
        values = []
        with torch.no_grad():
            for start in range(0, hidden_flat.shape[0], self.args.critic_minibatch_size):
                end = start + self.args.critic_minibatch_size
                pred = self.critic(
                    hidden_flat[start:end].to(self.device),
                    mask_flat[start:end].to(self.device),
                )
                values.append(pred.cpu())
        values = torch.cat(values, dim=0)
        if self.use_chunk_assignment:
            return reshape_step_chunk_tensor(rollout, values)
        return reshape_step_vector(rollout, values)

    def update(self, rollout) -> Dict[str, float]:
        ##############################################
        # Sequence mode keeps the original scalar value path. Chunk mode lifts
        # the same logic to [diffusion_step, chunk] tensors end-to-end.
        ##############################################
        values_old = self._predict_values(rollout).float()
        step_rewards = rollout.step_rewards.float()
        step_update_mask = build_step_update_mask(rollout, self.args.ft_denoising_steps).float()
        if self.use_chunk_assignment:
            gae_advantages, value_targets, td_deltas = compute_chunk_gae_from_rewards_and_values(
                rewards=step_rewards,
                values=values_old,
                gamma=self.args.gae_gamma,
                gae_lambda=self.args.gae_lambda,
            )
            chunk_exec_mask = rollout.chunk_exec_mask.float()
            valid_mask = step_update_mask.unsqueeze(-1) * chunk_exec_mask.unsqueeze(1).expand_as(gae_advantages)
            adv_mean = (gae_advantages * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)
            adv_var = (((gae_advantages - adv_mean) ** 2) * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)
            adv_std = adv_var.sqrt() + 1e-6
            normalized_advantages = ((gae_advantages - adv_mean) / adv_std) * valid_mask
        else:
            gae_advantages, value_targets, td_deltas = compute_gae_from_rewards_and_values(
                rewards=step_rewards,
                values=values_old,
                gamma=self.args.gae_gamma,
                gae_lambda=self.args.gae_lambda,
            )
            valid_mask = step_update_mask
            adv_mean = (gae_advantages * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)
            adv_var = (((gae_advantages - adv_mean) ** 2) * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)
            adv_std = adv_var.sqrt() + 1e-6
            normalized_advantages = ((gae_advantages - adv_mean) / adv_std) * valid_mask

        critic_metrics = self._train_critic(rollout, value_targets, valid_mask)

        if self.use_chunk_assignment:
            rollout.chunk_values = values_old.cpu()
            rollout.chunk_value_targets = value_targets.cpu()
            rollout.td_deltas = td_deltas.cpu()
            rollout.chunk_advantages = normalized_advantages.cpu()
            valid_mask_cpu = valid_mask.cpu()
            rollout.advantages = (
                (rollout.chunk_advantages * valid_mask_cpu).sum(dim=(1, 2))
                / valid_mask_cpu.sum(dim=(1, 2)).clamp(min=1.0)
            ).cpu()
            step_advantages = rollout.chunk_advantages
            value_pred_mean = float(((values_old * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)).item())
            value_target_mean = float(((value_targets * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)).item())
            gae_adv_mean = float(((gae_advantages * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)).item())
            gae_adv_std = float(torch.sqrt(((((gae_advantages - gae_adv_mean) ** 2) * valid_mask).sum() / valid_mask.sum().clamp(min=1.0))).item())
            td_delta_mean = float(((td_deltas * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)).item())
            adv_mean_metric = float(((rollout.chunk_advantages * valid_mask_cpu).sum() / valid_mask_cpu.sum().clamp(min=1.0)).item())
            adv_std_metric = float(
                torch.sqrt(
                    ((((rollout.chunk_advantages - adv_mean_metric) ** 2) * valid_mask_cpu).sum() / valid_mask_cpu.sum().clamp(min=1.0))
                ).item()
            )
        else:
            rollout.values = values_old.cpu()
            rollout.returns = value_targets.cpu()
            rollout.td_deltas = td_deltas.cpu()
            rollout.step_advantages = normalized_advantages.cpu()
            valid_mask_cpu = valid_mask.cpu()
            rollout.advantages = ((rollout.step_advantages * valid_mask_cpu).sum(dim=1) / valid_mask_cpu.sum(dim=1).clamp(min=1.0)).cpu()
            step_advantages = rollout.step_advantages
            value_pred_mean = float(((values_old * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)).item())
            value_target_mean = float(((value_targets * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)).item())
            gae_adv_mean = float(((gae_advantages * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)).item())
            gae_adv_std = float(torch.sqrt(((((gae_advantages - gae_adv_mean) ** 2) * valid_mask).sum() / valid_mask.sum().clamp(min=1.0))).item())
            td_delta_mean = float(((td_deltas * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)).item())
            adv_mean_metric = float(((rollout.step_advantages * valid_mask_cpu).sum() / valid_mask_cpu.sum().clamp(min=1.0)).item())
            adv_std_metric = float(
                torch.sqrt(
                    ((((rollout.step_advantages - adv_mean_metric) ** 2) * valid_mask_cpu).sum() / valid_mask_cpu.sum().clamp(min=1.0))
                ).item()
            )

        actor_metrics = ppo_policy_update(
            model=self.model,
            diffusion=self.diffusion,
            optimizer=self.optimizer,
            trainable_params=self.trainable_params,
            rollout=rollout,
            step_advantages=step_advantages,
            device=self.device,
            args=self.args,
            reference_model=self.reference_model,
        )
        return {
            "policy_loss": actor_metrics["loss"],
            "approx_kl": actor_metrics["approx_kl"],
            "clipfrac": actor_metrics["clipfrac"],
            "ratio_mean": actor_metrics["ratio_mean"],
            "entropy": actor_metrics["entropy"],
            "reg_loss": actor_metrics["reg_loss"],
            "value_loss": critic_metrics["value_loss"],
            "value_pred_mean": value_pred_mean,
            "value_target_mean": value_target_mean,
            "gae_adv_mean": gae_adv_mean,
            "gae_adv_std": gae_adv_std,
            "td_delta_mean": td_delta_mean,
            "adv_mean": adv_mean_metric,
            "adv_std": adv_std_metric,
            "trainable_step_frac": actor_metrics["trainable_step_frac"],
            "trainable_t_min": actor_metrics["trainable_t_min"],
            "trainable_t_max": actor_metrics["trainable_t_max"],
        }

    def state_dict(self) -> Dict:
        return {
            "critic": self.critic.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }
