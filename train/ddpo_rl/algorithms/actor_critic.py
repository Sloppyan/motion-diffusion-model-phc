from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from train.ddpo_rl.algorithms.common import ppo_policy_update
from train.ddpo_rl.interfaces import RLAlgorithm
from train.ddpo_rl.models import SequenceValueCritic
from train.ddpo_rl.storage import (
    build_terminal_reward_sequence,
    compute_gae_from_rewards_and_values,
    flatten_hidden_features,
    reshape_step_vector,
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
        self.critic = SequenceValueCritic(
            input_dim=int(model.latent_dim),
            hidden_dim=args.critic_hidden_dim,
            dropout=args.critic_dropout,
        ).to(device)
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=args.critic_lr,
            weight_decay=args.critic_weight_decay,
        )

    def on_rollout_end(self, rollout) -> Dict[str, float]:
        build_terminal_reward_sequence(rollout)
        return {}

    def _critic_batch_iterator(self, rollout, value_targets: torch.Tensor):
        hidden_flat, mask_flat = flatten_hidden_features(rollout)
        returns_flat = value_targets.reshape(-1).float()
        total = hidden_flat.shape[0]
        order = torch.randperm(total)
        for start in range(0, total, self.args.critic_minibatch_size):
            idx = order[start : start + self.args.critic_minibatch_size]
            yield (
                hidden_flat[idx].to(self.device),
                mask_flat[idx].to(self.device),
                returns_flat[idx].to(self.device),
            )

    def _train_critic(self, rollout, value_targets: torch.Tensor) -> Dict[str, float]:
        self.critic.train()
        losses = []

        ##############################
        # Regress the critic to the bootstrapped value targets built from the
        # pre-update value sequence and terminal-only reward sequence.
        ##############################
        for _ in range(self.args.critic_num_epochs):
            for hidden_mb, mask_mb, returns_mb in self._critic_batch_iterator(rollout, value_targets):
                self.critic_optimizer.zero_grad(set_to_none=True)
                pred = self.critic(hidden_mb, mask_mb)
                loss = 0.5 * ((pred - returns_mb) ** 2).mean()
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
        return reshape_step_vector(rollout, torch.cat(values, dim=0))

    def update(self, rollout) -> Dict[str, float]:
        ##############################
        # First evaluate the current critic on the rollout, then build GAE
        # advantages and bootstrapped value targets from that fixed value
        # sequence. Targets stay frozen while the critic is updated.
        ##############################
        values_old = self._predict_values(rollout).float()
        step_rewards = rollout.step_rewards.float()
        gae_advantages, value_targets, td_deltas = compute_gae_from_rewards_and_values(
            rewards=step_rewards,
            values=values_old,
            gamma=self.args.gae_gamma,
            gae_lambda=self.args.gae_lambda,
        )

        critic_metrics = self._train_critic(rollout, value_targets)

        adv_mean = gae_advantages.mean()
        adv_std = gae_advantages.std(unbiased=False) + 1e-6
        normalized_advantages = (gae_advantages - adv_mean) / adv_std

        rollout.values = values_old.cpu()
        rollout.returns = value_targets.cpu()
        rollout.td_deltas = td_deltas.cpu()
        rollout.step_advantages = normalized_advantages.cpu()
        rollout.advantages = rollout.step_advantages.mean(dim=1).cpu()

        actor_metrics = ppo_policy_update(
            model=self.model,
            diffusion=self.diffusion,
            optimizer=self.optimizer,
            trainable_params=self.trainable_params,
            rollout=rollout,
            step_advantages=rollout.step_advantages,
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
            "value_pred_mean": float(values_old.mean().item()),
            "value_target_mean": float(value_targets.mean().item()),
            "gae_adv_mean": float(gae_advantages.mean().item()),
            "gae_adv_std": float(gae_advantages.std(unbiased=False).item()),
            "td_delta_mean": float(td_deltas.mean().item()),
            "adv_mean": float(rollout.step_advantages.mean().item()),
            "adv_std": float(rollout.step_advantages.std(unbiased=False).item()),
        }

    def state_dict(self) -> Dict:
        return {
            "critic": self.critic.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }
