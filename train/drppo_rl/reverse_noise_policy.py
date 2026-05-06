from typing import Tuple, Union

import torch
import torch.nn as nn
from torch.distributions import Independent, Normal

from model.mdm import PositionalEncoding, TimestepEmbedder


def _orthogonal_init(module: nn.Module, gain: float) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class FrameReverseNoisePolicy(nn.Module):
    """
    Shared frame-wise reverse-noise policy:
        pi(eps_{t,f} | x_t[f], c, sigma_t)
    """

    def __init__(
        self,
        text_dim: int = 512,
        sigma_embed_dim: int = 128,
        hidden_dim: int = 512,
        action_dim: int = 263,
        num_diffusion_steps: int = 1000,
        num_layers: int = 3,
        log_std_init: float = 0.0,
    ):
        super().__init__()
        self.text_dim = int(text_dim)
        self.sigma_embed_dim = int(sigma_embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.num_diffusion_steps = int(num_diffusion_steps)
        self.num_layers = max(1, int(num_layers))

        ##############################################
        # Reuse MDM-style timestep embedding for the
        # reverse denoising step condition sigma_t.
        ##############################################
        self.step_pos_encoder = PositionalEncoding(
            self.sigma_embed_dim,
            dropout=0.0,
            max_len=max(self.num_diffusion_steps + 1, 2),
        )
        self.step_embedder = TimestepEmbedder(self.sigma_embed_dim, self.step_pos_encoder)

        input_dim = self.action_dim + self.text_dim + self.sigma_embed_dim
        layers = []
        current_dim = input_dim
        for _ in range(self.num_layers):
            layers.append(nn.Linear(current_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            current_dim = self.hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(self.hidden_dim, self.action_dim)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), float(log_std_init), dtype=torch.float32))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.trunk:
            if isinstance(module, nn.Linear):
                _orthogonal_init(module, gain=torch.sqrt(torch.tensor(2.0)).item())
        _orthogonal_init(self.mean_head, gain=0.01)

    def _step_features(self, step_ids: torch.Tensor) -> torch.Tensor:
        step_embeds = self.step_embedder(step_ids.long()).squeeze(0)
        if step_embeds.dim() != 2:
            raise ValueError(f"Unexpected sigma embedding shape: {tuple(step_embeds.shape)}")
        return step_embeds

    def _prepare_inputs(
        self,
        x_t_frames: torch.Tensor,
        text_embeds: torch.Tensor,
        step_ids: Union[int, torch.Tensor],
    ) -> Tuple[torch.Tensor, int, int]:
        if x_t_frames.dim() != 3:
            raise ValueError(f"x_t_frames must be [B, F, D], got {tuple(x_t_frames.shape)}")
        batch_size, num_frames, frame_dim = x_t_frames.shape
        if frame_dim != self.action_dim:
            raise ValueError(f"Expected frame dim {self.action_dim}, got {frame_dim}")
        if text_embeds.shape != (batch_size, self.text_dim):
            raise ValueError(
                f"text_embeds must be {(batch_size, self.text_dim)}, got {tuple(text_embeds.shape)}"
            )

        if isinstance(step_ids, int):
            step_tensor = torch.full((batch_size,), int(step_ids), dtype=torch.long, device=x_t_frames.device)
        else:
            step_tensor = step_ids.to(device=x_t_frames.device, dtype=torch.long)
            if step_tensor.dim() == 0:
                step_tensor = step_tensor.expand(batch_size)
            if step_tensor.shape != (batch_size,):
                raise ValueError(f"step_ids must be scalar or [B], got {tuple(step_tensor.shape)}")

        sigma_h = self._step_features(step_tensor).unsqueeze(1).expand(batch_size, num_frames, -1)
        text_h = text_embeds.unsqueeze(1).expand(batch_size, num_frames, -1)
        inputs = torch.cat([x_t_frames, text_h, sigma_h], dim=-1)
        return inputs, batch_size, num_frames

    def forward(
        self,
        x_t_frames: torch.Tensor,
        text_embeds: torch.Tensor,
        step_ids: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        inputs, batch_size, num_frames = self._prepare_inputs(x_t_frames, text_embeds, step_ids)
        hidden = self.trunk(inputs.view(batch_size * num_frames, -1))
        mean = self.mean_head(hidden)
        return mean.view(batch_size, num_frames, self.action_dim)

    def dist(
        self,
        x_t_frames: torch.Tensor,
        text_embeds: torch.Tensor,
        step_ids: Union[int, torch.Tensor],
    ) -> Independent:
        mean = self.forward(x_t_frames, text_embeds, step_ids)
        std = self.log_std.exp().view(1, 1, -1).expand_as(mean)
        return Independent(Normal(mean, std), 1)

    def sample(
        self,
        x_t_frames: torch.Tensor,
        text_embeds: torch.Tensor,
        step_ids: Union[int, torch.Tensor],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.dist(x_t_frames, text_embeds, step_ids)
        actions = dist.mean if deterministic else dist.sample()
        return actions, dist.log_prob(actions)

    def log_prob(
        self,
        actions: torch.Tensor,
        x_t_frames: torch.Tensor,
        text_embeds: torch.Tensor,
        step_ids: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        return self.dist(x_t_frames, text_embeds, step_ids).log_prob(actions)

    def entropy(
        self,
        x_t_frames: torch.Tensor,
        text_embeds: torch.Tensor,
        step_ids: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        return self.dist(x_t_frames, text_embeds, step_ids).entropy()

    def prior_kl(
        self,
        x_t_frames: torch.Tensor,
        text_embeds: torch.Tensor,
        step_ids: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        ##############################################
        # Keep the learned reverse-noise policy near
        # the diffusion prior eps ~ N(0, I).
        ##############################################
        mean = self.forward(x_t_frames, text_embeds, step_ids)
        log_std = self.log_std.view(1, 1, -1)
        var = torch.exp(2.0 * log_std)
        kl_per_dim = 0.5 * (mean.pow(2) + var - 1.0 - 2.0 * log_std)
        return kl_per_dim.sum(dim=-1)
