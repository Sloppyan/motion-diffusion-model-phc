from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Independent, Normal

from model.mdm import PositionalEncoding, TimestepEmbedder


def _orthogonal_init(module: nn.Module, gain: float) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class FrameNoisePolicy(nn.Module):
    """
    Frame-wise Gaussian noise policy:
        pi(w_f | c, f) = N(mu(c, e_f), diag(sigma^2))

    The mean depends on text and frame index; the log-std is a shared
    learnable vector and does not depend on state in the first version.
    """

    def __init__(
        self,
        text_dim: int = 512,
        hidden_dim: int = 256,
        action_dim: int = 263,
        max_frames: int = 196,
        log_std_init: float = 0.0,
    ):
        super().__init__()
        self.text_dim = int(text_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.max_frames = int(max_frames)

        ##############################################
        # Reuse MDM-style frame timestep embedding:
        # PositionalEncoding -> Linear -> SiLU -> Linear.
        ##############################################
        self.frame_pos_encoder = PositionalEncoding(self.hidden_dim, dropout=0.0, max_len=max_frames + 1)
        self.frame_embedder = TimestepEmbedder(self.hidden_dim, self.frame_pos_encoder)

        self.text_proj = nn.Linear(self.text_dim, self.hidden_dim)
        self.trunk = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(self.hidden_dim, self.action_dim)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), float(log_std_init), dtype=torch.float32))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        _orthogonal_init(self.text_proj, gain=torch.sqrt(torch.tensor(2.0)).item())
        for module in self.trunk:
            if isinstance(module, nn.Linear):
                _orthogonal_init(module, gain=torch.sqrt(torch.tensor(2.0)).item())
        _orthogonal_init(self.mean_head, gain=0.01)

    def _frame_features(self, frame_indices: torch.Tensor) -> torch.Tensor:
        frame_embeds = self.frame_embedder(frame_indices.long()).squeeze(0)
        if frame_embeds.dim() != 2:
            raise ValueError(f"Unexpected frame embedding shape: {tuple(frame_embeds.shape)}")
        return frame_embeds

    def forward(self, text_embeds: torch.Tensor, frame_indices: torch.Tensor) -> torch.Tensor:
        batch_size = text_embeds.shape[0]
        num_frames = frame_indices.shape[0]
        text_h = self.text_proj(text_embeds).unsqueeze(1).expand(batch_size, num_frames, -1)
        frame_h = self._frame_features(frame_indices).unsqueeze(0).expand(batch_size, -1, -1)
        hidden = self.trunk(torch.cat([text_h, frame_h], dim=-1))
        return self.mean_head(hidden)

    def dist(self, text_embeds: torch.Tensor, frame_indices: torch.Tensor) -> Independent:
        mean = self.forward(text_embeds, frame_indices)
        std = self.log_std.exp().view(1, 1, -1).expand_as(mean)
        return Independent(Normal(mean, std), 1)

    def sample(
        self,
        text_embeds: torch.Tensor,
        frame_indices: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.dist(text_embeds, frame_indices)
        actions = dist.mean if deterministic else dist.sample()
        return actions, dist.log_prob(actions)

    def log_prob(self, actions: torch.Tensor, text_embeds: torch.Tensor, frame_indices: torch.Tensor) -> torch.Tensor:
        return self.dist(text_embeds, frame_indices).log_prob(actions)

    def entropy(self, text_embeds: torch.Tensor, frame_indices: torch.Tensor) -> torch.Tensor:
        return self.dist(text_embeds, frame_indices).entropy()

    def prior_kl(self, text_embeds: torch.Tensor, frame_indices: torch.Tensor) -> torch.Tensor:
        mean = self.forward(text_embeds, frame_indices)
        log_std = self.log_std.view(1, 1, -1)
        var = torch.exp(2.0 * log_std)
        kl_per_dim = 0.5 * (mean.pow(2) + var - 1.0 - 2.0 * log_std)
        return kl_per_dim.sum(dim=-1)
