import torch
import torch.nn as nn


class FrameCritic(nn.Module):
    def __init__(
        self,
        frame_dim: int = 263,
        text_dim: int = 512,
        index_dim: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        max_frames: int = 196,
        value_target_norm: str = "none",
        popart_beta: float = 5e-4,
        popart_epsilon: float = 1e-5,
    ):
        super().__init__()
        input_dim = int(frame_dim + text_dim + index_dim)
        self.value_target_norm = str(value_target_norm)
        self.popart_enabled = self.value_target_norm == "popart"
        self.popart_beta = float(popart_beta)
        self.popart_epsilon = float(popart_epsilon)
        self.frame_index = nn.Embedding(max_frames, index_dim)
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
        )
        self.value_head = nn.Linear(hidden_dim, 1)
        self.register_buffer("popart_mu", torch.zeros((), dtype=torch.float32))
        self.register_buffer("popart_nu", torch.ones((), dtype=torch.float32))
        self.register_buffer("popart_sigma", torch.ones((), dtype=torch.float32))

    def _build_inputs(self, frame_features: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, _ = frame_features.shape
        frame_ids = torch.arange(num_frames, device=frame_features.device)
        frame_index_emb = self.frame_index(frame_ids).unsqueeze(0).expand(batch_size, -1, -1)
        text = text_embeds.unsqueeze(1).expand(batch_size, num_frames, -1)
        return torch.cat([frame_features, text, frame_index_emb], dim=-1)

    def denormalize_values(self, normalized_values: torch.Tensor) -> torch.Tensor:
        if not self.popart_enabled:
            return normalized_values
        return normalized_values * self.popart_sigma.to(normalized_values.dtype) + self.popart_mu.to(normalized_values.dtype)

    def normalize_targets(self, targets: torch.Tensor) -> torch.Tensor:
        if not self.popart_enabled:
            return targets
        sigma = self.popart_sigma.to(targets.dtype).clamp(min=self.popart_epsilon)
        mu = self.popart_mu.to(targets.dtype)
        return (targets - mu) / sigma

    def update_popart_stats(self, targets: torch.Tensor, mask: torch.Tensor):
        if not self.popart_enabled:
            return {}

        valid_targets = targets[mask]
        if valid_targets.numel() == 0:
            return {
                "popart_mu": float(self.popart_mu.item()),
                "popart_sigma": float(self.popart_sigma.item()),
            }

        old_mu = self.popart_mu.clone()
        old_sigma = self.popart_sigma.clone()

        batch_mean = valid_targets.mean().to(self.popart_mu.dtype)
        batch_sq_mean = valid_targets.square().mean().to(self.popart_nu.dtype)
        beta = self.popart_beta

        new_mu = (1.0 - beta) * self.popart_mu + beta * batch_mean
        new_nu = (1.0 - beta) * self.popart_nu + beta * batch_sq_mean
        variance = torch.clamp(new_nu - new_mu.square(), min=float(self.popart_epsilon ** 2))
        new_sigma = torch.sqrt(variance)

        # Preserve raw value predictions while changing normalized target scale.
        with torch.no_grad():
            scale = (old_sigma / new_sigma).to(self.value_head.weight.dtype)
            self.value_head.weight.mul_(scale)
            self.value_head.bias.mul_(old_sigma.to(self.value_head.bias.dtype))
            self.value_head.bias.add_(old_mu.to(self.value_head.bias.dtype) - new_mu.to(self.value_head.bias.dtype))
            self.value_head.bias.div_(new_sigma.to(self.value_head.bias.dtype))

            self.popart_mu.copy_(new_mu)
            self.popart_nu.copy_(new_nu)
            self.popart_sigma.copy_(new_sigma)

        return {
            "popart_mu": float(new_mu.item()),
            "popart_sigma": float(new_sigma.item()),
        }

    def forward(
        self,
        frame_features: torch.Tensor,
        text_embeds: torch.Tensor,
        normalized: bool = False,
    ) -> torch.Tensor:
        inputs = self._build_inputs(frame_features, text_embeds)
        hidden = self.trunk(inputs)
        normalized_values = self.value_head(hidden).squeeze(-1)
        if normalized and self.popart_enabled:
            return normalized_values
        return self.denormalize_values(normalized_values)
