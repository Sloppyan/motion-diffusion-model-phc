from __future__ import annotations

import torch
import torch.nn as nn


class SequenceValueCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, hidden: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        hidden = hidden.float()
        frame_mask = frame_mask.float()
        denom = frame_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (hidden * frame_mask.unsqueeze(-1)).sum(dim=1) / denom
        return self.mlp(pooled).squeeze(-1)
