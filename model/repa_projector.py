import torch
import torch.nn as nn


class LayerWeightedFusion(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_layers))

    def forward(self, hidden_states):
        if len(hidden_states) != self.logits.numel():
            raise ValueError(
                f'expected {self.logits.numel()} hidden states, got {len(hidden_states)}'
            )
        weights = torch.softmax(self.logits, dim=0)
        fused = 0.0
        for idx, hidden in enumerate(hidden_states):
            fused = fused + weights[idx] * hidden
        return fused, weights


class RepaProjector(nn.Module):
    def __init__(self, hidden_dim, projector_hidden_dim=512, output_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, projector_hidden_dim),
            nn.SiLU(),
            nn.Linear(projector_hidden_dim, projector_hidden_dim),
            nn.SiLU(),
            nn.Linear(projector_hidden_dim, output_dim),
        )

    def forward(self, hidden_states):
        return self.net(hidden_states)
