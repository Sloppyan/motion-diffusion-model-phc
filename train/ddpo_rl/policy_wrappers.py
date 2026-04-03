from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn


def _slice_condition_dict(y: Dict[str, Any], mask: torch.Tensor) -> Dict[str, Any]:
    active_list = mask.detach().cpu().tolist()
    sliced = {}
    for key, value in y.items():
        if torch.is_tensor(value):
            sliced[key] = value[mask]
        elif isinstance(value, list):
            sliced[key] = [item for item, keep in zip(value, active_list) if keep]
        else:
            sliced[key] = value
    return sliced


class HybridDenoisingPolicy(nn.Module):
    def __init__(self, trainable_model: nn.Module, frozen_model: nn.Module, ft_denoising_steps: int):
        super().__init__()
        self.trainable_model = trainable_model
        self.frozen_model = frozen_model
        self.ft_denoising_steps = int(ft_denoising_steps)

        # Mirror the attributes used by diffusion sampling and postprocessing.
        self.model = getattr(self.trainable_model, "model", self.trainable_model)
        self.rot2xyz = self.trainable_model.rot2xyz
        self.translation = self.trainable_model.translation
        self.njoints = self.trainable_model.njoints
        self.nfeats = self.trainable_model.nfeats
        self.data_rep = self.trainable_model.data_rep
        self.cond_mode = self.trainable_model.cond_mode
        self.encode_text = self.trainable_model.encode_text

    def _use_trainable_mask(self, timesteps: torch.Tensor) -> torch.Tensor:
        if self.ft_denoising_steps <= 0:
            return torch.ones_like(timesteps, dtype=torch.bool)
        return timesteps <= int(self.ft_denoising_steps)

    def _dispatch(self, x: torch.Tensor, timesteps: torch.Tensor, y: Dict[str, Any], fn_name: str):
        use_trainable = self._use_trainable_mask(timesteps)
        if use_trainable.all():
            return getattr(self.trainable_model, fn_name)(x, timesteps, y)
        if (~use_trainable).all():
            return getattr(self.frozen_model, fn_name)(x, timesteps, y)

        output = torch.empty_like(x)
        hidden = None

        ##############################
        # Support mixed timestep batches by routing the trainable suffix and
        # frozen prefix through their respective models, then stitching the
        # results back into the original batch order.
        ##############################
        if use_trainable.any():
            trainable_y = _slice_condition_dict(y, use_trainable)
            trainable_out = getattr(self.trainable_model, fn_name)(
                x[use_trainable],
                timesteps[use_trainable],
                trainable_y,
            )
            if fn_name == "forward_with_hidden":
                trainable_out, trainable_hidden = trainable_out
                output[use_trainable] = trainable_out
                if hidden is None:
                    hidden = torch.empty(
                        (x.shape[0], trainable_hidden.shape[1], trainable_hidden.shape[2]),
                        device=trainable_hidden.device,
                        dtype=trainable_hidden.dtype,
                    )
                hidden[use_trainable] = trainable_hidden
            else:
                output[use_trainable] = trainable_out

        if (~use_trainable).any():
            frozen_mask = ~use_trainable
            frozen_y = _slice_condition_dict(y, frozen_mask)
            frozen_out = getattr(self.frozen_model, fn_name)(
                x[frozen_mask],
                timesteps[frozen_mask],
                frozen_y,
            )
            if fn_name == "forward_with_hidden":
                frozen_out, frozen_hidden = frozen_out
                output[frozen_mask] = frozen_out
                if hidden is None:
                    hidden = torch.empty(
                        (x.shape[0], frozen_hidden.shape[1], frozen_hidden.shape[2]),
                        device=frozen_hidden.device,
                        dtype=frozen_hidden.dtype,
                    )
                hidden[frozen_mask] = frozen_hidden
            else:
                output[frozen_mask] = frozen_out

        if fn_name == "forward_with_hidden":
            return output, hidden
        return output

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, y=None):
        return self._dispatch(x, timesteps, y or {}, "forward")

    def forward_with_hidden(self, x: torch.Tensor, timesteps: torch.Tensor, y=None):
        return self._dispatch(x, timesteps, y or {}, "forward_with_hidden")
