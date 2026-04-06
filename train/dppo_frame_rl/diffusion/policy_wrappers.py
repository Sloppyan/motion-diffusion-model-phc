from typing import Any, Dict

import torch
import torch.nn as nn

from model.cfg_sampler import ClassifierFreeSampleModel
from model.lora_attention import get_active_lora_bank, set_active_lora_bank

class LoRABankSelector(nn.Module):
    """
    Bind a shared MDM backbone to one named LoRA bank during forward passes.
    """

    def __init__(self, model: nn.Module, bank_name: str):
        super().__init__()
        self.model = model
        self.bank_name = str(bank_name)

        for name in ("rot2xyz", "translation", "njoints", "nfeats", "data_rep", "cond_mode", "cond_mask_prob"):
            if hasattr(model, name):
                setattr(self, name, getattr(model, name))
        if hasattr(model, "encode_text"):
            self.encode_text = model.encode_text

    def forward(self, x, timesteps, y=None):
        previous_bank = get_active_lora_bank(self.model)
        if previous_bank != self.bank_name:
            set_active_lora_bank(self.model, self.bank_name)
        try:
            return self.model(x, timesteps, y)
        finally:
            if previous_bank is not None and previous_bank != self.bank_name:
                set_active_lora_bank(self.model, previous_bank)


def build_sampling_actor(model: nn.Module, guidance_param: float, bank_name: str = "") -> nn.Module:
    actor: nn.Module = model if not bank_name else LoRABankSelector(model, bank_name)
    if float(guidance_param) == 1.0:
        return actor
    return ClassifierFreeSampleModel(actor)


def select_condition_batch(y: Dict[str, Any], mask: torch.Tensor) -> Dict[str, Any]:
    selected: Dict[str, Any] = {}
    keep = mask.tolist()
    batch = len(keep)
    for key, value in y.items():
        if torch.is_tensor(value) and value.shape[:1] == (batch,):
            selected[key] = value[mask]
        elif isinstance(value, list) and len(value) == batch:
            selected[key] = [item for item, active in zip(value, keep) if active]
        else:
            selected[key] = value
    return selected


def resolve_trainable_step_count(num_timesteps: int, ft_denoising_steps: int) -> int:
    stochastic_steps = max(0, num_timesteps - 1)
    if ft_denoising_steps <= 0:
        return stochastic_steps
    return min(int(ft_denoising_steps), stochastic_steps)


class HybridSuffixPolicy(nn.Module):
    """
    Use the frozen base actor on the denoising prefix and the trainable actor on
    the last-K denoising suffix, following the DPPO fine-tuning pattern.
    """

    def __init__(
        self,
        base_model: nn.Module,
        ft_model: nn.Module,
        trainable_steps: int,
    ):
        super().__init__()
        self.base_model = base_model
        self.ft_model = ft_model
        self.trainable_steps = int(trainable_steps)

        template = self.ft_model
        for name in ("rot2xyz", "translation", "njoints", "nfeats", "data_rep", "cond_mode"):
            if hasattr(template, name):
                setattr(self, name, getattr(template, name))
        if hasattr(template, "encode_text"):
            self.encode_text = template.encode_text

    def _use_finetuned_mask(self, timesteps: torch.Tensor) -> torch.Tensor:
        if self.trainable_steps <= 0:
            return torch.ones_like(timesteps, dtype=torch.bool)
        return timesteps <= self.trainable_steps

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, y=None):
        if y is None:
            y = {}
        out = self.base_model(x, timesteps, y)
        ft_mask = self._use_finetuned_mask(timesteps)
        if bool(ft_mask.any().item()):
            out_ft = self.ft_model(
                x[ft_mask],
                timesteps[ft_mask],
                select_condition_batch(y, ft_mask),
            )
            out[ft_mask] = out_ft
        return out
