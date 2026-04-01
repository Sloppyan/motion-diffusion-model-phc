import math
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAWeight(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def weight(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling


class LoRAMultiheadAttentionWrapper(nn.Module):
    def __init__(
        self,
        base_attn: nn.MultiheadAttention,
        rank: int = 8,
        alpha: float = 16.0,
        target_modules: Sequence[str] = ("q_proj", "k_proj", "v_proj", "out_proj"),
    ):
        super().__init__()
        if not base_attn._qkv_same_embed_dim:
            raise NotImplementedError("Only same-qkv-dim attention is supported.")

        self.base_attn = base_attn
        self.target_modules = tuple(target_modules)

        embed_dim = base_attn.embed_dim
        self.q_lora = LoRAWeight(embed_dim, embed_dim, rank, alpha) if "q_proj" in self.target_modules else None
        self.k_lora = LoRAWeight(embed_dim, embed_dim, rank, alpha) if "k_proj" in self.target_modules else None
        self.v_lora = LoRAWeight(embed_dim, embed_dim, rank, alpha) if "v_proj" in self.target_modules else None
        self.out_lora = LoRAWeight(embed_dim, embed_dim, rank, alpha) if "out_proj" in self.target_modules else None

    @property
    def embed_dim(self):
        return self.base_attn.embed_dim

    @property
    def kdim(self):
        return self.base_attn.kdim

    @property
    def vdim(self):
        return self.base_attn.vdim

    @property
    def num_heads(self):
        return self.base_attn.num_heads

    @property
    def dropout(self):
        return self.base_attn.dropout

    @property
    def head_dim(self):
        return self.base_attn.head_dim

    @property
    def batch_first(self):
        return self.base_attn.batch_first

    @property
    def bias_k(self):
        return self.base_attn.bias_k

    @property
    def bias_v(self):
        return self.base_attn.bias_v

    @property
    def add_zero_attn(self):
        return self.base_attn.add_zero_attn

    @property
    def _qkv_same_embed_dim(self):
        return self.base_attn._qkv_same_embed_dim

    @property
    def in_proj_bias(self):
        return self.base_attn.in_proj_bias

    @property
    def out_proj(self):
        return self.base_attn.out_proj

    def _proj_delta(self, lora: LoRAWeight, like: torch.Tensor) -> torch.Tensor:
        if lora is None:
            return like.new_zeros((self.embed_dim, self.embed_dim))
        return lora.weight().to(dtype=like.dtype, device=like.device)

    def _in_proj_weight(self) -> torch.Tensor:
        weight = self.base_attn.in_proj_weight
        q_delta = self._proj_delta(self.q_lora, weight)
        k_delta = self._proj_delta(self.k_lora, weight)
        v_delta = self._proj_delta(self.v_lora, weight)
        delta = torch.cat([q_delta, k_delta, v_delta], dim=0)
        return weight + delta

    def _out_proj_weight(self) -> torch.Tensor:
        weight = self.base_attn.out_proj.weight
        if self.out_lora is None:
            return weight
        return weight + self.out_lora.weight().to(dtype=weight.dtype, device=weight.device)

    @property
    def in_proj_weight(self):
        return self._in_proj_weight()

    def merge_masks(self, attn_mask, key_padding_mask, query):
        if hasattr(self.base_attn, "merge_masks"):
            return self.base_attn.merge_masks(attn_mask, key_padding_mask, query)
        return attn_mask, None

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
    ):
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        attn_output, attn_weights = F.multi_head_attention_forward(
            query=query,
            key=key,
            value=value,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=self._in_proj_weight(),
            in_proj_bias=self.in_proj_bias,
            bias_k=self.bias_k,
            bias_v=self.bias_v,
            add_zero_attn=self.add_zero_attn,
            dropout_p=self.dropout,
            out_proj_weight=self._out_proj_weight(),
            out_proj_bias=self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )

        if self.batch_first and is_batched:
            attn_output = attn_output.transpose(1, 0)

        return attn_output, attn_weights


def _resolve_layer_indices(num_layers: int, layer_scope: str) -> List[int]:
    if layer_scope == "all":
        return list(range(num_layers))
    if layer_scope == "last3":
        start = max(0, num_layers - 3)
        return list(range(start, num_layers))
    raise ValueError(f"Unsupported layer_scope: {layer_scope}")


def inject_lora_into_mdm(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    layer_scope: str = "all",
    target_modules: Sequence[str] = ("q_proj", "k_proj", "v_proj", "out_proj"),
) -> List[Tuple[int, LoRAMultiheadAttentionWrapper]]:
    if getattr(model, "arch", None) != "trans_enc":
        raise NotImplementedError("LoRA injection currently supports arch='trans_enc' only.")
    if not hasattr(model, "seqTransEncoder"):
        raise ValueError("Expected model.seqTransEncoder to exist.")

    layers = list(model.seqTransEncoder.layers)
    target_indices = _resolve_layer_indices(len(layers), layer_scope)
    injected = []
    for idx in target_indices:
        layer = layers[idx]
        if isinstance(layer.self_attn, LoRAMultiheadAttentionWrapper):
            wrapper = layer.self_attn
        else:
            wrapper = LoRAMultiheadAttentionWrapper(
                base_attn=layer.self_attn,
                rank=rank,
                alpha=alpha,
                target_modules=target_modules,
            )
            layer.self_attn = wrapper
        injected.append((idx, wrapper))
    return injected


def mark_only_lora_trainable(model: nn.Module) -> int:
    trainable = 0
    for param in model.parameters():
        param.requires_grad = False

    for module in model.modules():
        if not isinstance(module, LoRAMultiheadAttentionWrapper):
            continue
        for lora in (module.q_lora, module.k_lora, module.v_lora, module.out_lora):
            if lora is None:
                continue
            for param in lora.parameters():
                param.requires_grad = True
                trainable += param.numel()
    return trainable


def lora_state_dict(model: nn.Module):
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}


def load_lora_state_dict(model: nn.Module, state_dict):
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    unexpected = [k for k in unexpected if "lora_" in k]
    missing = [k for k in missing if "lora_" in k]
    if unexpected:
        raise RuntimeError(f"Unexpected LoRA keys: {unexpected}")
    if missing:
        raise RuntimeError(f"Missing LoRA keys: {missing}")


def iter_lora_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    for param in model.parameters():
        if param.requires_grad:
            yield param
