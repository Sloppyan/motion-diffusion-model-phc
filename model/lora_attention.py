import math
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAWeight(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank

        self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def weight(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling


class LoRAWeightBank(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        bank_name: str = "default",
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.banks = nn.ModuleDict()
        self.active_bank = str(bank_name)
        self.add_bank(self.active_bank)

    def add_bank(self, bank_name: str, init_from: Optional[str] = None) -> None:
        bank_name = str(bank_name)
        if bank_name in self.banks:
            return
        bank = LoRAWeight(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=self.rank,
            alpha=self.alpha,
        )
        if init_from is not None:
            if init_from not in self.banks:
                raise KeyError(f"Unknown source LoRA bank: {init_from}")
            bank.load_state_dict(self.banks[init_from].state_dict())
        self.banks[bank_name] = bank

    def set_active_bank(self, bank_name: str) -> None:
        bank_name = str(bank_name)
        if bank_name not in self.banks:
            raise KeyError(f"Unknown LoRA bank: {bank_name}")
        self.active_bank = bank_name

    def weight(self, bank_name: Optional[str] = None) -> torch.Tensor:
        selected = self.active_bank if bank_name is None else str(bank_name)
        return self.banks[selected].weight()

    def bank_parameters(self, bank_name: str) -> Iterable[nn.Parameter]:
        return self.banks[str(bank_name)].parameters()


class LoRAMultiheadAttentionWrapper(nn.Module):
    def __init__(
        self,
        base_attn: nn.MultiheadAttention,
        rank: int = 8,
        alpha: float = 16.0,
        target_modules: Sequence[str] = ("q_proj", "k_proj", "v_proj", "out_proj"),
        bank_name: str = "default",
    ):
        super().__init__()
        if not base_attn._qkv_same_embed_dim:
            raise NotImplementedError("Only same-qkv-dim attention is supported.")

        self.base_attn = base_attn
        self.target_modules = tuple(target_modules)
        self.active_lora_bank = str(bank_name)

        embed_dim = base_attn.embed_dim
        self.q_lora = (
            LoRAWeightBank(embed_dim, embed_dim, rank, alpha, bank_name=self.active_lora_bank)
            if "q_proj" in self.target_modules
            else None
        )
        self.k_lora = (
            LoRAWeightBank(embed_dim, embed_dim, rank, alpha, bank_name=self.active_lora_bank)
            if "k_proj" in self.target_modules
            else None
        )
        self.v_lora = (
            LoRAWeightBank(embed_dim, embed_dim, rank, alpha, bank_name=self.active_lora_bank)
            if "v_proj" in self.target_modules
            else None
        )
        self.out_lora = (
            LoRAWeightBank(embed_dim, embed_dim, rank, alpha, bank_name=self.active_lora_bank)
            if "out_proj" in self.target_modules
            else None
        )

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

    def add_lora_bank(self, bank_name: str, init_from_bank: Optional[str] = None) -> None:
        for lora in (self.q_lora, self.k_lora, self.v_lora, self.out_lora):
            if lora is None:
                continue
            lora.add_bank(bank_name, init_from=init_from_bank)

    def set_active_lora_bank(self, bank_name: str) -> None:
        self.active_lora_bank = str(bank_name)
        for lora in (self.q_lora, self.k_lora, self.v_lora, self.out_lora):
            if lora is None:
                continue
            lora.set_active_bank(bank_name)

    def _proj_delta(self, lora: LoRAWeightBank, like: torch.Tensor) -> torch.Tensor:
        if lora is None:
            return like.new_zeros((self.embed_dim, self.embed_dim))
        return lora.weight(self.active_lora_bank).to(dtype=like.dtype, device=like.device)

    def _in_proj_weight(self) -> torch.Tensor:
        weight = self.base_attn.in_proj_weight
        q_delta = self._proj_delta(self.q_lora, weight)
        k_delta = self._proj_delta(self.k_lora, weight)
        v_delta = self._proj_delta(self.v_lora, weight)
        return weight + torch.cat([q_delta, k_delta, v_delta], dim=0)

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


_LORA_MODULE_NAMES = ("q_lora", "k_lora", "v_lora", "out_lora")


def _resolve_layer_indices(num_layers: int, layer_scope: str) -> List[int]:
    if layer_scope == "all":
        return list(range(num_layers))
    if layer_scope == "last3":
        return list(range(max(0, num_layers - 3), num_layers))
    raise ValueError(f"Unsupported layer_scope: {layer_scope}")


def inject_lora_into_mdm(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    layer_scope: str = "all",
    target_modules: Sequence[str] = ("q_proj", "k_proj", "v_proj", "out_proj"),
    bank_name: str = "default",
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
            wrapper.add_lora_bank(bank_name)
        else:
            wrapper = LoRAMultiheadAttentionWrapper(
                base_attn=layer.self_attn,
                rank=rank,
                alpha=alpha,
                target_modules=target_modules,
                bank_name=bank_name,
            )
            layer.self_attn = wrapper
        injected.append((idx, wrapper))
    return injected


def _iter_lora_wrappers(model: nn.Module) -> Iterable[LoRAMultiheadAttentionWrapper]:
    for module in model.modules():
        if isinstance(module, LoRAMultiheadAttentionWrapper):
            yield module


def add_lora_bank_to_mdm(model: nn.Module, bank_name: str, init_from_bank: str = "default") -> None:
    for wrapper in _iter_lora_wrappers(model):
        wrapper.add_lora_bank(bank_name, init_from_bank=init_from_bank)


def set_active_lora_bank(model: nn.Module, bank_name: str) -> None:
    for wrapper in _iter_lora_wrappers(model):
        wrapper.set_active_lora_bank(bank_name)


def get_active_lora_bank(model: nn.Module) -> Optional[str]:
    for wrapper in _iter_lora_wrappers(model):
        return wrapper.active_lora_bank
    return None


def mark_only_lora_trainable(model: nn.Module, bank_name: str = "default") -> int:
    trainable = 0
    for param in model.parameters():
        param.requires_grad = False

    for module in _iter_lora_wrappers(model):
        for lora in (module.q_lora, module.k_lora, module.v_lora, module.out_lora):
            if lora is None:
                continue
            for param in lora.bank_parameters(bank_name):
                param.requires_grad = True
                trainable += param.numel()
    return trainable


def _is_lora_key(key: str) -> bool:
    return any(f".{name}." in key for name in _LORA_MODULE_NAMES)


def _legacy_to_bank_key(key: str, bank_name: str) -> str:
    for module_name in _LORA_MODULE_NAMES:
        plain = f".{module_name}."
        banked = f".{module_name}.banks."
        if banked in key:
            prefix, suffix = key.split(banked, 1)
            _, tail = suffix.split(".", 1)
            return f"{prefix}{banked}{bank_name}.{tail}"
        if plain in key:
            return key.replace(plain, f".{module_name}.banks.{bank_name}.", 1)
    return key


def _bank_to_legacy_key(key: str, bank_name: str) -> str:
    for module_name in _LORA_MODULE_NAMES:
        banked = f".{module_name}.banks.{bank_name}."
        if banked in key:
            return key.replace(banked, f".{module_name}.", 1)
    return key


def lora_state_dict(model: nn.Module, bank_name: str = "default"):
    state = {}
    for key, value in model.state_dict().items():
        if not _is_lora_key(key):
            continue
        remapped = _bank_to_legacy_key(key, bank_name)
        if remapped == key and ".banks." in key:
            continue
        state[remapped] = value.detach().cpu()
    return state


def load_lora_state_dict(model: nn.Module, state_dict, bank_name: str = "default") -> None:
    model_state = model.state_dict()
    target_keys = {
        key
        for key in model_state.keys()
        if _is_lora_key(key) and f".banks.{bank_name}." in key
    }
    remapped = {}
    for key, value in state_dict.items():
        if not _is_lora_key(key):
            continue
        banked_key = _legacy_to_bank_key(key, bank_name)
        remapped[banked_key] = value

    unexpected = sorted(key for key in remapped.keys() if key not in target_keys)
    missing = sorted(key for key in target_keys if key not in remapped)
    if unexpected:
        raise RuntimeError(f"Unexpected LoRA keys: {unexpected}")
    if missing:
        raise RuntimeError(f"Missing LoRA keys: {missing}")

    model.load_state_dict(remapped, strict=False)


def iter_lora_parameters(model: nn.Module, bank_name: str = "default") -> Iterable[nn.Parameter]:
    for wrapper in _iter_lora_wrappers(model):
        for lora in (wrapper.q_lora, wrapper.k_lora, wrapper.v_lora, wrapper.out_lora):
            if lora is None:
                continue
            yield from lora.bank_parameters(bank_name)
