import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, inject_adapter_in_model


def _active_adapters(module):
    adapters = getattr(module, 'active_adapters', None)
    if adapters is None:
        return []
    if callable(adapters):
        adapters = adapters()
    return list(adapters)


def _effective_linear_weight_and_bias(module):
    if hasattr(module, 'base_layer') and hasattr(module, 'get_delta_weight'):
        base_weight = module.base_layer.weight.detach().clone()
        if not getattr(module, 'merged', False):
            for adapter_name in _active_adapters(module):
                base_weight = base_weight + module.get_delta_weight(adapter_name).detach().to(base_weight.dtype)
        base_bias = None
        if module.base_layer.bias is not None:
            base_bias = module.base_layer.bias.detach().clone()
        return base_weight, base_bias

    weight = module.weight.detach().clone()
    bias = None
    if module.bias is not None:
        bias = module.bias.detach().clone()
    return weight, bias


class LoRAMultiheadAttentionWrapper(nn.Module):
    def __init__(self, attention_module):
        super().__init__()
        if attention_module.bias_k is not None or attention_module.bias_v is not None:
            raise ValueError('bias_k and bias_v are not supported by LoRAMultiheadAttentionWrapper')
        if attention_module.add_zero_attn:
            raise ValueError('add_zero_attn=True is not supported by LoRAMultiheadAttentionWrapper')

        device = attention_module.out_proj.weight.device
        dtype = attention_module.out_proj.weight.dtype
        has_bias = attention_module.in_proj_bias is not None

        self.embed_dim = attention_module.embed_dim
        self.num_heads = attention_module.num_heads
        self.dropout = float(attention_module.dropout)
        self.head_dim = self.embed_dim // self.num_heads
        self.batch_first = bool(attention_module.batch_first)

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=has_bias, device=device, dtype=dtype)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=has_bias, device=device, dtype=dtype)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=has_bias, device=device, dtype=dtype)
        self.out_proj = nn.Linear(
            self.embed_dim,
            self.embed_dim,
            bias=attention_module.out_proj.bias is not None,
            device=device,
            dtype=dtype,
        )

        q_weight, k_weight, v_weight = attention_module.in_proj_weight.detach().chunk(3, dim=0)
        self.q_proj.weight.data.copy_(q_weight)
        self.k_proj.weight.data.copy_(k_weight)
        self.v_proj.weight.data.copy_(v_weight)

        if has_bias:
            q_bias, k_bias, v_bias = attention_module.in_proj_bias.detach().chunk(3, dim=0)
            self.q_proj.bias.data.copy_(q_bias)
            self.k_proj.bias.data.copy_(k_bias)
            self.v_proj.bias.data.copy_(v_bias)

        self.out_proj.weight.data.copy_(attention_module.out_proj.weight.detach())
        if attention_module.out_proj.bias is not None:
            self.out_proj.bias.data.copy_(attention_module.out_proj.bias.detach())

    def _shape_projection(self, tensor, seq_len, batch_size):
        return tensor.contiguous().view(seq_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

    def _to_additive_mask(self, attn_mask, batch_size, tgt_len, src_len, device, dtype):
        if attn_mask is None:
            return None

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
        elif attn_mask.dim() == 3:
            if attn_mask.shape[0] == batch_size * self.num_heads:
                attn_mask = attn_mask.view(batch_size, self.num_heads, tgt_len, src_len)
            elif attn_mask.shape[0] == batch_size:
                attn_mask = attn_mask.unsqueeze(1)
            else:
                raise ValueError(f'unsupported 3D attn_mask shape: {tuple(attn_mask.shape)}')
        elif attn_mask.dim() != 4:
            raise ValueError(f'unsupported attn_mask rank: {attn_mask.dim()}')

        if attn_mask.dtype == torch.bool:
            additive_mask = torch.zeros(attn_mask.shape, device=device, dtype=dtype)
            additive_mask.masked_fill_(attn_mask.to(device), float('-inf'))
            return additive_mask

        return attn_mask.to(device=device, dtype=dtype)

    def _compute_attention(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        attn_mask=None,
        need_weights=True,
        average_attn_weights=True,
        is_causal=False,
    ):
        if query.dim() == 2:
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
            squeeze_batch = True
        else:
            squeeze_batch = False

        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        tgt_len, batch_size, embed_dim = query.shape
        src_len = key.shape[0]
        if embed_dim != self.embed_dim:
            raise ValueError(f'expected embed_dim={self.embed_dim}, got {embed_dim}')

        q = self._shape_projection(self.q_proj(query), tgt_len, batch_size)
        k = self._shape_projection(self.k_proj(key), src_len, batch_size)
        v = self._shape_projection(self.v_proj(value), src_len, batch_size)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        additive_mask = self._to_additive_mask(
            attn_mask=attn_mask,
            batch_size=batch_size,
            tgt_len=tgt_len,
            src_len=src_len,
            device=scores.device,
            dtype=scores.dtype,
        )
        if additive_mask is not None:
            scores = scores + additive_mask

        if key_padding_mask is not None:
            if key_padding_mask.dtype == torch.bool:
                scores = scores.masked_fill(key_padding_mask[:, None, None, :].to(scores.device), float('-inf'))
            else:
                scores = scores + key_padding_mask[:, None, None, :].to(device=scores.device, dtype=scores.dtype)

        if is_causal:
            causal_mask = torch.ones((tgt_len, src_len), device=scores.device, dtype=torch.bool).triu(diagonal=1)
            scores = scores.masked_fill(causal_mask[None, None, :, :], float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        if self.dropout > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(tgt_len, batch_size, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        if squeeze_batch:
            attn_output = attn_output.squeeze(1)

        if not need_weights:
            return attn_output, None

        weights = attn_weights
        if average_attn_weights:
            weights = weights.mean(dim=1)

        if squeeze_batch:
            weights = weights.squeeze(0)

        return attn_output, weights

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
        return self._compute_attention(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )

    def merged_parameters(self):
        q_weight, q_bias = _effective_linear_weight_and_bias(self.q_proj)
        k_weight, k_bias = _effective_linear_weight_and_bias(self.k_proj)
        v_weight, v_bias = _effective_linear_weight_and_bias(self.v_proj)
        out_weight, out_bias = _effective_linear_weight_and_bias(self.out_proj)

        in_proj_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        in_proj_bias = None
        if q_bias is not None:
            in_proj_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
        return in_proj_weight, in_proj_bias, out_weight, out_bias


def apply_lora_to_last_layers(model, layer_indices, rank, alpha, dropout):
    wrapped_layers = OrderedDict()
    for layer_idx in layer_indices:
        layer = model.seqTransEncoder.layers[layer_idx]
        wrapper = LoRAMultiheadAttentionWrapper(layer.self_attn)
        for param in wrapper.parameters():
            param.requires_grad = False
        layer.self_attn = wrapper

        config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias='none',
            target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj'],
        )
        inject_adapter_in_model(config, wrapper)
        wrapped_layers[layer_idx] = wrapper
    return wrapped_layers


def export_lora_merged_mdm_state_dict(model, wrapped_layers):
    state_dict = OrderedDict()
    wrapped_prefixes = tuple(f'seqTransEncoder.layers.{idx}.self_attn.' for idx in wrapped_layers.keys())

    for name, value in model.state_dict().items():
        if name.startswith('clip_model.'):
            continue
        if name.startswith(wrapped_prefixes):
            continue
        state_dict[name] = value.detach().cpu()

    for layer_idx, wrapper in wrapped_layers.items():
        prefix = f'seqTransEncoder.layers.{layer_idx}.self_attn'
        in_proj_weight, in_proj_bias, out_weight, out_bias = wrapper.merged_parameters()
        state_dict[f'{prefix}.in_proj_weight'] = in_proj_weight.cpu()
        if in_proj_bias is not None:
            state_dict[f'{prefix}.in_proj_bias'] = in_proj_bias.cpu()
        state_dict[f'{prefix}.out_proj.weight'] = out_weight.cpu()
        if out_bias is not None:
            state_dict[f'{prefix}.out_proj.bias'] = out_bias.cpu()

    return state_dict


def filter_clip_from_state_dict(state_dict):
    return OrderedDict(
        (name, tensor.detach().cpu())
        for name, tensor in state_dict.items()
        if not name.startswith('clip_model.')
    )
