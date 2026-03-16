import contextlib

import torch


class MDMHiddenProbe:
    def __init__(self, model, layer_indices):
        if getattr(model, 'arch', None) != 'trans_enc':
            raise ValueError('MDMHiddenProbe currently supports arch=trans_enc only.')

        self.model = model
        self.layer_indices = tuple(layer_indices)
        self._hidden = {}
        self._hooks = []

        num_layers = len(self.model.seqTransEncoder.layers)
        for layer_idx in self.layer_indices:
            if layer_idx < 0 or layer_idx >= num_layers:
                raise ValueError(f'layer index {layer_idx} is out of range for {num_layers} layers')
            layer = self.model.seqTransEncoder.layers[layer_idx]
            self._hooks.append(layer.register_forward_hook(self._make_hook(layer_idx)))

    def _make_hook(self, layer_idx):
        def hook(_module, _inputs, output):
            self._hidden[layer_idx] = output
        return hook

    def clear(self):
        self._hidden.clear()

    def get_hidden_states(self):
        outputs = []
        for layer_idx in self.layer_indices:
            if layer_idx not in self._hidden:
                raise RuntimeError(f'failed to capture hidden state for layer {layer_idx}')
            layer_hidden = self._hidden[layer_idx]
            if layer_hidden.dim() != 3:
                raise RuntimeError(
                    f'unexpected hidden shape for layer {layer_idx}: {tuple(layer_hidden.shape)}'
                )
            # Drop the prepended timestep/condition token: [T+1, B, D] -> [B, T, D]
            outputs.append(layer_hidden[1:].permute(1, 0, 2).contiguous())
        return outputs

    @torch.no_grad()
    def extract(self, x, timesteps, y):
        self.clear()
        self.model(x, timesteps, y)
        return self.get_hidden_states()

    def close(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


@contextlib.contextmanager
def hidden_probe(model, layer_indices):
    probe = MDMHiddenProbe(model=model, layer_indices=layer_indices)
    try:
        yield probe
    finally:
        probe.close()
