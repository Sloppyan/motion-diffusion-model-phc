import json
import pickle
import random
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import Dataset


def _load_jsonl(path):
    items = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


@lru_cache(maxsize=4096)
def _load_motion(path):
    return np.load(path).astype(np.float32)


@lru_cache(maxsize=4096)
def _load_text_lines(path):
    captions = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            captions.append(line.split('#', 1)[0].strip())
    return tuple(captions)


@lru_cache(maxsize=4096)
def _load_pickle_dict(path):
    try:
        return joblib.load(path)
    except Exception:
        with open(path, 'rb') as handle:
            return pickle.load(handle)


def _resample_sequence(sequence, source_fps, target_fps):
    if source_fps == target_fps:
        return sequence
    target_len = max(1, int(np.round(len(sequence) * float(target_fps) / float(source_fps))))
    source_indices = np.round(np.arange(target_len) * float(source_fps) / float(target_fps)).astype(np.int64)
    source_indices = np.clip(source_indices, 0, len(sequence) - 1)
    return sequence[source_indices]


class Stage1RepaDataset(Dataset):
    def __init__(self, manifest_path, split, target_fps=20, train=True):
        self.manifest_path = str(manifest_path)
        self.split = split
        self.target_fps = target_fps
        self.train = train
        self.items = _load_jsonl(self.manifest_path)
        if not self.items:
            raise ValueError(f'empty manifest: {self.manifest_path}')

        first_motion_path = Path(self.items[0]['motion_path'])
        humanml_root = first_motion_path.parents[1]
        self.mean = np.load(humanml_root / 'Mean.npy').astype(np.float32)
        self.std = np.load(humanml_root / 'Std.npy').astype(np.float32)

        lengths = [int(item.get('motion_length', 0)) for item in self.items if int(item.get('motion_length', 0)) > 0]
        self.stats = {
            'split': split,
            'num_samples': len(self.items),
            'avg_seq_len': float(np.mean(lengths)) if lengths else 0.0,
            'min_seq_len': int(np.min(lengths)) if lengths else 0,
            'max_seq_len': int(np.max(lengths)) if lengths else 0,
        }

    def __len__(self):
        return len(self.items)

    def _load_caption(self, text_path):
        captions = _load_text_lines(text_path)
        if not captions:
            return ''
        if self.train:
            return random.choice(captions)
        return captions[0]

    def _load_pulse(self, item):
        payload = _load_pickle_dict(item['pulse_path'])
        if not isinstance(payload, dict):
            raise TypeError(f"expected dict payload in {item['pulse_path']}")
        if not payload.get('is_succ', False):
            raise ValueError(f"invalid sample with is_succ=False: {item['pulse_path']}")
        if 'pulse_z' not in payload:
            raise KeyError(f"missing pulse_z in {item['pulse_path']}")

        pulse_z = np.asarray(payload['pulse_z'], dtype=np.float32)
        pulse_fps = int(payload.get('fps', 30))
        pulse_z = _resample_sequence(pulse_z, source_fps=pulse_fps, target_fps=self.target_fps)
        return pulse_z

    def __getitem__(self, index):
        item = self.items[index]
        motion = _load_motion(item['motion_path']).copy()
        pulse_z = self._load_pulse(item)
        caption = self._load_caption(item['text_path'])

        seq_len = min(len(motion), len(pulse_z))
        if seq_len <= 0:
            raise ValueError(f"empty aligned sequence for sample {item['id']}")

        motion = motion[:seq_len]
        pulse_z = pulse_z[:seq_len]
        motion = (motion - self.mean) / self.std

        return {
            'motion': torch.from_numpy(motion),
            'pulse_z': torch.from_numpy(pulse_z),
            'length': seq_len,
            'caption': caption,
            'id': item['id'],
        }


def stage1_repa_collate(batch):
    batch = [sample for sample in batch if sample is not None]
    if not batch:
        raise ValueError('received an empty batch')

    lengths = torch.as_tensor([sample['length'] for sample in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    batch_size = len(batch)

    motion = torch.zeros(batch_size, max_len, batch[0]['motion'].shape[-1], dtype=torch.float32)
    pulse_z = torch.zeros(batch_size, max_len, batch[0]['pulse_z'].shape[-1], dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    captions = []
    ids = []
    for idx, sample in enumerate(batch):
        seq_len = sample['length']
        motion[idx, :seq_len] = sample['motion']
        pulse_z[idx, :seq_len] = sample['pulse_z']
        mask[idx, :seq_len] = True
        captions.append(sample['caption'])
        ids.append(sample['id'])

    return {
        'motion': motion,
        'pulse_z': pulse_z,
        'mask': mask,
        'lengths': lengths,
        'caption': captions,
        'ids': ids,
    }
