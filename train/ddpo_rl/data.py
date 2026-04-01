import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np


_FAILURE_LOG_PATTERN = re.compile(r"db_key=(?P<db_key>[^\s]+).*?caption_idx=(?P<caption_idx>-?\d+)")
_FAILURE_PAIR_PATTERN = re.compile(r"^(?P<db_key>[A-Za-z0-9_]+)\s*[:\t, ]\s*(?P<caption_idx>-?\d+)\s*$")
_FAILURE_DBKEY_PATTERN = re.compile(r"^(?P<db_key>[A-Za-z0-9_]+)\s*$")


@dataclass
class PromptSample:
    db_key: str
    caption_idx: int
    text: str
    tokens: str
    length_20fps: int


@dataclass(frozen=True)
class FailureCaseKey:
    db_key: str
    caption_idx: Optional[int]


class HumanMLPromptDataset:
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        max_samples: int = 0,
        max_motion_frames: int = 120,
    ):
        self.data_root = Path(data_root).expanduser().resolve()
        self.text_root = self.data_root / "texts"
        self.motion_root = self.data_root / "new_joint_vecs"
        self.split_path = self.data_root / f"{split}.txt"
        self.max_motion_frames = int(max_motion_frames)
        self.samples = self._build_samples()
        if max_samples > 0:
            self.samples = self.samples[: int(max_samples)]
        if len(self.samples) == 0:
            raise RuntimeError("No prompt samples loaded for DDPO training.")

    def _build_samples(self) -> List[PromptSample]:
        if not self.split_path.is_file():
            raise FileNotFoundError(f"Split file not found: {self.split_path}")
        if not self.text_root.is_dir():
            raise FileNotFoundError(f"Text root not found: {self.text_root}")

        sample_ids = []
        with self.split_path.open("r", encoding="utf-8") as f:
            for line in f:
                sample_id = line.strip()
                if sample_id:
                    sample_ids.append(sample_id)

        samples: List[PromptSample] = []
        for sample_id in sample_ids:
            text_path = self.text_root / f"{sample_id}.txt"
            motion_path = self.motion_root / f"{sample_id}.npy"
            if not text_path.is_file() or not motion_path.is_file():
                continue

            try:
                motion_len = int(np.load(str(motion_path), mmap_mode="r").shape[0])
            except Exception:
                continue

            with text_path.open("r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            for caption_idx, line in enumerate(lines):
                parts = line.split("#")
                text = parts[0].strip()
                tokens = parts[1].strip() if len(parts) > 1 else ""
                length_20fps = self._resolve_length(parts, motion_len)
                if not text or length_20fps <= 1:
                    continue
                samples.append(
                    PromptSample(
                        db_key=sample_id,
                        caption_idx=caption_idx,
                        text=text,
                        tokens=tokens,
                        length_20fps=min(length_20fps, self.max_motion_frames),
                    )
                )
        return samples

    def _resolve_length(self, parts: List[str], motion_len: int) -> int:
        fallback = min(motion_len, self.max_motion_frames)
        if len(parts) < 4:
            return fallback

        try:
            f_tag = float(parts[2])
            to_tag = float(parts[3])
        except Exception:
            return fallback

        if np.isnan(f_tag):
            f_tag = 0.0
        if np.isnan(to_tag):
            to_tag = 0.0

        if f_tag == 0.0 and to_tag == 0.0:
            return fallback

        start = max(0, int(f_tag * 20.0))
        end = max(start + 1, int(to_tag * 20.0))
        start = min(start, max(0, motion_len - 1))
        end = min(max(start + 1, end), motion_len)
        return min(end - start, self.max_motion_frames)

    def draw_batch(self, batch_size: int, rng: np.random.RandomState) -> List[PromptSample]:
        replace = len(self.samples) < batch_size
        indices = rng.choice(len(self.samples), size=batch_size, replace=replace)
        return [self.samples[int(i)] for i in indices]

    def get_batch(self, indices: Sequence[int]) -> List[PromptSample]:
        return [self.samples[int(i)] for i in indices]

    def first_batch(self, batch_size: int) -> List[PromptSample]:
        return self.samples[: min(len(self.samples), int(batch_size))]

    def __len__(self) -> int:
        return len(self.samples)


class PromptSampleDataset:
    def __init__(self, samples: Sequence[PromptSample]):
        self.samples = list(samples)
        if len(self.samples) == 0:
            raise RuntimeError("PromptSampleDataset cannot be empty.")

    def get_batch(self, indices: Sequence[int]) -> List[PromptSample]:
        return [self.samples[int(i)] for i in indices]

    def first_batch(self, batch_size: int) -> List[PromptSample]:
        return self.samples[: min(len(self.samples), int(batch_size))]

    def __len__(self) -> int:
        return len(self.samples)


def _parse_failure_case_key(line: str) -> Optional[FailureCaseKey]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    match = _FAILURE_LOG_PATTERN.search(line)
    if match is not None:
        return FailureCaseKey(
            db_key=match.group("db_key"),
            caption_idx=int(match.group("caption_idx")),
        )

    match = _FAILURE_PAIR_PATTERN.match(line)
    if match is not None:
        return FailureCaseKey(
            db_key=match.group("db_key"),
            caption_idx=int(match.group("caption_idx")),
        )

    match = _FAILURE_DBKEY_PATTERN.match(line)
    if match is not None:
        return FailureCaseKey(
            db_key=match.group("db_key"),
            caption_idx=None,
        )

    return None


def load_failure_case_keys(path: str) -> List[FailureCaseKey]:
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"Failure case file not found: {file_path}")

    keys: List[FailureCaseKey] = []
    seen: Set[FailureCaseKey] = set()
    with file_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            key = _parse_failure_case_key(raw_line)
            if key is None or key in seen:
                continue
            seen.add(key)
            keys.append(key)

    if len(keys) == 0:
        raise RuntimeError(f"No failure cases could be parsed from: {file_path}")
    return keys


def filter_prompt_samples_by_keys(
    samples: Sequence[PromptSample],
    keys: Sequence[FailureCaseKey],
) -> List[PromptSample]:
    exact_keys = {(key.db_key, int(key.caption_idx)) for key in keys if key.caption_idx is not None}
    wildcard_db_keys = {key.db_key for key in keys if key.caption_idx is None}

    matched: List[PromptSample] = []
    for sample in samples:
        sample_key = (sample.db_key, int(sample.caption_idx))
        if sample.db_key in wildcard_db_keys or sample_key in exact_keys:
            matched.append(sample)
    return matched


class _CyclingSamplePool:
    def __init__(self, samples: Sequence[PromptSample], rng: np.random.RandomState, shuffle: bool = True):
        self.samples = list(samples)
        self.rng = rng
        self.shuffle = bool(shuffle)
        if len(self.samples) == 0:
            raise RuntimeError("Cycling sample pool cannot be empty.")
        self._order = np.empty((0,), dtype=np.int64)
        self._cursor = 0
        self._reset()

    def _reset(self):
        self._order = np.arange(len(self.samples), dtype=np.int64)
        if self.shuffle:
            self.rng.shuffle(self._order)
        self._cursor = 0

    def draw(self, count: int) -> List[PromptSample]:
        count = int(count)
        if count <= 0:
            return []

        if len(self.samples) < count:
            indices = self.rng.choice(len(self.samples), size=count, replace=True)
            return [self.samples[int(i)] for i in indices]

        batch: List[PromptSample] = []
        while len(batch) < count:
            if self._cursor >= len(self._order):
                self._reset()
            need = count - len(batch)
            end = min(self._cursor + need, len(self._order))
            indices = self._order[self._cursor:end]
            batch.extend(self.samples[int(i)] for i in indices)
            self._cursor = end
        return batch


class WithoutReplacementBatchSampler:
    def __init__(
        self,
        dataset,
        batch_size: int,
        rng: np.random.RandomState,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.rng = rng
        self.shuffle = bool(shuffle)
        self.epoch = -1
        self._order = np.empty((0,), dtype=np.int64)
        self._cursor = 0
        self._reset()

    @property
    def steps_per_epoch(self) -> int:
        return max(1, int(math.ceil(len(self.dataset) / float(self.batch_size))))

    def _reset(self):
        self._order = np.arange(len(self.dataset), dtype=np.int64)
        if self.shuffle:
            self.rng.shuffle(self._order)
        self._cursor = 0
        self.epoch += 1

    def next_batch(self) -> List[PromptSample]:
        if self._cursor >= len(self._order):
            self._reset()
        end = min(self._cursor + self.batch_size, len(self._order))
        batch = self.dataset.get_batch(self._order[self._cursor:end])
        self._cursor = end
        return batch


class FailureOnlyBatchSampler:
    def __init__(
        self,
        dataset,
        batch_size: int,
        rng: np.random.RandomState,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.rng = rng
        self.shuffle = bool(shuffle)
        self.epoch = -1
        self._step_in_epoch = 0
        self._steps_per_epoch = max(1, int(math.ceil(len(self.dataset) / float(self.batch_size))))
        self._pool = _CyclingSamplePool(self.dataset.samples, rng=self.rng, shuffle=self.shuffle)

    @property
    def steps_per_epoch(self) -> int:
        return self._steps_per_epoch

    def next_batch(self) -> List[PromptSample]:
        if self._step_in_epoch == 0:
            self.epoch += 1

        batch = self._pool.draw(self.batch_size)
        self._step_in_epoch += 1
        if self._step_in_epoch >= self._steps_per_epoch:
            self._step_in_epoch = 0
        return batch


class MixedBatchSampler:
    def __init__(
        self,
        base_dataset,
        failure_dataset,
        batch_size: int,
        failure_batch_size: int,
        rng: np.random.RandomState,
        shuffle: bool = True,
    ):
        self.base_dataset = base_dataset
        self.failure_dataset = failure_dataset
        self.batch_size = int(batch_size)
        self.failure_batch_size = int(failure_batch_size)
        self.base_batch_size = self.batch_size - self.failure_batch_size
        self.rng = rng
        self.shuffle = bool(shuffle)

        if self.failure_batch_size <= 0 or self.failure_batch_size >= self.batch_size:
            raise ValueError("failure_batch_size must satisfy 0 < failure_batch_size < batch_size.")

        self.epoch = -1
        self._step_in_epoch = 0
        self._steps_per_epoch = max(
            1,
            int(math.ceil(len(self.base_dataset) / float(self.base_batch_size))),
            int(math.ceil(len(self.failure_dataset) / float(self.failure_batch_size))),
        )
        self._base_pool = _CyclingSamplePool(self.base_dataset.samples, rng=self.rng, shuffle=self.shuffle)
        self._failure_pool = _CyclingSamplePool(self.failure_dataset.samples, rng=self.rng, shuffle=self.shuffle)

    @property
    def steps_per_epoch(self) -> int:
        return self._steps_per_epoch

    def next_batch(self) -> List[PromptSample]:
        if self._step_in_epoch == 0:
            self.epoch += 1

        ##############################
        # Draw from the failure pool and the base pool separately so the batch
        # composition stays fixed, then shuffle the merged prompt order.
        ##############################
        batch = self._failure_pool.draw(self.failure_batch_size)
        batch.extend(self._base_pool.draw(self.base_batch_size))
        if self.shuffle and len(batch) > 1:
            order = self.rng.permutation(len(batch))
            batch = [batch[int(idx)] for idx in order]

        self._step_in_epoch += 1
        if self._step_in_epoch >= self._steps_per_epoch:
            self._step_in_epoch = 0
        return batch
