import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


_FAIL_CASE_PATTERN = re.compile(r"db_key=(?P<db_key>\S+)\s+caption_idx=(?P<caption_idx>-?\d+)")


@dataclass(frozen=True)
class PromptEntry:
    db_key: str
    caption_idx: int
    caption: str
    tokens: str
    length_20fps: int


@dataclass(frozen=True)
class PromptEntryPools:
    all_entries: Sequence[PromptEntry]
    failure_entries: Sequence[PromptEntry]
    success_entries: Sequence[PromptEntry]


def _parse_text_line(line: str) -> Tuple[str, str, float, float]:
    parts = line.strip().split("#")
    if len(parts) < 4:
        raise ValueError(f"Unexpected text format: {line}")
    caption = parts[0].strip()
    tokens = parts[1].strip()
    f_tag = float(parts[2].strip())
    to_tag = float(parts[3].strip())
    return caption, tokens, f_tag, to_tag


def _compute_caption_length(motion_len_20fps: int, f_tag: float, to_tag: float) -> int:
    if np.isnan(f_tag) or np.isnan(to_tag):
        return max(1, int(motion_len_20fps))
    if abs(f_tag) < 1e-6 and abs(to_tag) < 1e-6:
        return max(1, int(motion_len_20fps))
    start = max(0, int(f_tag * 20.0))
    end = max(start + 1, int(to_tag * 20.0))
    return max(1, end - start)


def _load_failure_index(path: Path) -> set:
    keep = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = _FAIL_CASE_PATTERN.search(line)
            if match is None:
                continue
            keep.add((match.group("db_key"), int(match.group("caption_idx"))))
    return keep


def build_prompt_entries(
    data_root: str,
    split: str,
    max_motion_frames: int,
    failure_cases_file: Optional[str] = None,
) -> List[PromptEntry]:
    data_root_path = Path(data_root).expanduser().resolve()
    split_path = data_root_path / f"{split}.txt"
    text_root = data_root_path / "texts"
    motion_root = data_root_path / "new_joint_vecs"
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    if not text_root.is_dir():
        raise FileNotFoundError(f"Text root not found: {text_root}")
    if not motion_root.is_dir():
        raise FileNotFoundError(f"Motion root not found: {motion_root}")

    failure_index = None
    if failure_cases_file:
        failure_index = _load_failure_index(Path(failure_cases_file).expanduser().resolve())

    entries: List[PromptEntry] = []
    with split_path.open("r", encoding="utf-8") as split_handle:
        for raw in split_handle:
            db_key = raw.strip()
            if not db_key:
                continue
            text_path = text_root / f"{db_key}.txt"
            motion_path = motion_root / f"{db_key}.npy"
            if not text_path.is_file() or not motion_path.is_file():
                continue

            motion_len = int(np.load(motion_path, mmap_mode="r").shape[0])
            with text_path.open("r", encoding="utf-8") as text_handle:
                for caption_idx, line in enumerate(text_handle):
                    line = line.strip()
                    if not line:
                        continue
                    if failure_index is not None and (db_key, caption_idx) not in failure_index:
                        continue
                    caption, tokens, f_tag, to_tag = _parse_text_line(line)
                    length_20fps = min(
                        max_motion_frames,
                        _compute_caption_length(motion_len, f_tag, to_tag),
                    )
                    entries.append(
                        PromptEntry(
                            db_key=db_key,
                            caption_idx=caption_idx,
                            caption=caption,
                            tokens=tokens,
                            length_20fps=length_20fps,
                        )
                    )
    if not entries:
        raise RuntimeError(
            f"No prompt entries found for split={split} under {data_root_path} "
            f"(failure filter: {failure_cases_file or 'none'})."
        )
    return entries


def build_prompt_entry_pools(
    data_root: str,
    split: str,
    max_motion_frames: int,
    failure_cases_file: Optional[str] = None,
) -> PromptEntryPools:
    all_entries = build_prompt_entries(
        data_root=data_root,
        split=split,
        max_motion_frames=max_motion_frames,
        failure_cases_file=None,
    )

    if not failure_cases_file:
        return PromptEntryPools(
            all_entries=all_entries,
            failure_entries=[],
            success_entries=all_entries,
        )

    failure_index = _load_failure_index(Path(failure_cases_file).expanduser().resolve())
    failure_entries: List[PromptEntry] = []
    success_entries: List[PromptEntry] = []
    for entry in all_entries:
        if (entry.db_key, entry.caption_idx) in failure_index:
            failure_entries.append(entry)
        else:
            success_entries.append(entry)

    return PromptEntryPools(
        all_entries=all_entries,
        failure_entries=failure_entries,
        success_entries=success_entries,
    )


def _sample_from_pool(
    entries: Sequence[PromptEntry],
    batch_size: int,
    rng: random.Random,
) -> List[PromptEntry]:
    if batch_size <= 0:
        return []
    if not entries:
        raise RuntimeError("Cannot sample from an empty prompt pool.")
    if len(entries) >= batch_size:
        return rng.sample(list(entries), batch_size)
    return [rng.choice(entries) for _ in range(batch_size)]


def sample_prompt_batch(
    pools: PromptEntryPools,
    mode: str,
    batch_size: int,
    rng: random.Random,
    success_sampling_weight: float = 1.0,
    failure_sampling_weight: float = 1.0,
) -> Tuple[List[PromptEntry], Dict[str, float]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if mode == "uniform":
        batch = _sample_from_pool(pools.all_entries, batch_size, rng)
        return batch, {
            "failure_batch_frac": 0.0,
            "success_batch_frac": 0.0,
        }

    if mode == "failure_only":
        batch = _sample_from_pool(pools.failure_entries, batch_size, rng)
        return batch, {
            "failure_batch_frac": 1.0,
            "success_batch_frac": 0.0,
        }

    if mode != "mixed":
        raise ValueError(f"Unsupported sampling mode: {mode}")

    if success_sampling_weight < 0.0 or failure_sampling_weight < 0.0:
        raise ValueError("Sampling weights must be non-negative.")
    weight_sum = float(success_sampling_weight + failure_sampling_weight)
    if weight_sum <= 0.0:
        raise ValueError("At least one sampling weight must be positive.")

    ##############################################
    # Build a mixed batch from disjoint success/failure pools.
    # The user controls the expected class ratio via relative weights.
    ##############################################
    fail_frac = float(failure_sampling_weight) / weight_sum
    failure_batch_size = int(round(batch_size * fail_frac))
    success_batch_size = batch_size - failure_batch_size

    if failure_batch_size > 0 and not pools.failure_entries:
        raise RuntimeError("Mixed sampling requested failure samples, but failure pool is empty.")
    if success_batch_size > 0 and not pools.success_entries:
        raise RuntimeError("Mixed sampling requested success samples, but success pool is empty.")

    batch = _sample_from_pool(pools.failure_entries, failure_batch_size, rng)
    batch.extend(_sample_from_pool(pools.success_entries, success_batch_size, rng))
    rng.shuffle(batch)
    return batch, {
        "failure_batch_frac": float(failure_batch_size) / float(batch_size),
        "success_batch_frac": float(success_batch_size) / float(batch_size),
    }


def select_eval_batch(
    entries: Sequence[PromptEntry],
    batch_size: int,
    offset: int = 0,
) -> List[PromptEntry]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not entries:
        return []
    start = offset % len(entries)
    ordered = list(entries[start:]) + list(entries[:start])
    return ordered[: min(batch_size, len(ordered))]


def prompt_entries_to_meta(entries: Sequence[PromptEntry]) -> List[Dict]:
    meta: List[Dict] = []
    for sample_idx, entry in enumerate(entries):
        meta.append(
            {
                "sample_idx": sample_idx,
                "db_key": entry.db_key,
                "caption_idx": entry.caption_idx,
                "caption": entry.caption,
                "tokens": entry.tokens,
                "length": entry.length_20fps,
                "gt_len_20fps": entry.length_20fps,
                "text": entry.caption,
            }
        )
    return meta
