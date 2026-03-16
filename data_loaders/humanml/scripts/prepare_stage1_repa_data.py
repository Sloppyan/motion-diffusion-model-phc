"""Build stage1 train/val/test manifests from unified HumanML3D splits.

The script reads the validated HumanML3D <-> SMPL/PULSE intersection table,
assigns each valid HumanML3D ID to train/val/test, and writes compact JSONL
manifests with absolute paths for motion, text, and pulse files.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def _read_id_split(path):
    ids = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                ids.append(line)
    return ids


def _write_jsonl(path, rows):
    with open(path, 'w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare Stage1 REPA train/test manifests.')
    parser.add_argument('--manifest_csv', required=True, type=str)
    parser.add_argument('--humanml_root', required=True, type=str)
    parser.add_argument('--smpl_root', required=True, type=str)
    parser.add_argument('--train_split', required=True, type=str)
    parser.add_argument('--test_split', required=True, type=str)
    parser.add_argument('--val_split', default='', type=str)
    parser.add_argument('--out_dir', required=True, type=str)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    humanml_root = Path(args.humanml_root).resolve()
    smpl_root = Path(args.smpl_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    val_split_path = Path(args.val_split).resolve() if args.val_split else (humanml_root / 'val_uni.txt')

    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(f'output directory already exists: {out_dir}')
    out_dir.mkdir(parents=True, exist_ok=True)
    if not val_split_path.exists():
        raise FileNotFoundError(
            f'val split file not found: {val_split_path}. '
            f'Pass --val_split explicitly or generate val_uni.txt first.'
        )

    train_ids = set(_read_id_split(args.train_split))
    test_ids = set(_read_id_split(args.test_split))
    val_ids = set(_read_id_split(val_split_path))
    overlap = (train_ids & test_ids) | (train_ids & val_ids) | (test_ids & val_ids)
    if overlap:
        raise ValueError(f'train/val/test split overlap detected, example ids: {sorted(list(overlap))[:5]}')

    train_rows = []
    test_rows = []
    val_rows = []
    dataset_counter = Counter()
    skipped_counter = Counter()
    total_rows = 0

    with open(args.manifest_csv, 'r', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total_rows += 1
            humanml_id = row['humanml3d_id']
            split = None
            if humanml_id in train_ids:
                split = 'train'
            elif humanml_id in val_ids:
                split = 'val'
            elif humanml_id in test_ids:
                split = 'test'
            else:
                skipped_counter['missing_from_split'] += 1
                continue

            if row.get('is_succ', 'True') != 'True':
                skipped_counter['is_succ_false'] += 1
                continue

            motion_path = humanml_root / 'new_joint_vecs' / f'{humanml_id}.npy'
            text_path = humanml_root / 'texts' / f'{humanml_id}.txt'
            pulse_path = smpl_root / row['smpl_rel_path']

            if not motion_path.exists():
                skipped_counter['missing_motion'] += 1
                continue
            if not text_path.exists():
                skipped_counter['missing_text'] += 1
                continue
            if not pulse_path.exists():
                skipped_counter['missing_pulse'] += 1
                continue

            motion = np.load(motion_path)
            entry = {
                'id': humanml_id,
                'split': split,
                'dataset': row.get('dataset', ''),
                'source_path': row['source_path'],
                'motion_path': str(motion_path),
                'text_path': str(text_path),
                'pulse_path': str(pulse_path),
                'smpl_rel_path': row['smpl_rel_path'],
                'start_frame': int(row.get('start_frame', 0)),
                'end_frame': int(row.get('end_frame', -1)),
                'motion_length': int(len(motion)),
            }

            if split == 'train':
                train_rows.append(entry)
            elif split == 'val':
                val_rows.append(entry)
            else:
                test_rows.append(entry)
            dataset_counter[entry['dataset']] += 1

    _write_jsonl(out_dir / 'train_manifest.jsonl', train_rows)
    _write_jsonl(out_dir / 'val_manifest.jsonl', val_rows)
    _write_jsonl(out_dir / 'test_manifest.jsonl', test_rows)

    stats = {
        'total_manifest_rows': total_rows,
        'train_samples': len(train_rows),
        'val_samples': len(val_rows),
        'test_samples': len(test_rows),
        'datasets': dict(dataset_counter),
        'skipped': dict(skipped_counter),
    }
    with open(out_dir / 'stats.json', 'w', encoding='utf-8') as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)

    with open(out_dir / 'cache_version.txt', 'w', encoding='utf-8') as handle:
        handle.write('stage1_repa_cache_v1\n')

    print(f'wrote train manifest: {out_dir / "train_manifest.jsonl"} ({len(train_rows)} rows)')
    print(f'wrote val manifest: {out_dir / "val_manifest.jsonl"} ({len(val_rows)} rows)')
    print(f'wrote test manifest: {out_dir / "test_manifest.jsonl"} ({len(test_rows)} rows)')
    print(json.dumps(stats, indent=2, ensure_ascii=False))

'''
python -m data_loaders.humanml.scripts.prepare_stage1_repa_data \
  --manifest_csv /home/gxy/hay-thesis/HumanML3D/extra_scripts/generated/unified_valid/valid_shared_manifest.csv \
  --humanml_root /home/gxy/hay-thesis/HumanML3D/HumanML3D \
  --smpl_root /home/gxy/hay-thesis/SMPL-Humanoid/SMPL_Humanoid_offline_dataset/amass_state-action-pairs \
  --train_split /home/gxy/hay-thesis/HumanML3D/HumanML3D/train_uni.txt \
  --test_split /home/gxy/hay-thesis/HumanML3D/HumanML3D/test_uni.txt \
  --val_split /home/gxy/hay-thesis/HumanML3D/HumanML3D/val_uni.txt \
  --out_dir /home/gxy/hay-thesis/SMPL-Humanoid/SMPL_Humanoid_offline_dataset/train-test-split \
  --overwrite
'''
if __name__ == '__main__':
    main()
