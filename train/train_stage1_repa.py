import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / 'src'
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from mdm_core.data_loaders.humanml.data.stage1_repa_dataset import Stage1RepaDataset, stage1_repa_collate
from mdm_core.model.mdm_hidden_probe import MDMHiddenProbe
from mdm_core.model.repa_projector import LayerWeightedFusion, RepaProjector
from mdm_core.train.train_platforms import (
    ClearmlPlatform,
    NoPlatform,
    TensorboardPlatform,
    WandBPlatform,
    WandbPlatform,
)
from mdm_core.train.training_loop_stage1 import Stage1TrainLoop
from mdm_core.utils.fixseed import fixseed
from mdm_core.utils.model_util import create_model_and_diffusion, load_model_wo_clip


def parse_args():
    parser = argparse.ArgumentParser(description='Stage1 REPA probing for MDM hidden states.')
    parser.add_argument('--save_dir', required=True, type=str)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--resume_checkpoint', default='', type=str)
    parser.add_argument('--wandb_run_name', required=True, type=str)
    parser.add_argument('--mdm_checkpoint', required=True, type=str)
    parser.add_argument('--train_manifest', required=True, type=str)
    parser.add_argument('--val_manifest', required=True, type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--warmup_ratio', default=0.05, type=float)
    parser.add_argument('--min_lr_ratio', default=1e-2, type=float)
    parser.add_argument('--num_epochs', '--epochs', dest='num_epochs', default=100, type=int)
    parser.add_argument('--lambda_l2', default=0.1, type=float)
    parser.add_argument('--projector_hidden_dim', default=512, type=int)
    parser.add_argument('--hidden_layers', nargs='+', default=[6, 7, 8], type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--log-every', '--log_every', dest='log_every', default=1, type=int)
    parser.add_argument('--eval-every', '--eval_every', dest='eval_every', default=1, type=int)
    parser.add_argument('--save-every', '--save_every', dest='save_every', default=10, type=int)
    parser.add_argument(
        '--train_platform_type',
        default='NoPlatform',
        choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform', 'WandbPlatform', 'WandBPlatform'],
        type=str,
    )
    parser.add_argument('--env_file', default='', type=str)
    parser.add_argument('--wandb_project', default='', type=str)
    parser.add_argument('--wandb_entity', default='', type=str)
    parser.add_argument('--wandb_mode', default='', type=str)
    parser.add_argument('--wandb_group', default='', type=str)
    parser.add_argument('--wandb_tags', default='', type=str)
    return parser.parse_args()


def setup_logger(save_dir):
    logger = logging.getLogger('stage1_repa')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler = logging.FileHandler(os.path.join(save_dir, 'train.log'))
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def resolve_device(device_arg):
    if device_arg == 'cpu':
        return torch.device('cpu')
    if device_arg.startswith('cuda') and torch.cuda.is_available():
        return torch.device(device_arg)
    if device_arg.isdigit() and torch.cuda.is_available():
        return torch.device(f'cuda:{device_arg}')
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')


def create_save_dir(args):
    save_dir = Path(args.save_dir)
    if save_dir.exists() and any(save_dir.iterdir()) and not args.overwrite and not args.resume_checkpoint:
        raise FileExistsError(f'save_dir already exists and is not empty: {save_dir}')
    save_dir.mkdir(parents=True, exist_ok=True)


def resolve_save_paths(args):
    save_root_dir = Path(args.save_dir).expanduser().resolve()
    args.save_root_dir = str(save_root_dir)
    args.save_dir = str(save_root_dir / args.wandb_run_name)


def _platform_cls(name):
    mapping = {
        'NoPlatform': NoPlatform,
        'ClearmlPlatform': ClearmlPlatform,
        'TensorboardPlatform': TensorboardPlatform,
        'WandbPlatform': WandbPlatform,
        'WandBPlatform': WandBPlatform,
    }
    return mapping[name]


def load_mdm_bundle(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path).resolve()
    args_path = checkpoint_path.parent / 'args.json'
    if not args_path.exists():
        raise FileNotFoundError(f'missing args.json next to checkpoint: {args_path}')

    with args_path.open('r', encoding='utf-8') as handle:
        checkpoint_args = argparse.Namespace(**json.load(handle))

    dummy_data = SimpleNamespace(dataset=SimpleNamespace())
    model, diffusion = create_model_and_diffusion(checkpoint_args, dummy_data)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    model.to(device)
    model.eval()
    model.rot2xyz.smpl_model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, diffusion, checkpoint_args


def build_optimizer_scheduler(args, modules, train_loader, logger):
    trainable_params = []
    for module in modules:
        trainable_params.extend(p for p in module.parameters() if p.requires_grad)

    trainable_param_count = sum(param.numel() for param in trainable_params)
    logger.info('trainable params=%d', trainable_param_count)

    optimizer = optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, args.num_epochs * steps_per_epoch)
    warmup_steps = int(args.warmup_ratio * total_steps)
    min_lr_ratio = float(args.min_lr_ratio) if args.min_lr_ratio is not None else 1e-2

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    logger.info(
        'lr schedule: base_lr=%g total_steps=%d warmup_steps=%d min_lr_ratio=%g',
        args.lr,
        total_steps,
        warmup_steps,
        min_lr_ratio,
    )
    return optimizer, scheduler, total_steps, steps_per_epoch


def save_args(args):
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w', encoding='utf-8') as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)


def main():
    # -------------------------------------------------------
    # Parse runtime args and initialize logging / device state
    # -------------------------------------------------------
    args = parse_args()
    resolve_save_paths(args)
    create_save_dir(args)
    logger = setup_logger(args.save_dir)
    fixseed(args.seed)
    device = resolve_device(args.device)

    logger.info('stage1 start save_root_dir=%s', args.save_root_dir)
    logger.info('stage1 run_name=%s', args.wandb_run_name)
    logger.info('stage1 run_dir=%s', args.save_dir)
    logger.info('using device=%s', device)
    save_args(args)

    # -------------------------------------------------------
    # Initialize reporting backend and construct train/val data
    # -------------------------------------------------------
    train_platform = _platform_cls(args.train_platform_type)(args.save_dir)
    train_platform.report_args(args, name='Args')

    train_dataset = Stage1RepaDataset(args.train_manifest, split='train', train=True)
    val_dataset = Stage1RepaDataset(args.val_manifest, split='val', train=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=stage1_repa_collate,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=stage1_repa_collate,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info('loaded manifests train=%s val=%s', args.train_manifest, args.val_manifest)
    logger.info('train samples=%d val samples=%d', len(train_dataset), len(val_dataset))

    # -------------------------------------------------------
    # Load the frozen MDM backbone and build trainable heads
    # -------------------------------------------------------
    mdm_model, diffusion, _ = load_mdm_bundle(args.mdm_checkpoint, device=device)
    logger.info(
        'loaded mdm checkpoint=%s arch=%s latent_dim=%s cond_mode=%s',
        args.mdm_checkpoint,
        mdm_model.arch,
        mdm_model.latent_dim,
        mdm_model.cond_mode,
    )

    zero_based_layers = []
    for layer in args.hidden_layers:
        if layer < 1 or layer > mdm_model.num_layers:
            raise ValueError(f'hidden layer {layer} is out of range for {mdm_model.num_layers} layers')
        zero_based_layers.append(layer - 1)

    probe = MDMHiddenProbe(mdm_model, layer_indices=zero_based_layers)
    fusion = LayerWeightedFusion(num_layers=len(args.hidden_layers)).to(device)
    projector = RepaProjector(
        hidden_dim=mdm_model.latent_dim,
        projector_hidden_dim=args.projector_hidden_dim,
        output_dim=32,
    ).to(device)
    optimizer, scheduler, total_steps, steps_per_epoch = build_optimizer_scheduler(
        args=args,
        modules=[fusion, projector],
        train_loader=train_loader,
        logger=logger,
    )
    logger.info('steps_per_epoch=%d total_steps=%d', steps_per_epoch, total_steps)

    # -------------------------------------------------------
    # Build the stage1 training loop and optionally resume state
    # -------------------------------------------------------
    loop = Stage1TrainLoop(
        args=args,
        logger=logger,
        train_platform=train_platform,
        mdm_model=mdm_model,
        diffusion=diffusion,
        probe=probe,
        fusion=fusion,
        projector=projector,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    if args.resume_checkpoint:
        logger.info('resuming stage1 checkpoint from %s', args.resume_checkpoint)
        state = torch.load(args.resume_checkpoint, map_location='cpu')
        loop.load_resume_state(state)

    try:
        loop.run_loop()
    finally:
        probe.close()
        train_platform.close()
        logger.info('stage1 finished')

'''
python -m train.train_stage1_repa \
  --save_dir /home/gxy/hay-thesis/motion-diffusion-model-phc/save/stage1_repa \
  --overwrite \
  --mdm_checkpoint /home/gxy/hay-thesis/motion-diffusion-model-phc/save/humanml_enc_512_50steps/model000750000.pt \
  --train_manifest /home/gxy/hay-thesis/SMPL-Humanoid/SMPL_Humanoid_offline_dataset/train-test-split/train_manifest.jsonl \
  --val_manifest /home/gxy/hay-thesis/SMPL-Humanoid/SMPL_Humanoid_offline_dataset/train-test-split/val_manifest.jsonl \
  --batch_size 64 \
  --num_workers 4 \
  --num_epochs 100 \
  --lr 1e-4 \
  --warmup_ratio 0.05 \
  --min_lr_ratio 0.01 \
  --lambda_l2 0.1 \
  --hidden_layers 6 7 8 \
  --log-every 1 \
  --eval-every 1 \
  --save-every 10 \
  --device cuda:0 \
  --train_platform_type WandbPlatform \
  --wandb_run_name stage1_repa_run_001 \
  --wandb_group first_stage_training
'''
if __name__ == '__main__':
    main()
