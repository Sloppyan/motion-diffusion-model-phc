import argparse
import json
import logging
import math
import os
import shutil
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
from mdm_core.model.lora_attention import apply_lora_to_last_layers
from mdm_core.model.mdm_hidden_probe import MDMHiddenProbe
from mdm_core.model.repa_projector import LayerWeightedFusion, RepaProjector
from mdm_core.train.train_platforms import (
    ClearmlPlatform,
    NoPlatform,
    TensorboardPlatform,
    WandBPlatform,
    WandbPlatform,
)
from mdm_core.train.training_loop_stage2 import Stage2TrainLoop
from mdm_core.utils.fixseed import fixseed
from mdm_core.utils.model_util import create_model_and_diffusion, load_model_wo_clip


def parse_args():
    parser = argparse.ArgumentParser(description='Stage2 REPA LoRA finetuning for MDM.')
    parser.add_argument('--save_dir', required=True, type=str)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--resume_checkpoint', default='', type=str)
    parser.add_argument('--wandb_run_name', required=True, type=str)
    parser.add_argument('--mdm_checkpoint', required=True, type=str)
    parser.add_argument('--stage1_bridge_checkpoint', required=True, type=str)
    parser.add_argument('--train_manifest', required=True, type=str)
    parser.add_argument('--val_manifest', required=True, type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--warmup_ratio', default=0.01, type=float)
    parser.add_argument('--min_lr_ratio', default=1e-2, type=float)
    parser.add_argument('--num_steps', default=100000, type=int)
    parser.add_argument('--log_interval', default=1000, type=int)
    parser.add_argument('--save_interval', default=10000, type=int)
    parser.add_argument('--lambda_reg', default=0.05, type=float)
    parser.add_argument('--beta_l2', default=0.05, type=float)
    parser.add_argument('--hidden_layers', nargs='+', default=None, type=int)
    parser.add_argument('--finetune_mode', default='last3_lora', choices=['last3_lora', 'last3_full'], type=str)
    parser.add_argument('--lora_rank', default=8, type=int)
    parser.add_argument('--lora_alpha', default=16, type=int)
    parser.add_argument('--lora_dropout', default=0.05, type=float)
    parser.add_argument('--train_fusion', action='store_true')
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--eval_during_training', action='store_true')
    parser.add_argument('--eval_batch_size', default=32, type=int)
    parser.add_argument('--eval_split', default='test', choices=['val', 'test'], type=str)
    parser.add_argument('--eval_rep_times', default=3, type=int)
    parser.add_argument('--eval_num_samples', default=1000, type=int)
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
    logger = logging.getLogger('stage2_repa')
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
    model.rot2xyz.smpl_model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, diffusion, checkpoint_args


def load_stage1_bridge(stage1_checkpoint_path, hidden_dim, device):
    checkpoint = torch.load(stage1_checkpoint_path, map_location='cpu')
    stage1_args = checkpoint.get('args', {})
    hidden_layers = list(stage1_args.get('hidden_layers', [6, 7, 8]))
    projector_hidden_dim = int(stage1_args.get('projector_hidden_dim', 512))
    output_dim = int(checkpoint['projector']['net.4.weight'].shape[0])

    fusion = LayerWeightedFusion(num_layers=len(hidden_layers)).to(device)
    projector = RepaProjector(
        hidden_dim=hidden_dim,
        projector_hidden_dim=projector_hidden_dim,
        output_dim=output_dim,
    ).to(device)

    fusion.load_state_dict(checkpoint['fusion'])
    projector.load_state_dict(checkpoint['projector'])
    for param in fusion.parameters():
        param.requires_grad = False
    for param in projector.parameters():
        param.requires_grad = False
    fusion.eval()
    projector.eval()
    return fusion, projector, hidden_layers, stage1_args


def resolve_hidden_layers(args, stage1_hidden_layers, num_layers):
    resolved_layers = list(stage1_hidden_layers)
    if args.hidden_layers is not None:
        if list(args.hidden_layers) != resolved_layers:
            raise ValueError(
                f'--hidden_layers {args.hidden_layers} must match stage1 checkpoint hidden layers {resolved_layers}'
            )
    for layer in resolved_layers:
        if layer < 1 or layer > num_layers:
            raise ValueError(f'hidden layer {layer} is out of range for {num_layers} layers')
    args.hidden_layers = resolved_layers


def configure_trainable_modules(args, logger, mdm_model, fusion):
    zero_based_layers = [layer - 1 for layer in args.hidden_layers]
    wrapped_lora_layers = {}

    if args.finetune_mode == 'last3_lora':
        wrapped_lora_layers = apply_lora_to_last_layers(
            model=mdm_model,
            layer_indices=zero_based_layers,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
        logger.info(
            'applied LoRA to layers=%s rank=%d alpha=%d dropout=%g',
            args.hidden_layers,
            args.lora_rank,
            args.lora_alpha,
            args.lora_dropout,
        )
    else:
        for layer_idx in zero_based_layers:
            for param in mdm_model.seqTransEncoder.layers[layer_idx].parameters():
                param.requires_grad = True
        logger.info('unfroze full transformer layers=%s', args.hidden_layers)

    if args.train_fusion:
        for param in fusion.parameters():
            param.requires_grad = True
        logger.info('train_fusion=True, fusion weights are trainable')

    return wrapped_lora_layers, zero_based_layers


def build_optimizer_scheduler(args, modules, logger):
    trainable_params = []
    for module in modules:
        trainable_params.extend(param for param in module.parameters() if param.requires_grad)

    if not trainable_params:
        raise ValueError('no trainable parameters were found for stage2 optimization')

    trainable_param_count = sum(param.numel() for param in trainable_params)
    logger.info('trainable params=%d', trainable_param_count)

    optimizer = optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=args.weight_decay,
    )

    total_steps = max(1, int(args.num_steps))
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
    return optimizer, scheduler


def save_stage2_train_args(args):
    args_path = os.path.join(args.save_dir, 'stage2_train_args.json')
    with open(args_path, 'w', encoding='utf-8') as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)


def copy_base_inference_args(mdm_checkpoint, save_dir):
    source_args = Path(mdm_checkpoint).resolve().parent / 'args.json'
    target_args = Path(save_dir) / 'args.json'
    shutil.copyfile(source_args, target_args)


def main():
    # -------------------------------------------------------
    # Parse runtime args, create the run directory, and resolve
    # all static config needed before model construction.
    # -------------------------------------------------------
    args = parse_args()
    resolve_save_paths(args)
    create_save_dir(args)
    logger = setup_logger(args.save_dir)
    fixseed(args.seed)
    device = resolve_device(args.device)

    args.lora_target_modules = ['q_proj', 'k_proj', 'v_proj', 'out_proj']
    logger.info('stage2 start save_root_dir=%s', args.save_root_dir)
    logger.info('stage2 run_name=%s', args.wandb_run_name)
    logger.info('stage2 run_dir=%s', args.save_dir)
    logger.info('using device=%s', device)

    # -------------------------------------------------------
    # Initialize the reporting backend and construct the paired
    # stage1 data loaders that stage2 will continue to reuse.
    # -------------------------------------------------------
    train_platform = _platform_cls(args.train_platform_type)(args.save_dir)

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
    # Load the base MDM checkpoint, restore the frozen stage1
    # bridge, then inject LoRA into the requested attention layers.
    # -------------------------------------------------------
    mdm_model, diffusion, checkpoint_args = load_mdm_bundle(args.mdm_checkpoint, device=device)
    fusion, projector, stage1_hidden_layers, _ = load_stage1_bridge(
        stage1_checkpoint_path=args.stage1_bridge_checkpoint,
        hidden_dim=mdm_model.latent_dim,
        device=device,
    )
    resolve_hidden_layers(args, stage1_hidden_layers, mdm_model.num_layers)
    wrapped_lora_layers, zero_based_layers = configure_trainable_modules(
        args=args,
        logger=logger,
        mdm_model=mdm_model,
        fusion=fusion,
    )
    probe = MDMHiddenProbe(mdm_model, layer_indices=zero_based_layers)

    logger.info(
        'loaded mdm checkpoint=%s arch=%s latent_dim=%s cond_mode=%s dataset=%s',
        args.mdm_checkpoint,
        mdm_model.arch,
        mdm_model.latent_dim,
        mdm_model.cond_mode,
        checkpoint_args.dataset,
    )
    logger.info('loaded stage1 bridge checkpoint=%s', args.stage1_bridge_checkpoint)

    args.dataset = checkpoint_args.dataset
    args.unconstrained = getattr(checkpoint_args, 'unconstrained', False)
    save_stage2_train_args(args)
    copy_base_inference_args(args.mdm_checkpoint, args.save_dir)
    train_platform.report_args(args, name='Args')

    optimizer, scheduler = build_optimizer_scheduler(args=args, modules=[mdm_model, fusion], logger=logger)

    # -------------------------------------------------------
    # Build the stage2 training loop and optionally restore a
    # previous LoRA training state before stepping further.
    # -------------------------------------------------------
    loop = Stage2TrainLoop(
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
        wrapped_lora_layers=wrapped_lora_layers,
    )

    if args.resume_checkpoint:
        logger.info('resuming stage2 checkpoint from %s', args.resume_checkpoint)
        state = torch.load(args.resume_checkpoint, map_location='cpu')
        loop.load_resume_state(state)

    try:
        loop.run_loop()
    finally:
        probe.close()
        train_platform.close()
        logger.info('stage2 finished')


'''
python -m train.train_stage2_repa \
  --save_dir /home/gxy/hay-thesis/motion-diffusion-model-phc/save/stage2_repa \
  --overwrite \
  --wandb_run_name stage2_repa_run_002 \
  --mdm_checkpoint /home/gxy/hay-thesis/motion-diffusion-model-phc/save/humanml_enc_512_50steps/model000750000.pt \
  --stage1_bridge_checkpoint /home/gxy/hay-thesis/motion-diffusion-model-phc/save/stage1_repa/stage1_repa_run_001/best.pt \
  --train_manifest /home/gxy/hay-thesis/SMPL-Humanoid/SMPL_Humanoid_offline_dataset/train-test-split/train_manifest.jsonl \
  --val_manifest /home/gxy/hay-thesis/SMPL-Humanoid/SMPL_Humanoid_offline_dataset/train-test-split/val_manifest.jsonl \
  --hidden_layers 6 7 8 \
  --finetune_mode last3_lora \
  --lora_rank 16 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lambda_reg 0.2 \
  --beta_l2 0.05 \
  --lr 2e-5 \
  --weight_decay 0.0 \
  --batch_size 64 \
  --num_steps 100000 \
  --log_interval 1000 \
  --save_interval 2000 \
  --device cuda:0 \
  --train_platform_type WandbPlatform \
  --wandb_group second_stage_training
'''


if __name__ == '__main__':
    main()
