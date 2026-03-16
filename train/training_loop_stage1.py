import os
from collections import defaultdict

import torch
import torch.nn.functional as F


def _masked_mean(values, mask):
    denom = mask.float().sum().clamp_min(1.0)
    return (values * mask.float()).sum() / denom


def _compute_grad_norm(parameters):
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        grad_norm = param.grad.detach().data.norm(2).item()
        total += grad_norm * grad_norm
    return total ** 0.5


class Stage1TrainLoop:
    def __init__(
        self,
        args,
        logger,
        train_platform,
        mdm_model,
        diffusion,
        probe,
        fusion,
        projector,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        device,
    ):
        self.args = args
        self.logger = logger
        self.train_platform = train_platform
        self.mdm_model = mdm_model
        self.diffusion = diffusion
        self.probe = probe
        self.fusion = fusion
        self.projector = projector
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.step = 0
        self.epoch = 0
        self.best_val_cosine = float('-inf')
        self.log_every = args.log_every
        self.eval_every = args.eval_every
        self.save_every = args.save_every
        self.num_epochs = args.num_epochs
        self.lambda_l2 = args.lambda_l2
        self.layer_numbers = list(args.hidden_layers)
        self._last_eval_step = -1
        self._last_save_step = -1

    def load_resume_state(self, state):
        self.fusion.load_state_dict(state['fusion'])
        self.projector.load_state_dict(state['projector'])
        self.optimizer.load_state_dict(state['optimizer'])
        if self.scheduler is not None and state.get('scheduler') is not None:
            self.scheduler.load_state_dict(state['scheduler'])
        self.step = int(state.get('step', 0))
        self.epoch = int(state.get('epoch', 0))
        self.best_val_cosine = float(state.get('best_val_cosine', float('-inf')))

    def run_loop(self):
        # -------------------------------------------------------
        # Log dataset statistics once, then iterate epoch by epoch
        # -------------------------------------------------------
        self._log_data_stats()

        while self.epoch < self.num_epochs:
            self.epoch += 1
            self.logger.info(
                'starting epoch=%d/%d step=%d batches=%d',
                self.epoch,
                self.num_epochs,
                self.step,
                len(self.train_loader),
            )
            epoch_metrics = defaultdict(float)
            epoch_batches = 0

            # -------------------------------------------------------
            # Run all optimization steps for the current epoch
            # -------------------------------------------------------
            for batch in self.train_loader:
                metrics = self._run_train_step(batch)
                current_step = self.step + 1
                epoch_batches += 1
                for key, value in metrics.items():
                    epoch_metrics[key] += value

                self.step = current_step

            # -------------------------------------------------------
            # Emit epoch-level train metrics after averaging batches
            # -------------------------------------------------------
            if epoch_batches > 0:
                averaged = {key: value / epoch_batches for key, value in epoch_metrics.items()}
                self.logger.info(
                    'finished epoch=%d step=%d avg_loss=%.6f avg_cos=%.6f',
                    self.epoch,
                    self.step,
                    averaged.get('loss_total', 0.0),
                    averaged.get('cosine_sim', 0.0),
                )
                averaged['epoch'] = float(self.epoch)
                averaged['global_step'] = float(self.step)
                if self._should_log_epoch(self.epoch):
                    self._log_metrics('train', averaged, self.epoch)

            # -------------------------------------------------------
            # Run validation / checkpointing on epoch boundaries
            # -------------------------------------------------------
            if self._should_eval_epoch(self.epoch):
                val_metrics = self.evaluate()
                self._last_eval_step = self.step
                if val_metrics['cosine_sim'] > self.best_val_cosine:
                    self.best_val_cosine = val_metrics['cosine_sim']
                    self._safe_save_checkpoint('best.pt')
                val_metrics['epoch'] = float(self.epoch)
                val_metrics['global_step'] = float(self.step)
                self._log_metrics('val', val_metrics, self.epoch)

            if self._should_save_epoch(self.epoch):
                self._safe_save_checkpoint('latest.pt')
                self._last_save_step = self.step

        # -------------------------------------------------------
        # Ensure the final epoch leaves behind a fresh eval/save
        # -------------------------------------------------------
        if self._last_eval_step != self.step:
            val_metrics = self.evaluate()
            self._last_eval_step = self.step
            if val_metrics['cosine_sim'] > self.best_val_cosine:
                self.best_val_cosine = val_metrics['cosine_sim']
                self._safe_save_checkpoint('best.pt')
            val_metrics['epoch'] = float(self.epoch)
            val_metrics['global_step'] = float(self.step)
            self._log_metrics('val', val_metrics, self.epoch)
        if self._last_save_step != self.step:
            self._safe_save_checkpoint('latest.pt')
            self._last_save_step = self.step

    def _log_data_stats(self):
        train_stats = getattr(self.train_loader.dataset, 'stats', {})
        val_stats = getattr(self.val_loader.dataset, 'stats', {})
        self.logger.info('train dataset stats: %s', train_stats)
        self.logger.info('val dataset stats: %s', val_stats)

        for name, value in train_stats.items():
            if name == 'split':
                continue
            self.train_platform.report_scalar(name=name, value=value, iteration=0, group_name='data/train')
        for name, value in val_stats.items():
            if name == 'split':
                continue
            self.train_platform.report_scalar(name=name, value=value, iteration=0, group_name='data/val')

    def _should_log_epoch(self, epoch):
        return epoch == 1 or (epoch % self.log_every == 0)

    def _should_eval_epoch(self, epoch):
        return epoch % self.eval_every == 0

    def _should_save_epoch(self, epoch):
        return epoch % self.save_every == 0

    def _build_condition(self, captions):
        cond_mode = getattr(self.mdm_model, 'cond_mode', 'no_cond')
        if cond_mode == 'no_cond':
            return {}
        if 'text' in cond_mode:
            return {'text': captions}
        raise NotImplementedError(f'unsupported cond_mode for stage1: {cond_mode}')

    def _prepare_motion(self, motion):
        # [B, T, 263] -> [B, 263, 1, T]
        return motion.permute(0, 2, 1).unsqueeze(2).contiguous()

    def _compute_losses(self, pred_z, target_z, mask):
        pred_norm = F.normalize(pred_z, dim=-1, eps=1e-8)
        target_norm = F.normalize(target_z, dim=-1, eps=1e-8)
        cosine_sim = (pred_norm * target_norm).sum(dim=-1)

        loss_cos = _masked_mean(1.0 - cosine_sim, mask)
        loss_l2 = _masked_mean((pred_z - target_z).pow(2).mean(dim=-1), mask)
        loss_total = loss_cos + self.lambda_l2 * loss_l2

        return {
            'loss_total': loss_total,
            'loss_cos': loss_cos,
            'loss_l2': loss_l2,
            'cosine_sim': _masked_mean(cosine_sim, mask),
        }

    def _run_train_step(self, batch):
        # -------------------------------------------------------
        # Prepare trainable heads and move the current batch to device
        # -------------------------------------------------------
        self.fusion.train()
        self.projector.train()
        self.optimizer.zero_grad(set_to_none=True)

        motion = batch['motion'].to(self.device)
        pulse_z = batch['pulse_z'].to(self.device)
        mask = batch['mask'].to(self.device)
        captions = batch['caption']

        # -------------------------------------------------------
        # Build a noisy diffusion input from the clean motion sequence
        # -------------------------------------------------------
        x_start = self._prepare_motion(motion)
        timesteps = torch.randint(
            low=0,
            high=self.diffusion.num_timesteps,
            size=(x_start.shape[0],),
            device=self.device,
            dtype=torch.long,
        )
        noise = torch.randn_like(x_start)
        x_t = self.diffusion.q_sample(x_start=x_start, t=timesteps, noise=noise)
        cond = self._build_condition(captions)

        # -------------------------------------------------------
        # Extract frozen MDM features, fuse layers, and predict pulse_z
        # -------------------------------------------------------
        hidden_states = self.probe.extract(x_t, timesteps, cond)
        fused_hidden, alpha = self.fusion(hidden_states)
        pred_z = self.projector(fused_hidden)

        # -------------------------------------------------------
        # Backpropagate through fusion/projector and update scheduler
        # -------------------------------------------------------
        losses = self._compute_losses(pred_z=pred_z, target_z=pulse_z, mask=mask)
        losses['loss_total'].backward()
        grad_norm = _compute_grad_norm(list(self.fusion.parameters()) + list(self.projector.parameters()))
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        metrics = {
            'loss_total': losses['loss_total'].item(),
            'loss_cos': losses['loss_cos'].item(),
            'loss_l2': losses['loss_l2'].item(),
            'cosine_sim': losses['cosine_sim'].item(),
            'lr': self.optimizer.param_groups[0]['lr'],
            'grad_norm': grad_norm,
            'batch_frames': int(mask.sum().item()),
            'timestep_mean': timesteps.float().mean().item(),
            'epoch': self.epoch,
        }
        for idx, layer_number in enumerate(self.layer_numbers):
            metrics[f'alpha_layer{layer_number}'] = alpha[idx].item()
        return metrics

    @torch.no_grad()
    def evaluate(self):
        # -------------------------------------------------------
        # Switch heads to eval mode and run deterministic validation
        # -------------------------------------------------------
        self.fusion.eval()
        self.projector.eval()

        totals = defaultdict(float)
        num_batches = 0
        eval_seed_base = 12345
        for batch_idx, batch in enumerate(self.val_loader):
            motion = batch['motion'].to(self.device)
            pulse_z = batch['pulse_z'].to(self.device)
            mask = batch['mask'].to(self.device)
            captions = batch['caption']

            # -------------------------------------------------------
            # Recreate a fixed noisy input so validation stays comparable
            # -------------------------------------------------------
            x_start = self._prepare_motion(motion)
            generator = torch.Generator(device='cpu')
            generator.manual_seed(eval_seed_base + batch_idx)
            timesteps = torch.randint(
                low=0,
                high=self.diffusion.num_timesteps,
                size=(x_start.shape[0],),
                dtype=torch.long,
                generator=generator,
            ).to(self.device)
            noise = torch.randn(x_start.shape, generator=generator, dtype=x_start.dtype).to(self.device)
            x_t = self.diffusion.q_sample(x_start=x_start, t=timesteps, noise=noise)
            cond = self._build_condition(captions)

            # -------------------------------------------------------
            # Reuse the same feature-extract / predict / loss pipeline
            # -------------------------------------------------------
            hidden_states = self.probe.extract(x_t, timesteps, cond)
            fused_hidden, alpha = self.fusion(hidden_states)
            pred_z = self.projector(fused_hidden)
            losses = self._compute_losses(pred_z=pred_z, target_z=pulse_z, mask=mask)

            totals['loss_total'] += losses['loss_total'].item()
            totals['loss_cos'] += losses['loss_cos'].item()
            totals['loss_l2'] += losses['loss_l2'].item()
            totals['cosine_sim'] += losses['cosine_sim'].item()
            for idx, layer_number in enumerate(self.layer_numbers):
                totals[f'alpha_layer{layer_number}'] += alpha[idx].item()
            num_batches += 1

        if num_batches == 0:
            raise RuntimeError('validation loader is empty')

        metrics = {key: value / num_batches for key, value in totals.items()}
        metrics['best_val_cosine'] = max(self.best_val_cosine, metrics['cosine_sim'])
        self.logger.info(
            'validation step=%d loss=%.6f cosine=%.6f',
            self.step,
            metrics['loss_total'],
            metrics['cosine_sim'],
        )
        return metrics

    def _log_metrics(self, split, metrics, iteration):
        summary = ', '.join(
            f'{key}={value:.6f}' for key, value in metrics.items()
            if key.startswith('loss') or key.endswith('cosine_sim') or key == 'cosine_sim'
        )
        self.logger.info('%s epoch=%d %s', split, iteration, summary)

        for key, value in metrics.items():
            if key.startswith('alpha_layer'):
                self.train_platform.report_scalar(
                    name=key,
                    value=value,
                    iteration=iteration,
                    group_name='model',
                )
                continue
            self.train_platform.report_scalar(
                name=key,
                value=value,
                iteration=iteration,
                group_name=split,
            )

    def save_checkpoint(self, filename):
        checkpoint = {
            'fusion': self.fusion.state_dict(),
            'projector': self.projector.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
            'step': self.step,
            'epoch': self.epoch,
            'best_val_cosine': self.best_val_cosine,
            'mdm_checkpoint': self.args.mdm_checkpoint,
            'args': vars(self.args),
        }
        path = os.path.join(self.args.save_dir, filename)
        torch.save(checkpoint, path)
        self.logger.info('saved checkpoint: %s', path)

    def _safe_save_checkpoint(self, filename):
        path = os.path.join(self.args.save_dir, filename)
        self.logger.info('saving checkpoint: %s', path)
        try:
            self.save_checkpoint(filename)
        except Exception:
            self.logger.exception('failed to save checkpoint: %s', path)
            raise
