import os
import time
from collections import defaultdict
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from mdm_core.data_loaders.get_data import get_dataset_loader
from mdm_core.data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from mdm_core.diffusion.resample import create_named_schedule_sampler
from mdm_core.eval import eval_humanact12_uestc, eval_humanml
from mdm_core.model.lora_attention import export_lora_merged_mdm_state_dict, filter_clip_from_state_dict


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


class Stage2TrainLoop:
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
        wrapped_lora_layers,
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
        self.wrapped_lora_layers = wrapped_lora_layers

        self.step = 0
        self.epoch = 0
        self.best_val_cosine = float('-inf')
        self.num_steps = int(args.num_steps)
        self.log_interval = int(args.log_interval)
        self.save_interval = int(args.save_interval)
        self.lambda_reg = float(args.lambda_reg)
        self.beta_l2 = float(args.beta_l2)
        self.hidden_layers = list(args.hidden_layers)
        self.trainable_params = [param for group in self.optimizer.param_groups for param in group['params']]
        self.dataset_stub = SimpleNamespace(dataname=getattr(self.args, 'dataset', 'humanml'))
        self._last_saved_step = -1
        self._last_val_step = -1
        self._last_generation_eval_step = -1
        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, self.diffusion)

        self.eval_wrapper = None
        self.eval_data = None
        self.eval_gt_data = None
        self._build_generation_eval()

    def load_resume_state(self, state):
        missing_keys, unexpected_keys = self.mdm_model.load_state_dict(state['mdm_model'], strict=False)
        if unexpected_keys:
            raise RuntimeError(f'unexpected stage2 model keys while resuming: {unexpected_keys}')
        invalid_missing = [key for key in missing_keys if not key.startswith('clip_model.')]
        if invalid_missing:
            raise RuntimeError(f'unexpected missing stage2 model keys while resuming: {invalid_missing}')

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
        # Log dataset statistics once, then keep stepping until
        # the requested number of optimizer updates is reached.
        # -------------------------------------------------------
        self._log_data_stats()

        while self.step < self.num_steps:
            self.epoch += 1
            self.logger.info(
                'starting epoch=%d step=%d/%d batches=%d',
                self.epoch,
                self.step,
                self.num_steps,
                len(self.train_loader),
            )
            epoch_metrics = defaultdict(float)
            epoch_batches = 0

            # -------------------------------------------------------
            # Run one full pass over the paired training loader.
            # Step-level logging/checkpointing remains step-based.
            # -------------------------------------------------------
            for batch in self.train_loader:
                metrics = self._run_train_step(batch)
                self.step += 1
                epoch_batches += 1
                for key, value in metrics.items():
                    epoch_metrics[key] += value

                if self.step == 1 or self.step % self.log_interval == 0:
                    self._log_metrics('train', metrics, self.step)

                if self.step % self.save_interval == 0:
                    val_metrics = self.evaluate_paired()
                    self._last_val_step = self.step
                    self._log_metrics('val_reg', val_metrics, self.step)
                    if val_metrics['pulse_cosine_sim'] > self.best_val_cosine:
                        self.best_val_cosine = val_metrics['pulse_cosine_sim']
                        self._save_best_checkpoint()
                    self._save_periodic_checkpoint()
                    if self.args.eval_during_training:
                        self.evaluate_generation()
                        self._last_generation_eval_step = self.step

                if self.step >= self.num_steps:
                    break

            # -------------------------------------------------------
            # Emit an epoch-level summary for readability in logs.
            # -------------------------------------------------------
            if epoch_batches > 0:
                averaged = {key: value / epoch_batches for key, value in epoch_metrics.items()}
                self.logger.info(
                    'finished epoch=%d step=%d avg_loss=%.6f avg_diff=%.6f avg_reg=%.6f avg_cos=%.6f',
                    self.epoch,
                    self.step,
                    averaged.get('loss_total', 0.0),
                    averaged.get('loss_diff', 0.0),
                    averaged.get('loss_reg', 0.0),
                    averaged.get('pulse_cosine_sim', 0.0),
                )

        # -------------------------------------------------------
        # Always leave behind a final validation + checkpoint.
        # -------------------------------------------------------
        if self._last_val_step != self.step:
            val_metrics = self.evaluate_paired()
            self._last_val_step = self.step
            self._log_metrics('val_reg', val_metrics, self.step)
            if val_metrics['pulse_cosine_sim'] > self.best_val_cosine:
                self.best_val_cosine = val_metrics['pulse_cosine_sim']
                self._save_best_checkpoint()
        if self._last_saved_step != self.step:
            self._save_periodic_checkpoint()
        if self.args.eval_during_training and self._last_generation_eval_step != self.step:
            self.evaluate_generation()
            self._last_generation_eval_step = self.step

    def _build_generation_eval(self):
        if not self.args.eval_during_training:
            return
        if getattr(self.args, 'dataset', '') not in ['kit', 'humanml']:
            return

        mm_num_samples = 0
        mm_num_repeats = 0
        gen_loader = get_dataset_loader(
            name=self.args.dataset,
            batch_size=self.args.eval_batch_size,
            num_frames=None,
            split=self.args.eval_split,
            hml_mode='eval',
        )
        self.eval_gt_data = get_dataset_loader(
            name=self.args.dataset,
            batch_size=self.args.eval_batch_size,
            num_frames=None,
            split=self.args.eval_split,
            hml_mode='gt',
        )
        self.eval_wrapper = EvaluatorMDMWrapper(self.args.dataset, self.device)
        self.eval_data = {
            'test': lambda: eval_humanml.get_mdm_loader(
                self.mdm_model,
                self.diffusion,
                self.args.eval_batch_size,
                gen_loader,
                mm_num_samples,
                mm_num_repeats,
                gen_loader.dataset.opt.max_motion_length,
                self.args.eval_num_samples,
                scale=1.0,
            )
        }

    def _log_data_stats(self):
        train_stats = getattr(self.train_loader.dataset, 'stats', {})
        val_stats = getattr(self.val_loader.dataset, 'stats', {})
        self.logger.info('train dataset stats: %s', train_stats)
        self.logger.info('val dataset stats: %s', val_stats)

    def _set_train_mode(self):
        self.mdm_model.train()
        if hasattr(self.mdm_model, 'clip_model'):
            self.mdm_model.clip_model.eval()
        self.mdm_model.rot2xyz.smpl_model.eval()
        self.fusion.eval()
        self.projector.eval()
        if self.args.train_fusion:
            self.fusion.train()

    def _set_eval_mode(self):
        self.mdm_model.eval()
        if hasattr(self.mdm_model, 'clip_model'):
            self.mdm_model.clip_model.eval()
        self.mdm_model.rot2xyz.smpl_model.eval()
        self.fusion.eval()
        self.projector.eval()

    def _adapt_batch(self, batch):
        motion = batch['motion'].to(self.device).permute(0, 2, 1).unsqueeze(2).contiguous()
        pulse_z = batch['pulse_z'].to(self.device)
        mask_bt = batch['mask'].to(self.device)
        cond = {
            'y': {
                'mask': mask_bt.unsqueeze(1).unsqueeze(1),
                'lengths': batch['lengths'].to(self.device),
                'text': batch['caption'],
                'pulse_z': pulse_z,
                'ids': batch['ids'],
            }
        }
        return motion, cond, pulse_z, mask_bt

    def _compute_reg_losses(self, pred_z, target_z, mask_bt):
        pred_norm = F.normalize(pred_z, dim=-1, eps=1e-8)
        target_norm = F.normalize(target_z, dim=-1, eps=1e-8)
        cosine_sim = (pred_norm * target_norm).sum(dim=-1)

        loss_cos = _masked_mean(1.0 - cosine_sim, mask_bt)
        loss_l2 = _masked_mean((pred_z - target_z).pow(2).mean(dim=-1), mask_bt)
        loss_reg = loss_cos + self.beta_l2 * loss_l2

        return {
            'loss_reg': loss_reg,
            'loss_cos': loss_cos,
            'loss_l2': loss_l2,
            'pulse_cosine_sim': _masked_mean(cosine_sim, mask_bt),
        }

    def _run_train_step(self, batch):
        # -------------------------------------------------------
        # Adapt the stage1 batch into the tensor layout expected
        # by MDM's diffusion training path.
        # -------------------------------------------------------
        self._set_train_mode()
        self.optimizer.zero_grad(set_to_none=True)
        motion, cond, pulse_z, mask_bt = self._adapt_batch(batch)

        # -------------------------------------------------------
        # Reuse the standard diffusion training loss, while the
        # probe captures hidden states from the same forward pass.
        # -------------------------------------------------------
        self.probe.clear()
        timesteps, weights = self.schedule_sampler.sample(motion.shape[0], self.device)

        terms = self.diffusion.training_losses(
            self.mdm_model,
            motion,
            timesteps,
            model_kwargs=cond,
            dataset=self.dataset_stub,
        )
        hidden_states = self.probe.get_hidden_states()

        # -------------------------------------------------------
        # Run the frozen bridge on the captured hidden states and
        # combine diffusion loss with the latent regularizer.
        # -------------------------------------------------------
        fused_hidden, alpha = self.fusion(hidden_states)
        pred_z = self.projector(fused_hidden)
        reg_terms = self._compute_reg_losses(pred_z=pred_z, target_z=pulse_z, mask_bt=mask_bt)
        diff_loss = (terms['loss'] * weights).mean()
        total_loss = diff_loss + self.lambda_reg * reg_terms['loss_reg']

        total_loss.backward()
        grad_norm = _compute_grad_norm(self.trainable_params)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        metrics = {
            'loss_total': total_loss.item(),
            'loss_diff': diff_loss.item(),
            'loss_reg': reg_terms['loss_reg'].item(),
            'loss_cos': reg_terms['loss_cos'].item(),
            'loss_l2': reg_terms['loss_l2'].item(),
            'pulse_cosine_sim': reg_terms['pulse_cosine_sim'].item(),
            'reg_to_diff_ratio': reg_terms['loss_reg'].item() / max(diff_loss.item(), 1e-8),
            'lr': self.optimizer.param_groups[0]['lr'],
            'grad_norm': grad_norm,
            'batch_frames': int(mask_bt.sum().item()),
            'timestep_mean': timesteps.float().mean().item(),
            'epoch': float(self.epoch),
        }
        for idx, layer_number in enumerate(self.hidden_layers):
            metrics[f'alpha_layer{layer_number}'] = alpha[idx].item()
        return metrics

    @torch.no_grad()
    def evaluate_paired(self):
        # -------------------------------------------------------
        # Run a deterministic paired validation pass to monitor
        # whether the bridge-aligned latent objective is improving.
        # -------------------------------------------------------
        self._set_eval_mode()
        totals = defaultdict(float)
        num_batches = 0
        eval_seed_base = 12345

        for batch_idx, batch in enumerate(self.val_loader):
            motion, cond, pulse_z, mask_bt = self._adapt_batch(batch)
            generator = torch.Generator(device='cpu')
            generator.manual_seed(eval_seed_base + batch_idx)
            timesteps = torch.randint(
                low=0,
                high=self.diffusion.num_timesteps,
                size=(motion.shape[0],),
                generator=generator,
                dtype=torch.long,
            ).to(self.device)
            noise = torch.randn(motion.shape, generator=generator, dtype=motion.dtype).to(self.device)

            self.probe.clear()
            terms = self.diffusion.training_losses(
                self.mdm_model,
                motion,
                timesteps,
                model_kwargs=cond,
                noise=noise,
                dataset=self.dataset_stub,
            )
            hidden_states = self.probe.get_hidden_states()
            fused_hidden, alpha = self.fusion(hidden_states)
            pred_z = self.projector(fused_hidden)
            reg_terms = self._compute_reg_losses(pred_z=pred_z, target_z=pulse_z, mask_bt=mask_bt)
            diff_loss = terms['loss'].mean()
            total_loss = diff_loss + self.lambda_reg * reg_terms['loss_reg']

            totals['loss_total'] += total_loss.item()
            totals['loss_diff'] += diff_loss.item()
            totals['loss_reg'] += reg_terms['loss_reg'].item()
            totals['loss_cos'] += reg_terms['loss_cos'].item()
            totals['loss_l2'] += reg_terms['loss_l2'].item()
            totals['pulse_cosine_sim'] += reg_terms['pulse_cosine_sim'].item()
            totals['reg_to_diff_ratio'] += reg_terms['loss_reg'].item() / max(diff_loss.item(), 1e-8)
            for idx, layer_number in enumerate(self.hidden_layers):
                totals[f'alpha_layer{layer_number}'] += alpha[idx].item()
            num_batches += 1

        if num_batches == 0:
            raise RuntimeError('validation loader is empty')

        metrics = {key: value / num_batches for key, value in totals.items()}
        metrics['best_val_cosine'] = max(self.best_val_cosine, metrics['pulse_cosine_sim'])
        self.logger.info(
            'paired validation step=%d loss=%.6f diff=%.6f reg=%.6f cosine=%.6f',
            self.step,
            metrics['loss_total'],
            metrics['loss_diff'],
            metrics['loss_reg'],
            metrics['pulse_cosine_sim'],
        )
        return metrics

    @torch.no_grad()
    def evaluate_generation(self):
        if not self.args.eval_during_training:
            return

        start_eval = time.time()
        if self.eval_wrapper is not None:
            self.logger.info('running HumanML generation evaluation at step=%d', self.step)
            log_file = os.path.join(self.args.save_dir, f'eval_humanml_{self.step:09d}.log')
            eval_dict = eval_humanml.evaluation(
                self.eval_wrapper,
                self.eval_gt_data,
                self.eval_data,
                log_file,
                replication_times=self.args.eval_rep_times,
                diversity_times=300,
                mm_num_times=0,
                run_mm=False,
            )
            for key, value in eval_dict.items():
                if key.startswith('R_precision'):
                    for idx, item in enumerate(value):
                        self.train_platform.report_scalar(
                            name=f'top{idx + 1}_{key}',
                            value=item,
                            iteration=self.step,
                            group_name='Eval',
                        )
                else:
                    self.train_platform.report_scalar(
                        name=key,
                        value=value,
                        iteration=self.step,
                        group_name='Eval',
                    )
        elif getattr(self.args, 'dataset', '') in ['humanact12', 'uestc']:
            eval_args = SimpleNamespace(
                num_seeds=self.args.eval_rep_times,
                num_samples=self.args.eval_num_samples,
                batch_size=self.args.eval_batch_size,
                device=self.device,
                guidance_param=1,
                dataset=self.args.dataset,
                unconstrained=self.args.unconstrained,
                model_path=os.path.join(self.args.save_dir, self._model_ckpt_name()),
            )
            eval_dict = eval_humanact12_uestc.evaluate(
                eval_args,
                model=self.mdm_model,
                diffusion=self.diffusion,
                data=self.train_loader.dataset,
            )
            for key, value in eval_dict['feats'].items():
                self.train_platform.report_scalar(
                    name=key,
                    value=float(torch.as_tensor(value).float().mean().item()),
                    iteration=self.step,
                    group_name='Eval',
                )

        elapsed_minutes = (time.time() - start_eval) / 60.0
        self.logger.info('generation evaluation time=%.2fmin', elapsed_minutes)

    def _log_metrics(self, split, metrics, iteration):
        summary = ', '.join(
            f'{key}={value:.6f}'
            for key, value in metrics.items()
            if key.startswith('loss') or key.endswith('cosine_sim') or key == 'reg_to_diff_ratio'
        )
        self.logger.info('%s step=%d %s', split, iteration, summary)

        for key, value in metrics.items():
            if key.startswith('alpha_layer'):
                continue
            self.train_platform.report_scalar(name=key, value=value, iteration=iteration, group_name=split)

    def _model_ckpt_name(self):
        return 'latest_model.pt'

    def _train_ckpt_name(self):
        return 'latest_train.pt'

    def _export_model_state_dict(self):
        if self.args.finetune_mode == 'last3_lora':
            return export_lora_merged_mdm_state_dict(self.mdm_model, self.wrapped_lora_layers)
        return filter_clip_from_state_dict(self.mdm_model.state_dict())

    def _save_train_state(self, path):
        state = {
            'mdm_model': filter_clip_from_state_dict(self.mdm_model.state_dict()),
            'fusion': self.fusion.state_dict(),
            'projector': self.projector.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
            'step': self.step,
            'epoch': self.epoch,
            'best_val_cosine': self.best_val_cosine,
            'args': vars(self.args),
        }
        torch.save(state, path)

    def _save_periodic_checkpoint(self):
        # -------------------------------------------------------
        # Save both the pure MDM checkpoint for inference and the
        # sidecar training state needed for stage2 resume.
        # -------------------------------------------------------
        model_path = os.path.join(self.args.save_dir, self._model_ckpt_name())
        train_path = os.path.join(self.args.save_dir, self._train_ckpt_name())
        self.logger.info('saving checkpoints: %s | %s', model_path, train_path)
        try:
            torch.save(self._export_model_state_dict(), model_path)
            self._save_train_state(train_path)
            self._last_saved_step = self.step
        except Exception:
            self.logger.exception('failed to save periodic checkpoints at step=%d', self.step)
            raise

    def _save_best_checkpoint(self):
        model_path = os.path.join(self.args.save_dir, 'best_model.pt')
        train_path = os.path.join(self.args.save_dir, 'best_train.pt')
        self.logger.info('saving best checkpoints: %s | %s', model_path, train_path)
        try:
            torch.save(self._export_model_state_dict(), model_path)
            self._save_train_state(train_path)
        except Exception:
            self.logger.exception('failed to save best checkpoints at step=%d', self.step)
            raise
