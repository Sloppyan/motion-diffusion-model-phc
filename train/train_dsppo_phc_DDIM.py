import argparse
import json
from dataclasses import dataclass
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Sequence, Tuple

try:  # Isaac Gym must be imported before torch in this process.
    import isaacgym  # noqa: F401
except ImportError:
    isaacgym = None

import numpy as np
import torch
from torch.optim import AdamW

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.dppo_frame_rl.data import (
    build_prompt_entries,
    build_prompt_entry_pools,
    sample_prompt_batch,
    select_eval_batch,
)
from train.dppo_frame_rl.logging import merge_metrics
from train.dppo_frame_rl.models.frame_critic import FrameCritic
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from train.dsppo_rl import FrameNoisePolicy
from train.dsppo_rl.ppo import compute_noise_policy_loss
from train.dsppo_rl.runtime import DSPPORollout, DSPPORuntime
from train.train_dsppo_phc import (
    maybe_init_wandb,
    prefix_metrics,
    resolve_scheduled_lr,
    save_checkpoint,
    select_best_metric,
    set_optimizer_lr,
    set_seed,
    train_critic,
    iter_sample_minibatches,
)
from utils.model_util import create_model_and_diffusion, load_model_wo_clip


def _create_respaced_ddim_diffusion(model_args, ddim_steps: Optional[int]):
    ##############################################
    # Build a respaced diffusion over the original training
    # timeline, while keeping the MDM weights unchanged.
    ##############################################
    base_steps = int(model_args.diffusion_steps)
    target_steps = base_steps if ddim_steps is None else int(ddim_steps)
    if target_steps < 1 or target_steps > base_steps:
        raise ValueError(f"--ddim_steps must be in [1, {base_steps}], got {target_steps}.")

    betas = gd.get_named_beta_schedule(model_args.noise_schedule, base_steps, 1.0)
    use_timesteps = space_timesteps(base_steps, [target_steps])
    return SpacedDiffusion(
        use_timesteps=use_timesteps,
        betas=betas,
        model_mean_type=gd.ModelMeanType.START_X,
        model_var_type=gd.ModelVarType.FIXED_SMALL if model_args.sigma_small else gd.ModelVarType.FIXED_LARGE,
        loss_type=gd.LossType.MSE,
        rescale_timesteps=False,
        lambda_vel=model_args.lambda_vel,
        lambda_rcxyz=model_args.lambda_rcxyz,
        lambda_fc=model_args.lambda_fc,
    )


class DSPPODDIMRuntime(DSPPORuntime):
    @dataclass
    class Rollout(DSPPORollout):
        frame_noise_gt_inv: torch.Tensor

    def _build_actor_and_diffusion(self, model_path: str):
        model_path = Path(model_path).expanduser().resolve()
        args_path = model_path.parent / "args.json"
        if not args_path.is_file():
            raise FileNotFoundError(f"MDM args.json not found: {args_path}")
        with args_path.open("r", encoding="utf-8") as handle:
            model_args = json.load(handle)
        arg_namespace = SimpleNamespace(**model_args)
        dummy_data = SimpleNamespace(dataset=SimpleNamespace(num_actions=1))

        actor, _ = create_model_and_diffusion(arg_namespace, dummy_data)
        diffusion = _create_respaced_ddim_diffusion(
            arg_namespace,
            getattr(self.args, "ddim_steps", None),
        )
        state_dict = torch.load(model_path, map_location="cpu")
        load_model_wo_clip(actor, state_dict)
        actor.to(self.device)
        actor.eval()
        for parameter in actor.parameters():
            parameter.requires_grad = False
        print(
            f"[dsppo-ddim] base_diffusion_steps={arg_namespace.diffusion_steps} "
            f"effective_ddim_steps={diffusion.num_timesteps}"
        )
        return actor, diffusion

    def __init__(self, args):
        super().__init__(args)
        self.ddim_eta = float(args.ddim_eta)
        if self.ddim_eta < 0.0:
            raise ValueError("--ddim_eta must be non-negative.")
        if self.stochastic_first_k_steps > 0:
            raise ValueError("DDIM runtime does not support stochastic_first_k_steps.")
        self._gt_inversion_cache: Dict[Tuple[str, int, int, int], torch.Tensor] = {}

    def _sample_clean_motion(
        self,
        x_t: torch.Tensor,
        model_kwargs: Dict,
        deterministic_denoising: bool,
        reverse_noise_seed: Optional[int] = None,
    ) -> torch.Tensor:
        ##############################################
        # Replace the DDPM reverse chain with DDIM while
        # keeping x_T, PPO, reward, and evaluation intact.
        # The original deterministic_denoising flag remains
        # meaningful by forcing the DDIM path to eta = 0.
        ##############################################
        eta = 0.0 if deterministic_denoising else self.ddim_eta
        devices = [] if self.device.index is None else [self.device.index]

        def _run_chain() -> torch.Tensor:
            with torch.no_grad():
                return self.diffusion.ddim_sample_loop(
                    self.sampling_actor,
                    shape=tuple(x_t.shape),
                    noise=x_t,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    device=self.device,
                    progress=False,
                    eta=eta,
                ).detach()

        if reverse_noise_seed is None:
            return _run_chain()
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(reverse_noise_seed))
            return _run_chain()

    def _gt_cache_key(self, entry) -> Tuple[str, int, int, int]:
        return (entry.db_key, int(entry.caption_idx), int(entry.start_20fps), int(entry.length_20fps))

    def _build_gt_x0_batch(self, entries: Sequence) -> torch.Tensor:
        feature_dim = self.actor.njoints * self.actor.nfeats
        gt_norm = torch.zeros(
            (len(entries), self.max_motion_frames, feature_dim),
            dtype=torch.float32,
            device=self.device,
        )
        mean = self._mean.view(1, 1, -1)
        std = self._std.view(1, 1, -1)

        ##############################################
        # Build normalized x_0 in the exact feature space
        # used by the diffusion model before DDIM inversion.
        ##############################################
        for sample_idx, entry in enumerate(entries):
            motion = self._load_motion_vec(entry.db_key)
            start = int(max(0, min(entry.start_20fps, motion.shape[0] - 1)))
            end = int(min(motion.shape[0], start + entry.length_20fps))
            segment = np.asarray(motion[start:end], dtype=np.float32).copy()
            seg_len = min(int(segment.shape[0]), self.max_motion_frames)
            if seg_len > 0:
                gt_norm[sample_idx, :seg_len] = torch.from_numpy(segment[:seg_len]).to(self.device)

        gt_norm = (gt_norm - mean) / std
        return gt_norm.permute(0, 2, 1).unsqueeze(2).contiguous()

    def _invert_gt_batch(self, entries: Sequence) -> torch.Tensor:
        texts = [entry.caption for entry in entries]
        lengths_20fps = torch.tensor([entry.length_20fps for entry in entries], dtype=torch.long, device=self.device)
        model_kwargs = self._build_model_kwargs(
            texts=texts,
            lengths_20fps=lengths_20fps,
            max_motion_frames=self.max_motion_frames,
        )
        sample = self._build_gt_x0_batch(entries)
        with torch.no_grad():
            ##############################################
            # Starting from x_0^{gt}, follow the same DDIM
            # reverse-ODE schedule to obtain the target x_T.
            ##############################################
            for step in range(int(self.diffusion.num_timesteps)):
                t = torch.full((sample.shape[0],), step, dtype=torch.long, device=self.device)
                out = self.diffusion.ddim_reverse_sample(
                    self.sampling_actor,
                    sample,
                    t,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    eta=self.ddim_eta,
                )
                sample = out["sample"]
        return sample.squeeze(2).permute(0, 2, 1).contiguous().detach()

    def _load_gt_inversion_noise_batch(self, entries: Sequence) -> torch.Tensor:
        feature_dim = self.actor.njoints * self.actor.nfeats
        batch = torch.zeros(
            (len(entries), self.max_motion_frames, feature_dim),
            dtype=torch.float32,
            device=self.device,
        )

        missing_entries = []
        missing_indices = []
        for sample_idx, entry in enumerate(entries):
            cached = self._gt_inversion_cache.get(self._gt_cache_key(entry))
            if cached is None:
                missing_entries.append(entry)
                missing_indices.append(sample_idx)
            else:
                batch[sample_idx] = cached.to(self.device)

        if missing_entries:
            inverted = self._invert_gt_batch(missing_entries)
            for local_idx, sample_idx in enumerate(missing_indices):
                cached = inverted[local_idx].detach().cpu()
                self._gt_inversion_cache[self._gt_cache_key(entries[sample_idx])] = cached
                batch[sample_idx] = cached.to(self.device)

        return batch

    def collect_rollout(
        self,
        entries: Sequence,
        noise_policy,
        critic,
        policy_deterministic: bool = False,
        sampling_seed: Optional[int] = None,
    ):
        rollout, metrics = super().collect_rollout(
            entries=entries,
            noise_policy=noise_policy,
            critic=critic,
            policy_deterministic=policy_deterministic,
            sampling_seed=sampling_seed,
        )
        frame_noise_gt_inv = self._load_gt_inversion_noise_batch(entries)
        rollout = self.Rollout(**rollout.__dict__, frame_noise_gt_inv=frame_noise_gt_inv)
        return rollout, metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--data_root", required=True, type=str)
    parser.add_argument("--run_name", required=True, type=str)

    parser.add_argument("--device_id", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--max_motion_frames", default=196, type=int)
    parser.add_argument("--prompt_batch_size", default=16, type=int)
    parser.add_argument("--ppo_minibatch_size", default=8, type=int)
    parser.add_argument("--critic_minibatch_size", default=16, type=int)
    parser.add_argument("--num_outer_steps", default=100, type=int)
    parser.add_argument("--actor_num_epochs", default=1, type=int)
    parser.add_argument("--critic_num_epochs", default=2, type=int)
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("--critic_lr", default=3e-4, type=float)
    parser.add_argument("--lr_schedule", default="constant", choices=["constant", "cosine"], type=str)
    parser.add_argument("--min_lr", default=0.0, type=float)
    parser.add_argument("--critic_lr_schedule", default="constant", choices=["constant", "cosine"], type=str)
    parser.add_argument("--critic_min_lr", default=0.0, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--clip_range", default=1e-2, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)

    parser.add_argument("--guidance_param", default=2.5, type=float)
    parser.add_argument("--noise_hidden_dim", default=256, type=int)
    parser.add_argument("--noise_prior_kl_coef", default=0.0, type=float)
    parser.add_argument("--inversion_loss_coef", default=0.0, type=float)
    parser.add_argument("--entropy_coef", default=0.0, type=float)
    parser.add_argument("--noise_log_std_init", default=0.0, type=float)
    parser.add_argument("--ddim_eta", default=0.0, type=float)
    parser.add_argument("--ddim_steps", default=None, type=int)
    parser.add_argument("--deterministic_denoising", action="store_true")
    parser.add_argument("--deterministic_eval_policy", action="store_true")

    parser.add_argument("--frame_gamma", default=0.995, type=float)
    parser.add_argument("--frame_lambda", default=0.95, type=float)
    parser.add_argument("--adv_norm_std_only", action="store_true")
    parser.add_argument("--adv_norm_epsilon", default=1e-8, type=float)
    parser.add_argument("--dense_reward_weight", default=1.0, type=float)
    parser.set_defaults(reward_norm=False)
    parser.add_argument("--reward_norm", dest="reward_norm", action="store_true")
    parser.add_argument("--pose_reward_weight", default=0.0, type=float)
    parser.add_argument("--pose_reward_alpha", default=1.0, type=float)
    parser.add_argument("--velocity_reward_weight", default=0.0, type=float)
    parser.add_argument("--velocity_reward_alpha", default=1.0, type=float)
    parser.add_argument("--success_bonus", default=0.0, type=float)
    parser.add_argument("--fail_penalty", default=-5.0, type=float)
    parser.add_argument("--value_target_norm", default="none", choices=["none", "popart"], type=str)
    parser.add_argument("--popart_beta", default=5e-4, type=float)
    parser.add_argument("--popart_epsilon", default=1e-5, type=float)

    parser.add_argument("--train_split", default="train", type=str)
    parser.add_argument("--eval_split", default="test", type=str)
    parser.add_argument("--train_sampling_mode", choices=["uniform", "failure_only", "mixed"], default="uniform")
    parser.add_argument("--failure_cases_file", default="", type=str)
    parser.add_argument("--failure_eval_cases_file", default="", type=str)
    parser.add_argument("--success_sampling_weight", default=1.0, type=float)
    parser.add_argument("--failure_sampling_weight", default=1.0, type=float)

    parser.add_argument("--phc_num_envs", default=16, type=int)
    parser.add_argument("--phc_max_steps", default=420, type=int)
    parser.add_argument(
        "--phc_actor_ckpt",
        default="/home/gxy/hay-thesis/PHC/output/HumanoidIm/phc_kp_mcp_iccv/Humanoid.pth",
        type=str,
    )
    parser.add_argument("--eval_interval", default=5, type=int)
    parser.add_argument("--eval_prompt_batch_size", default=4, type=int)
    parser.add_argument("--fixed_sampling_seed", action="store_true")

    parser.add_argument("--save_root", default="save", type=str)
    parser.add_argument("--save_interval", default=10, type=int)
    parser.add_argument("--log_interval", default=1, type=int)
    parser.add_argument("--wandb_mode", default="disabled", choices=["disabled", "offline", "online"], type=str)
    parser.add_argument("--wandb_project", default="mdm-phc-dsppo-ddim", type=str)
    return parser.parse_args()


def normalize_advantages(
    advantages: torch.Tensor,
    mask: torch.Tensor,
    std_only: bool = False,
    epsilon: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    valid = advantages[mask]
    if valid.numel() == 0:
        return {
            "advantages": torch.zeros_like(advantages),
            "adv_mean": advantages.new_zeros(()),
            "adv_std": advantages.new_zeros(()),
        }
    mean = valid.mean()
    std = valid.std(unbiased=False) + float(epsilon)
    if std_only:
        normalized = (advantages / std) * mask.float()
    else:
        normalized = ((advantages - mean) / std) * mask.float()
    return {"advantages": normalized, "adv_mean": mean, "adv_std": std}


def summarize_masked_tensor(values: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    valid = values[mask]
    if valid.numel() == 0:
        return {"mean": 0.0, "std": 0.0}
    return {
        "mean": float(valid.mean().item()),
        "std": float(valid.std(unbiased=False).item()),
    }


def train_actor(args, noise_policy, actor_optimizer, rollout) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    num_updates = 0

    ##############################################
    # Keep PPO and prior-KL identical to DSPPO, and
    # add an optional entropy bonus only on this DDIM path.
    ##############################################
    for _ in range(args.actor_num_epochs):
        for batch_inds in iter_sample_minibatches(len(rollout.texts), args.ppo_minibatch_size, rollout.frame_noise.device):
            loss_dict = compute_noise_policy_loss(
                noise_policy=noise_policy,
                text_embeds=rollout.text_embeds[batch_inds],
                frame_noise=rollout.frame_noise[batch_inds],
                frame_logprobs_old=rollout.frame_logprobs_old[batch_inds],
                frame_advantages=rollout.frame_advantages[batch_inds],
                frame_exec_mask=rollout.frame_exec_mask[batch_inds],
                clip_range=args.clip_range,
                noise_prior_kl_coef=args.noise_prior_kl_coef,
            )

            inversion_loss = torch.zeros((), device=rollout.frame_noise.device, dtype=loss_dict["loss"].dtype)
            if float(args.inversion_loss_coef) > 0.0 and hasattr(rollout, "frame_noise_gt_inv"):
                num_frames = rollout.frame_noise.shape[1]
                frame_indices = torch.arange(num_frames, device=rollout.frame_noise.device, dtype=torch.long)
                current_mean = noise_policy.forward(rollout.text_embeds[batch_inds], frame_indices)
                inversion_error = torch.norm(
                    current_mean - rollout.frame_noise_gt_inv[batch_inds],
                    p=2,
                    dim=-1,
                )
                mask_f = rollout.frame_exec_mask[batch_inds].float()
                denom = mask_f.sum().clamp(min=1.0)
                inversion_loss = float(args.inversion_loss_coef) * (inversion_error * mask_f).sum() / denom

            entropy_bonus = float(args.entropy_coef) * loss_dict["entropy"]
            total_loss = loss_dict["loss"] + inversion_loss - entropy_bonus

            actor_optimizer.zero_grad()
            total_loss.backward()
            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(noise_policy.parameters(), args.max_grad_norm)
            actor_optimizer.step()

            with torch.no_grad():
                metrics["total_loss"] = metrics.get("total_loss", 0.0) + float(total_loss.item())
                metrics["inversion_loss"] = metrics.get("inversion_loss", 0.0) + float(inversion_loss.item())
                metrics["entropy_bonus"] = metrics.get("entropy_bonus", 0.0) + float(entropy_bonus.item())
                for key, value in loss_dict.items():
                    if key == "loss":
                        continue
                    if key == "prior_kl":
                        value = float(args.noise_prior_kl_coef) * value
                    metrics[key] = metrics.get(key, 0.0) + float(value.item())
                num_updates += 1

    if num_updates == 0:
        return metrics
    return {key: value / num_updates for key, value in metrics.items()}


def main():
    args = parse_args()
    set_seed(args.seed)

    save_dir = Path(args.save_root).expanduser().resolve() / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)

    if args.train_sampling_mode in {"failure_only", "mixed"} and not args.failure_cases_file:
        raise ValueError("--failure_cases_file is required when train_sampling_mode is failure_only or mixed.")

    train_pools = build_prompt_entry_pools(
        data_root=args.data_root,
        split=args.train_split,
        max_motion_frames=args.max_motion_frames,
        failure_cases_file=args.failure_cases_file or None,
    )
    eval_entries = build_prompt_entries(
        data_root=args.data_root,
        split=args.eval_split,
        max_motion_frames=args.max_motion_frames,
    )
    failure_eval_entries: Optional[Sequence] = None
    if args.failure_eval_cases_file:
        failure_eval_entries = build_prompt_entries(
            data_root=args.data_root,
            split=args.eval_split,
            max_motion_frames=args.max_motion_frames,
            failure_cases_file=args.failure_eval_cases_file,
        )

    runtime = DSPPODDIMRuntime(args)
    noise_policy = FrameNoisePolicy(
        text_dim=runtime.actor.clip_dim,
        hidden_dim=args.noise_hidden_dim,
        action_dim=runtime.actor.njoints * runtime.actor.nfeats,
        max_frames=args.max_motion_frames,
        log_std_init=args.noise_log_std_init,
    ).to(runtime.device)
    critic = FrameCritic(
        max_frames=args.max_motion_frames,
        value_target_norm=args.value_target_norm,
        popart_beta=args.popart_beta,
        popart_epsilon=args.popart_epsilon,
    ).to(runtime.device)

    actor_optimizer = AdamW(noise_policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    critic_optimizer = AdamW(critic.parameters(), lr=args.critic_lr, weight_decay=args.weight_decay)
    wandb_run = maybe_init_wandb(args)

    rng = random.Random(args.seed)
    best_eval_score = float("-inf")
    try:
        for outer_step in range(1, args.num_outer_steps + 1):
            actor_lr = resolve_scheduled_lr(args.lr_schedule, outer_step, args.num_outer_steps, args.lr, args.min_lr)
            critic_lr = resolve_scheduled_lr(
                args.critic_lr_schedule,
                outer_step,
                args.num_outer_steps,
                args.critic_lr,
                args.critic_min_lr,
            )
            set_optimizer_lr(actor_optimizer, actor_lr)
            set_optimizer_lr(critic_optimizer, critic_lr)

            batch_entries, _ = sample_prompt_batch(
                pools=train_pools,
                mode=args.train_sampling_mode,
                batch_size=args.prompt_batch_size,
                rng=rng,
                success_sampling_weight=args.success_sampling_weight,
                failure_sampling_weight=args.failure_sampling_weight,
            )
            rollout, rollout_metrics = runtime.collect_rollout(
                entries=batch_entries,
                noise_policy=noise_policy,
                critic=critic,
                policy_deterministic=False,
                sampling_seed=args.seed if args.fixed_sampling_seed else None,
            )

            norm_adv = normalize_advantages(
                rollout.frame_advantages,
                rollout.frame_exec_mask,
                std_only=args.adv_norm_std_only,
                epsilon=args.adv_norm_epsilon,
            )
            rollout.frame_advantages = norm_adv["advantages"]
            normed_adv_stats = summarize_masked_tensor(rollout.frame_advantages, rollout.frame_exec_mask)
            rollout_metrics = merge_metrics(
                rollout_metrics,
                {
                    "frame_adv_mean": float(norm_adv["adv_mean"].item()),
                    "frame_adv_std": float(norm_adv["adv_std"].item()),
                    "normed_adv_mean": normed_adv_stats["mean"],
                    "normed_adv_std": normed_adv_stats["std"],
                },
            )

            critic_metrics = train_critic(args, critic, critic_optimizer, rollout)
            actor_metrics = train_actor(args, noise_policy, actor_optimizer, rollout)
            metrics = merge_metrics(
                rollout_metrics,
                critic_metrics,
                actor_metrics,
                {
                    "actor_lr": actor_lr,
                    "critic_lr": critic_lr,
                },
            )
            train_metrics = prefix_metrics("train/", metrics)

            if outer_step % args.log_interval == 0:
                print(f"[train] step={outer_step} " + " ".join(f"{k}={v:.4f}" for k, v in sorted(train_metrics.items())))
            if wandb_run is not None:
                wandb_run.log({**train_metrics, "outer_step": outer_step}, step=outer_step)

            if outer_step % args.eval_interval == 0:
                eval_batch = select_eval_batch(eval_entries, args.eval_prompt_batch_size, offset=0)
                eval_metrics = prefix_metrics(
                    "eval/",
                    runtime.evaluate_entries(
                        entries=eval_batch,
                        noise_policy=noise_policy,
                        policy_deterministic=args.deterministic_eval_policy,
                        sampling_seed=args.seed if args.fixed_sampling_seed else None,
                    ),
                )
                merged_eval = dict(eval_metrics)

                if failure_eval_entries:
                    failure_batch = select_eval_batch(failure_eval_entries, args.eval_prompt_batch_size, offset=0)
                    failure_metrics = prefix_metrics(
                        "eval_failure/",
                        runtime.evaluate_entries(
                            entries=failure_batch,
                            noise_policy=noise_policy,
                            policy_deterministic=args.deterministic_eval_policy,
                            sampling_seed=args.seed if args.fixed_sampling_seed else None,
                        ),
                    )
                    merged_eval.update(failure_metrics)

                print(f"[eval] step={outer_step} " + " ".join(f"{k}={v:.4f}" for k, v in sorted(merged_eval.items())))
                if wandb_run is not None:
                    wandb_run.log({**merged_eval, "outer_step": outer_step}, step=outer_step)

                best_metric = select_best_metric(merged_eval)
                if best_metric is not None:
                    _, metric_value = best_metric
                    if metric_value > best_eval_score:
                        best_eval_score = metric_value
                        save_checkpoint(
                            save_dir=save_dir,
                            filename="best.pt",
                            step=outer_step,
                            noise_policy=noise_policy,
                            critic=critic,
                            actor_optimizer=actor_optimizer,
                            critic_optimizer=critic_optimizer,
                            args=args,
                        )
                        print(f"[checkpoint] step={outer_step} saved best.pt")

            if outer_step % args.save_interval == 0 or outer_step == args.num_outer_steps:
                save_checkpoint(
                    save_dir=save_dir,
                    filename="latest.pt",
                    step=outer_step,
                    noise_policy=noise_policy,
                    critic=critic,
                    actor_optimizer=actor_optimizer,
                    critic_optimizer=critic_optimizer,
                    args=args,
                )
    finally:
        runtime.close()
        if wandb_run is not None:
            wandb_run.finish()

'''
python train/train_dsppo_phc_DDIM.py \
  --model_path /home/gxy/hay-thesis/motion-diffusion-model-phc/save/humanml_enc_512_50steps/model000750000.pt \
  --data_root /home/gxy/hay-thesis/HumanML3D/HumanML3D \
  --run_name dsppo_data_failure_0048_reward_weight \
  --train_split train \
  --eval_split test \
  --train_sampling_mode failure_only \
  --failure_cases_file /home/gxy/hay-thesis/PHC/output/eval_mdm/mdm_eval_train_full/falling_cases.txt \
  --failure_eval_cases_file /home/gxy/hay-thesis/PHC/output/eval_mdm/mdm_eval_v7_full/falling_cases.txt \
  --max_motion_frames 196 \
  --prompt_batch_size 512 \
  --ppo_minibatch_size 128 \
  --critic_minibatch_size 128 \
  --actor_num_epochs 2 \
  --critic_num_epochs 2 \
  --noise_prior_kl_coef 0 \
  --inversion_loss_coef 5e-5 \
  --lr_schedule cosine \
  --lr 5e-4 \
  --min_lr 1e-5 \
  --critic_lr_schedule cosine \
  --critic_lr 1e-4 \
  --critic_min_lr 1e-6 \
  --clip_range 2e-1 \
  --guidance_param 2.5 \
  --noise_hidden_dim 256 \
  --noise_log_std_init -0.0 \
  --value_target_norm popart \
  --popart_beta 0.005 \
  --popart_epsilon 1e-5 \
  --phc_num_envs 512 \
  --phc_max_steps 400 \
  --num_outer_steps 100 \
  --eval_interval 10 \
  --eval_prompt_batch_size 512 \
  --save_root /home/gxy/hay-thesis/motion-diffusion-model-phc/save \
  --save_interval 10 \
  --log_interval 5 \
  --wandb_mode online \
  --fixed_sampling_seed \
  --fail_penalty -0.05 \
  --reward_norm \
  --dense_reward_weight 1.5 \
  --pose_reward_weight 2.5 \
  --pose_reward_alpha 5 \
  --velocity_reward_weight 0.5 \
  --velocity_reward_alpha 0.5 \
  --ddim_eta 0.0 \
  --deterministic_eval_policy \
  --entropy_coef 1e-5 \
  --ddim_steps 10
'''
if __name__ == "__main__":
    main()
