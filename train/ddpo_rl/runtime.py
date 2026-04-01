from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import wandb

from mdm_core.data_loaders.tensors import lengths_to_mask
from mdm_core.utils.model_util import create_model_and_diffusion, load_model_wo_clip

from model.lora_attention import inject_lora_into_mdm, iter_lora_parameters, load_lora_state_dict, lora_state_dict, mark_only_lora_trainable
from train.ddpo_rl.buffer import build_rollout_batch
from train.ddpo_rl.reward import extract_episode_reward_terms, score_reward_terms
from train.phc_reward_runner import PHCRewardRunner
from utils.mdm_phc_postprocess import HumanML3DPostprocessor

from train.ddpo_rl.algorithms.common import build_sampling_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_dummy_data():
    return SimpleNamespace(dataset=SimpleNamespace())


def load_policy(args, device, trainable: bool = True):
    model, diffusion = create_model_and_diffusion(args, create_dummy_data())
    state_dict = torch.load(args.model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)

    inject_lora_into_mdm(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        layer_scope=args.lora_layer_scope,
    )
    if args.resume_lora_path:
        payload = torch.load(args.resume_lora_path, map_location="cpu")
        load_lora_state_dict(model, payload["lora"] if "lora" in payload else payload)

    if trainable:
        mark_only_lora_trainable(model)
    else:
        for param in model.parameters():
            param.requires_grad = False

    model.to(device)
    model.eval()
    return model, diffusion


def init_wandb(args):
    if args.wandb_mode == "disabled":
        return None
    run_name = args.wandb_run_name or Path(args.save_dir).name
    return wandb.init(
        project=args.wandb_project,
        name=run_name,
        dir=args.save_dir,
        mode=args.wandb_mode,
        config=vars(args),
    )


def save_checkpoint(save_dir: Path, step: int, model, optimizer, algorithm_state=None, best: bool = False):
    payload = {
        "step": int(step),
        "lora": lora_state_dict(model),
        "optimizer": optimizer.state_dict(),
    }
    if algorithm_state:
        payload["algorithm"] = algorithm_state
    torch.save(payload, save_dir / "latest_lora.pt")
    if best:
        torch.save(payload, save_dir / "best_lora.pt")


class DDPORuntime:
    def __init__(self, args, device, model, diffusion, reward_spec):
        self.args = args
        self.device = device
        self.model = model
        self.diffusion = diffusion
        self.reward_spec = reward_spec
        self.postprocessor = HumanML3DPostprocessor(model, data_root=args.data_root)
        self.reward_runner = PHCRewardRunner(
            config_path=args.phc_config_path,
            actor_ckpt=args.phc_actor_ckpt,
            num_envs=args.phc_num_envs,
            max_steps=args.phc_max_steps,
        )

    def collect_rollout_batch(self, samples, sampling_args=None, collect_hidden: bool = False):
        args = sampling_args or self.args
        model = self.model
        diffusion = self.diffusion
        device = self.device

        ##############################
        # Build the prompt-conditioned MDM inputs for one rollout batch.
        ##############################
        model.eval()
        texts = [sample.text for sample in samples]
        tokens = [sample.tokens for sample in samples]
        db_keys = [sample.db_key for sample in samples]
        caption_indices = [sample.caption_idx for sample in samples]
        lengths = torch.as_tensor([sample.length_20fps for sample in samples], device=device, dtype=torch.long)
        max_frames = int(lengths.max().item())
        mask = lengths_to_mask(lengths, max_frames).unsqueeze(1).unsqueeze(1)
        scales = torch.full((len(samples),), float(args.guidance_param), device=device)

        model_kwargs = {
            "y": {
                "text": texts,
                "tokens": tokens,
                "lengths": lengths,
                "mask": mask,
                "text_embed": model.encode_text(texts).detach(),
            }
        }
        if args.guidance_param != 1.0:
            model_kwargs["y"]["scale"] = scales

        ##############################
        # Sample motions with the diffusion policy, then postprocess them into
        # PHC reference motions. Actor-critic optionally asks for hidden states
        # here, but the rollout path itself stays shared across algorithms.
        ##############################
        sampling_model = build_sampling_model(model, args.guidance_param)
        with torch.no_grad():
            if args.fixed_sampling_seed:
                torch.manual_seed(args.seed)
                torch.cuda.manual_seed_all(args.seed)

            # Sampling and PHC execution stay shared across all algorithms.
            # Actor-critic only asks for extra hidden features at this stage.
            out = diffusion.p_sample_loop_collect(
                sampling_model,
                (len(samples), model.njoints, model.nfeats, max_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=args.sample_progress,
                store_cpu=True,
                collect_hidden=collect_hidden,
            )
            final_sample = out["sample"].to(device)
            joints24 = self.postprocessor.sample_to_joints24_20fps(final_sample, lengths, mask)
            ref_motion_batch, ref_lengths = self.postprocessor.joints24_20fps_to_phc_ref(joints24, lengths)

        ##############################
        # Run PHC tracking on the generated references and collect one episode
        # record per prompt in the batch.
        ##############################
        meta = [
            {
                "sample_idx": idx,
                "db_key": sample.db_key,
                "caption_idx": sample.caption_idx,
                "caption": sample.text,
                "tokens": sample.tokens,
                "length": sample.length_20fps,
            }
            for idx, sample in enumerate(samples)
        ]
        episodes = self.reward_runner.evaluate_batch(ref_motion_batch, ref_lengths, meta=meta)
        if len(episodes) != len(samples):
            raise RuntimeError(f"PHC returned {len(episodes)} episodes for batch size {len(samples)}.")

        ##############################
        # Convert PHC episode statistics into scalar rewards and package the
        # diffusion trajectory plus PHC outputs into one rollout object.
        ##############################
        reward_terms = [extract_episode_reward_terms(ep) for ep in episodes]
        rewards = torch.as_tensor(
            [score_reward_terms(term, self.reward_spec)[0] for term in reward_terms],
            dtype=torch.float32,
        )
        return build_rollout_batch(
            texts=texts,
            tokens=tokens,
            db_keys=db_keys,
            caption_indices=caption_indices,
            text_embeds=model_kwargs["y"]["text_embed"],
            lengths=lengths.detach().cpu(),
            mask=mask.detach().cpu(),
            scales=scales.detach().cpu(),
            sample=out["sample"],
            trajectory=out["trajectory"],
            rewards=rewards,
            episodes=episodes,
        )
