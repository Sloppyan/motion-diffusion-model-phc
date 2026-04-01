from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence

import numpy as np
import torch

sys.path.append(os.getcwd())

from mdm_core.data_loaders.tensors import lengths_to_mask
from mdm_core.model.cfg_sampler import ClassifierFreeSampleModel
from mdm_core.utils import dist_util
from mdm_core.utils.fixseed import fixseed
from mdm_core.utils.model_util import create_model_and_diffusion, load_model_wo_clip
from mdm_core.utils.mdm_phc_postprocess import HumanML3DPostprocessor
from mdm_core.utils.parser_util import generate_args


def _create_dummy_data() -> SimpleNamespace:
    return SimpleNamespace(dataset=SimpleNamespace())


def _normalize_prompt_list(prompts: Sequence[str] | str) -> List[str]:
    if isinstance(prompts, str):
        prompts = [prompts]
    prompts = [str(prompt).strip() for prompt in prompts]
    if len(prompts) == 0:
        raise ValueError("prompts must not be empty.")
    return prompts


def _resolve_data_root(args) -> Path:
    repo_root = Path(__file__).resolve().parent

    explicit_root = getattr(args, "data_root", "") or getattr(args, "data_dir", "")
    if explicit_root:
        candidate = Path(str(explicit_root)).expanduser()
        if not candidate.is_absolute():
            candidate = (repo_root / candidate).resolve()
        if candidate.is_dir():
            return candidate

    if args.dataset == "humanml":
        fallback_candidates = [
            repo_root / "dataset" / "HumanML3D",
            repo_root / "data",
        ]
    elif args.dataset == "kit":
        fallback_candidates = [
            repo_root / "dataset" / "KIT-ML",
        ]
    else:
        fallback_candidates = []

    for candidate in fallback_candidates:
        if candidate.is_dir():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Unable to resolve data root for dataset={args.dataset}. "
        f"Tried explicit data_root/data_dir and repo-local defaults under {repo_root}."
    )


class MDMTalker:
    def __init__(self):
        self.args = args = generate_args()
        fixseed(args.seed)
        dist_util.setup_dist(args.device)

        self.max_frames = 196 if args.dataset in ["kit", "humanml"] else 60
        self.fps = 12.5 if args.dataset == "kit" else 20.0
        self.default_n_frames = max(2, min(int(self.max_frames), int(round(args.motion_length * self.fps))))

        # Model initialization is intentionally length-agnostic. Per-sample lengths are
        # injected during generate_motion() so one talker instance can serve variable-length requests.
        self.model, self.diffusion = create_model_and_diffusion(args, _create_dummy_data())
        state_dict = torch.load(args.model_path, map_location="cpu")
        load_model_wo_clip(self.model, state_dict)
        self.model.to(dist_util.dev())
        self.model.eval()

        if args.guidance_param != 1:
            self.sampling_model = ClassifierFreeSampleModel(self.model)
        else:
            self.sampling_model = self.model

        # Post-processing needs the HumanML/KIT normalization statistics even when
        # generate_args() only exposes the legacy data_dir option.
        self.data_root = _resolve_data_root(args)
        self.postprocessor = HumanML3DPostprocessor(
            self.model,
            data_root=str(self.data_root),
        )

    def _normalize_lengths(
        self,
        lengths_20fps: Optional[Sequence[int] | int],
        batch_size: int,
    ) -> torch.Tensor:
        if lengths_20fps is None:
            lengths = [self.default_n_frames] * batch_size
        elif isinstance(lengths_20fps, (int, np.integer)):
            lengths = [int(lengths_20fps)] * batch_size
        else:
            lengths = [int(length) for length in lengths_20fps]
            if len(lengths) != batch_size:
                raise ValueError(
                    f"lengths_20fps batch mismatch: expected {batch_size}, got {len(lengths)}"
                )

        lengths = np.asarray(lengths, dtype=np.int64)
        lengths = np.clip(lengths, 2, int(self.max_frames))
        return torch.as_tensor(lengths, device=dist_util.dev(), dtype=torch.long)

    def _normalize_tokens(
        self,
        tokens: Optional[Sequence[str] | str],
        batch_size: int,
    ) -> List[str]:
        if tokens is None:
            return [""] * batch_size
        if isinstance(tokens, str):
            return [tokens] * batch_size
        tokens = [str(token) for token in tokens]
        if len(tokens) != batch_size:
            raise ValueError(f"tokens batch mismatch: expected {batch_size}, got {len(tokens)}")
        return tokens

    def generate_motion(
        self,
        prompts,
        out_path: str = "mdm_out",
        num_repetitions: int = 1,
        *,
        lengths_20fps: Optional[Sequence[int] | int] = None,
        tokens: Optional[Sequence[str] | str] = None,
        show_progress: bool = True,
    ):
        del out_path  # Kept for backward compatibility with older callers.

        prompts = _normalize_prompt_list(prompts)
        batch_size = len(prompts)
        lengths = self._normalize_lengths(lengths_20fps, batch_size)
        max_n_frames = int(lengths.max().item())
        token_list = self._normalize_tokens(tokens, batch_size)

        # Sampling inputs are rebuilt for every call so lengths can vary per sample.
        with torch.no_grad():
            mask = lengths_to_mask(lengths, max_n_frames).unsqueeze(1).unsqueeze(1)
            model_kwargs = {
                "y": {
                    "text": prompts,
                    "tokens": token_list,
                    "lengths": lengths,
                    "mask": mask,
                    "text_embed": self.model.encode_text(prompts).detach(),
                }
            }
            if self.args.guidance_param != 1:
                model_kwargs["y"]["scale"] = torch.full(
                    (batch_size,),
                    float(self.args.guidance_param),
                    device=dist_util.dev(),
                )

            # Repetition handling stays local to the sampler. Each repetition uses the same
            # text and length conditions and appends another batch of sampled motions.
            all_joints24 = []
            for _ in range(int(max(1, num_repetitions))):
                sample_out = self.diffusion.p_sample_loop_collect(
                    self.sampling_model,
                    (batch_size, self.model.njoints, self.model.nfeats, max_n_frames),
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    progress=bool(show_progress),
                    store_cpu=True,
                )
                final_sample = sample_out["sample"].to(dist_util.dev())
                joints24 = self.postprocessor.sample_to_joints24_20fps(final_sample, lengths, mask)
                all_joints24.append(joints24.cpu().numpy())

        motions = np.concatenate(all_joints24, axis=0)
        return motions.squeeze()


if __name__ == "__main__":
    mdm_talker = MDMTalker()
    mdm_talker.generate_motion(["Running round"])
