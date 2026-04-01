#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import imageio.v2 as imageio
import numpy as np
import isaacgym  # noqa: F401
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from mdm_core.data_loaders.tensors import lengths_to_mask
from mdm_core.model.cfg_sampler import ClassifierFreeSampleModel
from mdm_core.utils.model_util import create_model_and_diffusion, load_model_wo_clip

from model.lora_attention import inject_lora_into_mdm, load_lora_state_dict
from train.phc_reward_runner import PHCRewardRunner
from utils.ddpo_parser import _load_checkpoint_args
from utils.mdm_phc_postprocess import HumanML3DPostprocessor


SMPL_2_MUJOCO = [
    0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9,
    12, 15, 13, 16, 18, 20, 22, 14, 17, 19, 21, 23,
]
MUJOCO_TO_SMPL = np.argsort(np.asarray(SMPL_2_MUJOCO, dtype=np.int64))
ISAAC_TO_HML_MAT = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)
T2M_KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]
CHAIN_COLORS = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]


@dataclass
class EvalSample:
    sample_idx: int
    sample_id: str
    source: str
    db_key: str
    caption_idx: int
    text: str
    tokens: str
    length_20fps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PHC tracking evaluation on MDM-generated motions for one prompt or a list of HumanML3D sample ids.",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--prompt", type=str, default="")
    input_group.add_argument("--prompt-file", type=str, default="")
    input_group.add_argument("--sample-list", type=str, default="")
    input_group.add_argument("--db-key", type=str, default="")

    parser.add_argument("--caption-idx", type=int, default=0)
    parser.add_argument(
        "--data-root",
        type=str,
        default=str((REPO_ROOT / "dataset" / "HumanML3D").resolve()),
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--lora-path", type=str, default="")
    parser.add_argument("--guidance-param", type=float, default=2.5)
    parser.add_argument("--motion-length-sec", type=float, default=6.0)
    parser.add_argument("--length-20fps", type=int, default=0)
    parser.add_argument("--max-motion-frames", type=int, default=120)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-layer-scope", type=str, default="all", choices=["all", "last3"])

    parser.add_argument(
        "--phc-config-path",
        type=str,
        default="/home/gxy/hay-thesis/PHC/output/HumanoidIm/phc_kp_mcp_iccv/.hydra/config.yaml",
    )
    parser.add_argument(
        "--phc-actor-ckpt",
        type=str,
        default="/home/gxy/hay-thesis/PHC/output/HumanoidIm/phc_kp_mcp_iccv/Humanoid.pth",
    )
    parser.add_argument("--phc-num-envs", type=int, default=1)
    parser.add_argument("--phc-max-steps", type=int, default=512)
    parser.add_argument("--show-viewer", action="store_true")

    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--render-limit", type=int, default=0)
    parser.add_argument("--render-fps", type=int, default=30)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_dummy_data():
    return SimpleNamespace(dataset=SimpleNamespace())


def normalize_model_args(model_path: Path, guidance_param: float, device: torch.device) -> SimpleNamespace:
    checkpoint_args = _load_checkpoint_args(str(model_path))
    model_args = SimpleNamespace(**checkpoint_args)
    model_args.model_path = str(model_path)
    model_args.guidance_param = float(guidance_param)
    model_args.device = str(device)
    model_args.max_motion_frames = getattr(model_args, "max_motion_frames", 120)
    if getattr(model_args, "cond_mask_prob", 0.0) == 0:
        model_args.guidance_param = 1.0
    return model_args


def load_sampling_stack(
    model_args: SimpleNamespace,
    lora_path: Optional[Path],
    device: torch.device,
    lora_rank: int,
    lora_alpha: float,
    lora_layer_scope: str,
):
    model, diffusion = create_model_and_diffusion(model_args, create_dummy_data())
    state_dict = torch.load(model_args.model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)

    if lora_path is not None:
        inject_lora_into_mdm(
            model,
            rank=int(lora_rank),
            alpha=float(lora_alpha),
            layer_scope=str(lora_layer_scope),
        )
        payload = torch.load(str(lora_path), map_location="cpu")
        load_lora_state_dict(model, payload["lora"] if "lora" in payload else payload)

    model.to(device)
    model.eval()
    sampling_model = model if model_args.guidance_param == 1.0 else ClassifierFreeSampleModel(model)
    postprocessor = HumanML3DPostprocessor(model, data_root=str(Path(model_args.data_root).resolve()))
    return model, sampling_model, diffusion, postprocessor


def _parse_humanml_length(parts: List[str], motion_len: int, max_motion_frames: int) -> int:
    fallback = min(int(motion_len), int(max_motion_frames))
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
    return min(end - start, int(max_motion_frames))


def load_dataset_sample(data_root: Path, db_key: str, caption_idx: int, max_motion_frames: int, sample_idx: int) -> EvalSample:
    text_path = data_root / "texts" / f"{db_key}.txt"
    motion_path = data_root / "new_joint_vecs" / f"{db_key}.npy"
    if not text_path.is_file():
        raise FileNotFoundError(f"Text file not found: {text_path}")
    if not motion_path.is_file():
        raise FileNotFoundError(f"Motion file not found: {motion_path}")

    motion_len = int(np.load(str(motion_path), mmap_mode="r").shape[0])
    lines = [line.strip() for line in text_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if caption_idx < 0 or caption_idx >= len(lines):
        raise IndexError(f"caption_idx={caption_idx} out of range for {db_key}, num_captions={len(lines)}")

    parts = lines[caption_idx].split("#")
    text = parts[0].strip()
    tokens = parts[1].strip() if len(parts) > 1 else ""
    length_20fps = _parse_humanml_length(parts, motion_len=motion_len, max_motion_frames=max_motion_frames)
    sample_id = f"{db_key}_c{caption_idx}"
    return EvalSample(
        sample_idx=int(sample_idx),
        sample_id=sample_id,
        source="dataset",
        db_key=str(db_key),
        caption_idx=int(caption_idx),
        text=text,
        tokens=tokens,
        length_20fps=int(length_20fps),
    )


def parse_sample_spec(line: str) -> Tuple[str, int]:
    entry = line.strip()
    if not entry:
        raise ValueError("Empty sample spec")
    if ":" not in entry:
        return entry, 0

    db_key, caption_idx = entry.rsplit(":", 1)
    return db_key.strip(), int(caption_idx)


def resolve_samples(args: argparse.Namespace) -> List[EvalSample]:
    data_root = Path(args.data_root).expanduser().resolve()
    samples: List[EvalSample] = []

    # Input resolution is separated here so the rest of the pipeline only deals with
    # a single, explicit sample structure regardless of whether the source is text or db_key.
    if args.prompt:
        length_20fps = args.length_20fps if args.length_20fps > 0 else int(round(float(args.motion_length_sec) * 20.0))
        length_20fps = max(2, min(int(args.max_motion_frames), int(length_20fps)))
        samples.append(
            EvalSample(
                sample_idx=0,
                sample_id="prompt_0000",
                source="prompt",
                db_key="",
                caption_idx=-1,
                text=args.prompt.strip(),
                tokens="",
                length_20fps=length_20fps,
            )
        )
        return samples

    if args.prompt_file:
        prompt_path = Path(args.prompt_file).expanduser().resolve()
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        length_20fps = args.length_20fps if args.length_20fps > 0 else int(round(float(args.motion_length_sec) * 20.0))
        length_20fps = max(2, min(int(args.max_motion_frames), int(length_20fps)))
        prompts = [line.strip() for line in prompt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for idx, prompt in enumerate(prompts):
            samples.append(
                EvalSample(
                    sample_idx=idx,
                    sample_id=f"prompt_{idx:04d}",
                    source="prompt_file",
                    db_key="",
                    caption_idx=-1,
                    text=prompt,
                    tokens="",
                    length_20fps=length_20fps,
                )
            )
        return samples

    if args.db_key:
        return [
            load_dataset_sample(
                data_root=data_root,
                db_key=str(args.db_key).strip(),
                caption_idx=int(args.caption_idx),
                max_motion_frames=int(args.max_motion_frames),
                sample_idx=0,
            )
        ]

    sample_list_path = Path(args.sample_list).expanduser().resolve()
    if not sample_list_path.is_file():
        raise FileNotFoundError(f"Sample list not found: {sample_list_path}")

    raw_specs = [
        line.strip()
        for line in sample_list_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and (not line.strip().startswith("#"))
    ]
    for idx, spec in enumerate(raw_specs):
        db_key, caption_idx = parse_sample_spec(spec)
        samples.append(
            load_dataset_sample(
                data_root=data_root,
                db_key=db_key,
                caption_idx=caption_idx,
                max_motion_frames=int(args.max_motion_frames),
                sample_idx=idx,
            )
        )
    return samples


def iter_batches(items: Sequence[EvalSample], batch_size: int) -> Iterable[List[EvalSample]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def sample_reference_batch(
    model,
    sampling_model,
    diffusion,
    postprocessor: HumanML3DPostprocessor,
    samples: Sequence[EvalSample],
    device: torch.device,
    guidance_param: float,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    texts = [sample.text for sample in samples]
    tokens = [sample.tokens for sample in samples]
    lengths = torch.as_tensor([sample.length_20fps for sample in samples], device=device, dtype=torch.long)
    max_frames = int(lengths.max().item())
    mask = lengths_to_mask(lengths, max_frames).unsqueeze(1).unsqueeze(1)

    model_kwargs = {
        "y": {
            "text": texts,
            "tokens": tokens,
            "lengths": lengths,
            "mask": mask,
            "text_embed": model.encode_text(texts).detach(),
        }
    }
    if guidance_param != 1.0:
        model_kwargs["y"]["scale"] = torch.full((len(samples),), float(guidance_param), device=device)

    # Sampling, post-processing, and PHC reference packaging are kept together so every batch
    # uses exactly the same motion-to-reference transform as DDPO PHC rollout training.
    with torch.no_grad():
        out = diffusion.p_sample_loop_collect(
            sampling_model,
            (len(samples), model.njoints, model.nfeats, max_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            store_cpu=True,
        )
        final_sample = out["sample"].to(device)
        joints24 = postprocessor.sample_to_joints24_20fps(final_sample, lengths, mask)
        ref_motion_batch, ref_lengths = postprocessor.joints24_20fps_to_phc_ref(joints24, lengths)

    meta = []
    for sample in samples:
        meta.append(
            {
                "sample_idx": int(sample.sample_idx),
                "sample_id": sample.sample_id,
                "source": sample.source,
                "db_key": sample.db_key if sample.db_key else sample.sample_id,
                "caption_idx": int(sample.caption_idx),
                "caption": sample.text,
                "tokens": sample.tokens,
                "length": int(sample.length_20fps),
            }
        )

    del out, final_sample, joints24
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ref_motion_batch, ref_lengths, meta


def run_tracking_eval(
    samples: Sequence[EvalSample],
    model,
    sampling_model,
    diffusion,
    postprocessor: HumanML3DPostprocessor,
    reward_runner: PHCRewardRunner,
    device: torch.device,
    guidance_param: float,
    batch_size: int,
) -> List[Dict]:
    episodes: List[Dict] = []
    total = len(samples)
    for batch_idx, batch_samples in enumerate(iter_batches(samples, batch_size), start=1):
        ref_motion_batch, ref_lengths, meta = sample_reference_batch(
            model=model,
            sampling_model=sampling_model,
            diffusion=diffusion,
            postprocessor=postprocessor,
            samples=batch_samples,
            device=device,
            guidance_param=guidance_param,
        )
        batch_episodes = reward_runner.evaluate_batch(ref_motion_batch, ref_lengths, meta=meta)
        episodes.extend(batch_episodes)
        print(f"[tracking-eval] finished batch {batch_idx}, episodes={min(batch_idx * batch_size, total)}/{total}")

    episodes = sorted(episodes, key=lambda episode: int(episode.get("sample_idx", 0)))
    return episodes


def tracking_motion_to_plot_coords(motion: np.ndarray) -> np.ndarray:
    motion = np.asarray(motion, dtype=np.float32)
    if motion.ndim != 3:
        raise ValueError(f"Expected motion with shape [T, J, 3], got {motion.shape}")
    if motion.shape[1] >= 24:
        motion = motion[:, :24, :]
        motion = motion[:, MUJOCO_TO_SMPL, :]
    if motion.shape[1] < 22:
        raise ValueError(f"Expected at least 22 joints, got {motion.shape}")

    motion22 = motion[:, :22, :]
    hml = np.matmul(motion22, ISAAC_TO_HML_MAT)
    hml = hml.copy()
    hml[:, :, 1] -= float(hml[:, :, 1].min())
    return hml[:, :, [0, 2, 1]]


def compute_plot_limits(motion: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    mins = motion.min(axis=(0, 1))
    maxs = motion.max(axis=(0, 1))
    spans = np.maximum(maxs - mins, 1e-3)
    pad = np.maximum(spans * 0.08, 0.05)
    xlim = (float(mins[0] - pad[0]), float(maxs[0] + pad[0]))
    ylim = (float(mins[1] - pad[1]), float(maxs[1] + pad[1]))
    zlim = (0.0, float(maxs[2] + pad[2]))
    return xlim, ylim, zlim


def build_video_title(label: str, episode: Dict) -> str:
    status = "SUCCESS" if bool(episode.get("success", False)) else "FAIL"
    prompt = str(episode.get("caption", ""))
    return (
        f"{label}\n"
        f"{status} | exec={int(episode.get('exec_len', -1))}/{int(episode.get('target_len', -1))}"
        f" | return={float(episode.get('return_mean', 0.0)):.3f}\n"
        f"{prompt}"
    )


def render_motion_video(save_path: Path, motion: np.ndarray, title: str, fps: int) -> None:
    xlim, ylim, zlim = compute_plot_limits(motion)
    fig = plt.figure(figsize=(6, 6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    writer = imageio.get_writer(str(save_path), fps=fps, codec="libx264", quality=8)

    floor = [
        [xlim[0], ylim[0], 0.0],
        [xlim[0], ylim[1], 0.0],
        [xlim[1], ylim[1], 0.0],
        [xlim[1], ylim[0], 0.0],
    ]
    title_wrapped = "\n".join(textwrap.wrap(title, width=40))

    for frame_idx in range(motion.shape[0]):
        ax.cla()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.view_init(elev=24, azim=-58)
        ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], max(zlim[1] - zlim[0], 1e-3)))
        ax.set_title(title_wrapped, fontsize=10, pad=10)
        ax.add_collection3d(Poly3DCollection([floor], facecolor=(0.75, 0.75, 0.75, 0.18), edgecolor="none"))

        traj = motion[: frame_idx + 1, 0]
        ax.plot(traj[:, 0], traj[:, 1], np.zeros_like(traj[:, 0]), color="#777777", linewidth=1.2, alpha=0.9)

        joints = motion[frame_idx]
        for chain, color in zip(T2M_KINEMATIC_CHAIN, CHAIN_COLORS):
            ax.plot(joints[chain, 0], joints[chain, 1], joints[chain, 2], color=color, linewidth=3.0)

        ax.set_axis_off()
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        writer.append_data(frame)

    writer.close()
    plt.close(fig)


def episode_row(episode: Dict) -> Dict[str, object]:
    target_len = int(episode.get("target_len", 0))
    exec_len = int(episode.get("exec_len", 0))
    exec_ratio = float(exec_len / target_len) if target_len > 0 else float("nan")
    return {
        "sample_idx": int(episode.get("sample_idx", -1)),
        "sample_id": str(episode.get("sample_id", "")),
        "source": str(episode.get("source", "")),
        "db_key": str(episode.get("db_key", "")),
        "caption_idx": int(episode.get("caption_idx", -1)),
        "caption": str(episode.get("caption", "")),
        "length_20fps": int(episode.get("length", -1)),
        "success": bool(episode.get("success", False)),
        "terminate": bool(episode.get("terminate", False)),
        "exec_len": exec_len,
        "target_len": target_len,
        "exec_ratio": exec_ratio,
        "return_mean": float(episode.get("return_mean", 0.0)),
        "return_sum": float(episode.get("return_sum", 0.0)),
    }


def save_summary_csv(path: Path, episodes: Sequence[Dict]) -> None:
    rows = [episode_row(episode) for episode in episodes]
    fieldnames = [
        "sample_idx",
        "sample_id",
        "source",
        "db_key",
        "caption_idx",
        "caption",
        "length_20fps",
        "success",
        "terminate",
        "exec_len",
        "target_len",
        "exec_ratio",
        "return_mean",
        "return_sum",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_aggregate_metrics(episodes: Sequence[Dict]) -> Dict[str, object]:
    if len(episodes) == 0:
        return {
            "num_episodes": 0,
            "success_rate": float("nan"),
            "terminate_rate": float("nan"),
            "mean_return_mean": float("nan"),
            "mean_exec_ratio": float("nan"),
        }

    success = np.asarray([float(bool(ep.get("success", False))) for ep in episodes], dtype=np.float32)
    terminate = np.asarray([float(bool(ep.get("terminate", False))) for ep in episodes], dtype=np.float32)
    return_mean = np.asarray([float(ep.get("return_mean", 0.0)) for ep in episodes], dtype=np.float32)
    exec_ratio = np.asarray(
        [
            float(ep.get("exec_len", 0)) / max(1.0, float(ep.get("target_len", 1)))
            for ep in episodes
        ],
        dtype=np.float32,
    )
    return {
        "num_episodes": int(len(episodes)),
        "success_rate": float(success.mean()),
        "terminate_rate": float(terminate.mean()),
        "mean_return_mean": float(return_mean.mean()),
        "mean_exec_ratio": float(exec_ratio.mean()),
    }


def maybe_render_episodes(output_dir: Path, label: str, episodes: Sequence[Dict], render_limit: int, fps: int) -> List[str]:
    if render_limit == 0:
        return []

    render_dir = output_dir / "videos"
    render_dir.mkdir(parents=True, exist_ok=True)
    selected = list(episodes) if render_limit < 0 else list(episodes[:render_limit])
    video_paths: List[str] = []

    for episode in selected:
        pred_motion = np.asarray(episode["pred_motion"], dtype=np.float32)
        plot_motion = tracking_motion_to_plot_coords(pred_motion)
        save_path = render_dir / f"{episode.get('sample_id', 'sample')}.mp4"
        render_motion_video(
            save_path=save_path,
            motion=plot_motion,
            title=build_video_title(label, episode),
            fps=fps,
        )
        video_paths.append(str(save_path))

    return video_paths


def derive_label(args: argparse.Namespace) -> str:
    if args.label:
        return args.label.strip()
    if args.lora_path:
        return Path(args.lora_path).stem
    return f"{Path(args.model_path).stem}_base"


def derive_output_dir(args: argparse.Namespace, label: str) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    return (REPO_ROOT / "eval" / "tracking" / "output" / label).resolve()


def main() -> None:
    args = parse_args()
    label = derive_label(args)
    output_dir = derive_output_dir(args, label)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    lora_path = Path(args.lora_path).expanduser().resolve() if args.lora_path else None
    if not model_path.is_file():
        raise FileNotFoundError(f"model_path not found: {model_path}")
    if lora_path is not None and not lora_path.is_file():
        raise FileNotFoundError(f"lora_path not found: {lora_path}")

    samples = resolve_samples(args)
    if len(samples) == 0:
        raise RuntimeError("No evaluation samples resolved.")
    if args.show_viewer and int(args.phc_num_envs) != 1:
        raise ValueError("--show-viewer currently requires --phc-num-envs 1 for a clean online preview.")

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    model_args = normalize_model_args(model_path=model_path, guidance_param=args.guidance_param, device=device)
    model_args.data_root = str(data_root)
    model_args.max_motion_frames = int(args.max_motion_frames)

    reward_runner = PHCRewardRunner(
        config_path=args.phc_config_path,
        actor_ckpt=args.phc_actor_ckpt,
        num_envs=int(args.phc_num_envs),
        max_steps=int(args.phc_max_steps),
        headless=not bool(args.show_viewer),
        no_virtual_display=True,
    )

    # Seed after PHC runner construction so PHC policy initialization does not perturb
    # the diffusion RNG stream. This matches the compare_fixed_eval_tracking.py behavior.
    set_seed(int(args.seed))

    model, sampling_model, diffusion, postprocessor = load_sampling_stack(
        model_args=model_args,
        lora_path=lora_path,
        device=device,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_layer_scope=args.lora_layer_scope,
    )

    episodes = run_tracking_eval(
        samples=samples,
        model=model,
        sampling_model=sampling_model,
        diffusion=diffusion,
        postprocessor=postprocessor,
        reward_runner=reward_runner,
        device=device,
        guidance_param=float(model_args.guidance_param),
        batch_size=max(1, int(args.phc_num_envs)),
    )

    tracking_pkl = output_dir / "tracking_episodes.pkl"
    summary_csv = output_dir / "summary.csv"
    aggregate_json = output_dir / "summary.json"
    sample_json = output_dir / "samples.json"
    config_json = output_dir / "run_config.json"

    joblib.dump(episodes, tracking_pkl)
    save_summary_csv(summary_csv, episodes)

    aggregate = {
        "label": label,
        "model_path": str(model_path),
        "lora_path": str(lora_path) if lora_path is not None else "",
        "data_root": str(data_root),
        "tracking_pkl": str(tracking_pkl),
        "summary_csv": str(summary_csv),
        "aggregate": compute_aggregate_metrics(episodes),
    }
    aggregate_json.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    sample_json.write_text(
        json.dumps([asdict(sample) for sample in samples], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    config_json.write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")

    video_paths = maybe_render_episodes(
        output_dir=output_dir,
        label=label,
        episodes=episodes,
        render_limit=int(args.render_limit),
        fps=int(args.render_fps),
    )
    if video_paths:
        aggregate["videos"] = video_paths
        aggregate_json.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "tracking_pkl": str(tracking_pkl),
                "summary_csv": str(summary_csv),
                "summary_json": str(aggregate_json),
                "num_episodes": len(episodes),
                "success_rate": aggregate["aggregate"]["success_rate"],
                "videos": video_paths,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
