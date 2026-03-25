from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, REPO_ROOT / "src"):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from mdm_core import activate_import_context

activate_import_context(REPO_ROOT)
os.chdir(REPO_ROOT)

from mdm_core.data_loaders import humanml_utils
from mdm_core.data_loaders.get_data import get_dataset
from mdm_core.data_loaders.humanml.scripts.motion_process import process_file, recover_from_ric
import mdm_core.data_loaders.humanml.utils.paramUtil as paramUtil
from mdm_core.data_loaders.humanml.utils.plot_script import plot_3d_motion
from mdm_core.model.cfg_sampler import ClassifierFreeSampleModel
from mdm_core.utils import dist_util
from mdm_core.utils.fixseed import fixseed
from mdm_core.utils.model_util import create_model_and_diffusion, load_model_wo_clip


DEFAULT_MODEL_PATH = REPO_ROOT / "save" / "humanml_enc_512_50steps" / "model000750000.pt"
DEFAULT_INPUT_MOTION = Path(
    "/home/gxy/hay-thesis/ACMDM/data/dataset/aist-mini/22-joints/gBR_sBM_cAll_d04_mBR1_ch01.npy"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "save" / "test-viz"
DEFAULT_RAW_FRAMES = 196
DEFAULT_MAX_FRAMES = 196
DEFAULT_FPS = 20
LOCKED_JOINTS = ("pelvis", "left_foot", "right_foot", "head", "left_wrist", "right_wrist")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inpaint a HumanML motion sequence from six known joint positions.",
    )
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--input_motion", type=Path, default=DEFAULT_INPUT_MOTION)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--raw_frames", type=int, default=DEFAULT_RAW_FRAMES)
    parser.add_argument("--max_frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--feet_thre", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--guidance_param", type=float, default=0.0)
    parser.add_argument("--text_condition", type=str, default="")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    return parser.parse_args()


def load_model_args(model_path: Path):
    args_path = model_path.resolve().parent / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"missing args.json next to checkpoint: {args_path}")
    with args_path.open("r", encoding="utf-8") as handle:
        return argparse.Namespace(**json.load(handle))


def load_raw_joints(path: Path, start_frame: int, raw_frames: int):
    if raw_frames < 2:
        raise ValueError("raw_frames must be at least 2.")
    joints = np.load(path)
    if joints.ndim != 3 or joints.shape[1:] != (22, 3):
        raise ValueError(f"expected input motion shaped as (T,22,3), got {tuple(joints.shape)}")

    end_frame = start_frame + raw_frames
    if start_frame < 0 or end_frame > joints.shape[0]:
        raise ValueError(
            f"requested frames [{start_frame}, {end_frame}) exceed available range [0, {joints.shape[0]})"
        )
    return joints[start_frame:end_frame].astype(np.float32, copy=False)


def build_dataset(max_frames: int):
    dataset = get_dataset(name="humanml", num_frames=max_frames, split="test", hml_mode="text_only")
    dataset.t2m_dataset.fixed_length = max_frames
    return dataset


def build_known_feature_mask():
    feature_mask = torch.zeros(263, dtype=torch.bool)
    feature_mask[0:4] = True
    feature_mask[31:34] = True
    feature_mask[34:37] = True
    feature_mask[46:49] = True
    feature_mask[61:64] = True
    feature_mask[64:67] = True
    return feature_mask


def hml_vec_to_xyz(motion_hml_vec):
    motion_tensor = torch.from_numpy(motion_hml_vec).float().unsqueeze(0).unsqueeze(1)
    xyz = recover_from_ric(motion_tensor, joints_num=humanml_utils.NUM_HML_JOINTS)
    xyz = xyz.view(-1, *xyz.shape[2:]).permute(0, 2, 3, 1)
    return xyz[0].permute(2, 0, 1).cpu().numpy()


def sample_to_xyz(sample, dataset, length: int):
    sample = sample.cpu().permute(0, 2, 3, 1)
    mean = torch.as_tensor(dataset.t2m_dataset.mean, dtype=sample.dtype).view(1, 1, 1, -1)
    std = torch.as_tensor(dataset.t2m_dataset.std, dtype=sample.dtype).view(1, 1, 1, -1)
    sample = (sample * std + mean).float()
    sample = recover_from_ric(sample, joints_num=humanml_utils.NUM_HML_JOINTS)
    sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)
    return sample[0].permute(2, 0, 1).cpu().numpy()[:length]


def build_inpainting_tensors(gt_hml_vec, mean, std, max_frames: int, device):
    length = int(gt_hml_vec.shape[0])
    if length > max_frames:
        raise ValueError(f"processed motion length {length} exceeds max_frames {max_frames}")

    gt_hml_vec_norm = (gt_hml_vec - mean) / std

    inpainted_motion = torch.zeros((1, 263, 1, max_frames), dtype=torch.float32, device=device)
    inpainted_motion[0, :, 0, :length] = torch.from_numpy(gt_hml_vec_norm.T).to(device=device, dtype=torch.float32)

    feature_mask = build_known_feature_mask().to(device=device)
    inpainting_mask = torch.zeros_like(inpainted_motion, dtype=torch.bool)
    inpainting_mask[0, :, 0, :length] = feature_mask[:, None].expand(-1, length)

    lengths = torch.tensor([length], device=device, dtype=torch.long)
    seq_mask = (
        torch.arange(max_frames, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    ).unsqueeze(1).unsqueeze(1)

    return inpainted_motion, inpainting_mask, seq_mask, lengths


def build_model_kwargs(text_condition, guidance_param, inpainted_motion, inpainting_mask, seq_mask, lengths, device):
    model_kwargs = {
        "y": {
            "mask": seq_mask,
            "lengths": lengths,
            "text": [text_condition],
            "inpainted_motion": inpainted_motion,
            "inpainting_mask": inpainting_mask,
        }
    }
    if guidance_param != 1:
        model_kwargs["y"]["scale"] = torch.ones(1, device=device, dtype=torch.float32) * guidance_param
    return model_kwargs


def load_model(model_path: Path, model_args, dataset, guidance_param: float):
    holder = SimpleNamespace(dataset=dataset)
    model, diffusion = create_model_and_diffusion(model_args, holder)
    state_dict = torch.load(model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)

    if guidance_param != 1:
        if getattr(model_args, "cond_mask_prob", 0) == 0:
            raise ValueError("guidance_param != 1 requires a checkpoint trained with cond_mask_prob > 0.")
        model = ClassifierFreeSampleModel(model)

    model.to(dist_util.dev())
    model.eval()
    return model, diffusion


def render_videos(output_dir: Path, stem: str, gt_motion, inpaint_motion, fps: int):
    gt_path = output_dir / f"{stem}_gt.mp4"
    inpaint_path = output_dir / f"{stem}_inpaint.mp4"
    compare_path = output_dir / f"{stem}_gt_vs_inpaint.mp4"
    skeleton = paramUtil.t2m_kinematic_chain

    plot_3d_motion(
        str(gt_path),
        skeleton,
        gt_motion,
        title="GT",
        dataset="humanml",
        fps=fps,
        vis_mode="gt",
    )
    plot_3d_motion(
        str(inpaint_path),
        skeleton,
        inpaint_motion,
        title="Inpainting",
        dataset="humanml",
        fps=fps,
    )

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "warning",
            "-i",
            str(gt_path),
            "-i",
            str(inpaint_path),
            "-filter_complex",
            "hstack=inputs=2",
            str(compare_path),
        ],
        check=True,
    )
    return gt_path, inpaint_path, compare_path


def compute_metrics(pred_xyz, gt_xyz, pred_sample, inpainted_motion, inpainting_mask, length: int):
    locked_joint_indices = [humanml_utils.HML_JOINT_NAMES.index(name) for name in LOCKED_JOINTS]
    feature_diff = (pred_sample.detach().cpu() - inpainted_motion.detach().cpu()).abs()
    feature_error = float(feature_diff[inpainting_mask.cpu()].max().item()) if inpainting_mask.any().item() else 0.0
    joint_error = float(np.abs(pred_xyz[:, locked_joint_indices] - gt_xyz[:, locked_joint_indices]).max())

    return {
        "length": int(length),
        "raw_frames_used": int(length + 1),
        "locked_joints": list(LOCKED_JOINTS),
        "locked_feature_dims_per_frame": int(inpainting_mask[0, :, 0, :length].sum().item() / max(length, 1)),
        "max_abs_feature_error_on_locked_dims": feature_error,
        "max_abs_xyz_error_on_locked_joints": joint_error,
    }


def main():
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"model checkpoint not found: {args.model_path}")
    if not args.input_motion.exists():
        raise FileNotFoundError(f"input motion not found: {args.input_motion}")

    fixseed(args.seed)
    dist_util.setup_dist(args.device)

    model_args = load_model_args(args.model_path)
    dataset = build_dataset(args.max_frames)
    model, diffusion = load_model(args.model_path, model_args, dataset, args.guidance_param)

    raw_joints = load_raw_joints(args.input_motion, args.start_frame, args.raw_frames)
    gt_hml_vec, _, _, _ = process_file(raw_joints, args.feet_thre)
    gt_hml_vec = gt_hml_vec.astype(np.float32, copy=False)
    length = int(gt_hml_vec.shape[0])

    inpainted_motion, inpainting_mask, seq_mask, lengths = build_inpainting_tensors(
        gt_hml_vec=gt_hml_vec,
        mean=dataset.t2m_dataset.mean,
        std=dataset.t2m_dataset.std,
        max_frames=args.max_frames,
        device=dist_util.dev(),
    )
    model_kwargs = build_model_kwargs(
        text_condition=args.text_condition,
        guidance_param=args.guidance_param,
        inpainted_motion=inpainted_motion,
        inpainting_mask=inpainting_mask,
        seq_mask=seq_mask,
        lengths=lengths,
        device=dist_util.dev(),
    )

    sample = diffusion.p_sample_loop(
        model,
        (1, model.njoints, model.nfeats, args.max_frames),
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=True,
        noise=None,
        const_noise=False,
    )

    gt_xyz = hml_vec_to_xyz(gt_hml_vec)
    pred_xyz = sample_to_xyz(sample, dataset, length)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.input_motion.stem}_first{args.raw_frames}"
    gt_path, inpaint_path, compare_path = render_videos(
        output_dir=args.output_dir,
        stem=stem,
        gt_motion=gt_xyz,
        inpaint_motion=pred_xyz,
        fps=args.fps,
    )

    metrics = compute_metrics(
        pred_xyz=pred_xyz,
        gt_xyz=gt_xyz,
        pred_sample=sample[:, :, :, :length],
        inpainted_motion=inpainted_motion[:, :, :, :length],
        inpainting_mask=inpainting_mask[:, :, :, :length],
        length=length,
    )

    np.save(
        args.output_dir / f"{stem}_results.npy",
        {
            "gt_xyz": gt_xyz,
            "pred_xyz": pred_xyz,
            "gt_hml_vec": gt_hml_vec,
            "length": length,
            "locked_joints": LOCKED_JOINTS,
        },
    )
    with (args.output_dir / f"{stem}_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"GT video: {gt_path}")
    print(f"Inpainting video: {inpaint_path}")
    print(f"Compare video: {compare_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
