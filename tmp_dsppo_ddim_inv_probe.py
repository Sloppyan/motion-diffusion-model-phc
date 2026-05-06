try:
    import isaacgym  # noqa: F401
except ImportError:
    isaacgym = None

import random
from types import SimpleNamespace
import torch
from train.dsppo_rl.runtime_ddim import DSPPODDIMInversionRuntime
from train.dsppo_rl.noise_policy_mean import FrameNoiseMeanPolicy
from train.dppo_frame_rl.data import build_prompt_entry_pools, sample_prompt_batch

args = SimpleNamespace(
    model_path='/home/gxy/hay-thesis/motion-diffusion-model-phc/save/humanml_enc_512_50steps/model000750000.pt',
    data_root='/home/gxy/hay-thesis/HumanML3D/HumanML3D',
    device_id=0,
    max_motion_frames=196,
    deterministic_denoising=True,
    stochastic_first_k_steps=0,
    guidance_param=2.5,
    phc_num_envs=1,
    phc_max_steps=160,
    phc_actor_ckpt='/home/gxy/hay-thesis/PHC/output/HumanoidIm/phc_kp_mcp_iccv/Humanoid.pth',
    ddim_eta=0.0,
    frame_gamma=0.995,
    frame_lambda=0.95,
    dense_reward_weight=1.0,
    reward_norm=True,
    success_bonus=0.0,
    fail_penalty=-0.2,
    pose_reward_weight=1.0,
    pose_reward_alpha=5.0,
    velocity_reward_weight=0.0,
    velocity_reward_alpha=0.5,
)

runtime = DSPPODDIMInversionRuntime(args)
policy = FrameNoiseMeanPolicy(
    text_dim=runtime.actor.clip_dim,
    hidden_dim=256,
    action_dim=runtime.actor.njoints * runtime.actor.nfeats,
    max_frames=196,
).to(runtime.device)

pools = build_prompt_entry_pools(
    data_root=args.data_root,
    split='train',
    max_motion_frames=args.max_motion_frames,
    failure_cases_file='/home/gxy/hay-thesis/PHC/output/eval_mdm/mdm_eval_train_full/falling_cases.txt',
)
entries, _ = sample_prompt_batch(pools, 'failure_only', 32, random.Random(0), 1.0, 1.0)

sampled = runtime._generate_batch(entries=entries, noise_policy=policy, policy_deterministic=False, sampling_seed=0)
gt = runtime._load_gt_inversion_noise_batch(entries)
mask = torch.arange(args.max_motion_frames, device=runtime.device).unsqueeze(0) < sampled['lengths_20fps'].unsqueeze(1)

z = sampled['frame_noise']
diff = z - gt

stats = {
    'sample_noise_sqnorm_mean': float((z.pow(2).sum(dim=-1)[mask]).mean().item()),
    'gt_inv_sqnorm_mean': float((gt.pow(2).sum(dim=-1)[mask]).mean().item()),
    'diff_sqnorm_mean': float((diff.pow(2).sum(dim=-1)[mask]).mean().item()),
    'per_dim_mse': float(diff.pow(2)[mask].mean().item()),
    'sample_abs_mean': float(z.abs()[mask].mean().item()),
    'gt_inv_abs_mean': float(gt.abs()[mask].mean().item()),
}
print(stats)
runtime.close()
