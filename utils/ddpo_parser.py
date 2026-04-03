import json
from argparse import ArgumentParser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAVE_ROOT = REPO_ROOT / "save"


def _load_checkpoint_args(model_path: str):
    model_path = Path(model_path).expanduser().resolve()
    args_path = model_path.parent / "args.json"
    if not args_path.is_file():
        raise FileNotFoundError(f"Checkpoint args.json not found: {args_path}")
    with args_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_ddpo_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--data_root", required=True, type=str)
    parser.add_argument("--save_dir", default="", type=str)
    parser.add_argument("--run_name", default="", type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--algo", default="reinforce", choices=["reinforce", "actor_critic"], type=str)

    parser.add_argument("--prompt_batch_size", default=4, type=int)
    parser.add_argument("--max_train_samples", default=0, type=int)
    parser.add_argument(
        "--train_sampling_mode",
        default="uniform",
        choices=["uniform", "failure_only", "mixed"],
        type=str,
    )
    parser.add_argument("--failure_cases_file", default="", type=str)
    parser.add_argument(
        "--failure_cases_split",
        default="train",
        choices=["train", "test", "val"],
        type=str,
    )
    parser.add_argument("--failure_batch_size", default=0, type=int)
    parser.add_argument("--failure_eval_cases_file", default="", type=str)
    parser.add_argument("--num_outer_steps", default=200, type=int)
    parser.add_argument("--num_inner_epochs", default=4, type=int)
    parser.add_argument("--ppo_minibatch_size", default=2, type=int)
    parser.add_argument("--eval_interval", default=0, type=int)
    parser.add_argument("--eval_prompt_batch_size", default=32, type=int)
    parser.add_argument("--eval_split", default="test", choices=["train", "test", "val"], type=str)

    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--clip_range", default=1e-4, type=float)
    parser.add_argument("--adv_clip_max", default=5.0, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--actor_num_epochs", default=0, type=int)
    parser.add_argument("--critic_lr", default=5e-4, type=float)
    parser.add_argument("--critic_weight_decay", default=0.0, type=float)
    parser.add_argument("--critic_hidden_dim", default=256, type=int)
    parser.add_argument("--critic_dropout", default=0.0, type=float)
    parser.add_argument("--critic_num_epochs", default=3, type=int)
    parser.add_argument("--critic_minibatch_size", default=256, type=int)
    parser.add_argument("--kl_coef", default=0.0, type=float)
    parser.add_argument("--gae_gamma", default=0.99, type=float)
    parser.add_argument("--gae_lambda", default=0.95, type=float)
    parser.add_argument("--ft_denoising_steps", default=0, type=int)

    parser.add_argument("--guidance_param", default=2.5, type=float)
    parser.add_argument("--max_motion_frames", default=196, type=int)

    parser.add_argument("--lora_rank", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16.0, type=float)
    parser.add_argument("--lora_layer_scope", default="all", choices=["all", "last3"], type=str)

    parser.add_argument("--phc_config_path", default="/home/gxy/hay-thesis/PHC/output/HumanoidIm/phc_kp_mcp_iccv/.hydra/config.yaml", type=str)
    parser.add_argument("--phc_actor_ckpt", default="/home/gxy/hay-thesis/PHC/output/HumanoidIm/phc_kp_mcp_iccv/Humanoid.pth", type=str)
    parser.add_argument("--phc_num_envs", default=2, type=int)
    parser.add_argument("--phc_max_steps", default=512, type=int)

    parser.add_argument("--reward_mode", default="refine", choices=["refine", "failure"], type=str)
    parser.add_argument("--reward_assignment", default="sequence", choices=["sequence", "chunk"], type=str)
    parser.add_argument("--reward_chunk_size", default=14, type=int)
    parser.add_argument("--reward_chunk_reduce", default="mean", choices=["mean"], type=str)
    parser.add_argument("--chunk_mean_weight", default=None, type=float)
    parser.add_argument("--chunk_q10_weight", default=None, type=float)
    parser.add_argument("--chunk_success_weight", default=None, type=float)
    parser.add_argument("--chunk_fail_weight", default=None, type=float)
    parser.add_argument("--chunk_fail_prev_weight", default=None, type=float)
    parser.add_argument("--chunk_early_weight", default=None, type=float)
    parser.add_argument("--chunk_early_prev_weight", default=None, type=float)
    parser.add_argument("--reward_fail_penalty", default=None, type=float)
    parser.add_argument("--reward_return_mean_weight", default=None, type=float)
    parser.add_argument("--reward_q10_weight", default=None, type=float)
    parser.add_argument("--reward_early_term_weight", default=None, type=float)
    parser.add_argument("--reward_success_weight", default=None, type=float)
    parser.add_argument("--reward_scale", default=1.0, type=float)
    parser.add_argument("--reward_success_bonus", default=0.5, type=float)
    parser.add_argument("--reward_terminate_penalty", default=0.5, type=float)
    parser.add_argument("--prompt_stat_buffer_size", default=16, type=int)
    parser.add_argument("--prompt_stat_min_count", default=16, type=int)
    parser.add_argument(
        "--prompt_baseline_mode",
        default="zscore",
        choices=["zscore", "ema", "history_zscore"],
        type=str,
    )
    parser.add_argument("--prompt_ema_alpha", default=0.05, type=float)

    parser.add_argument("--wandb_project", default="mdm-phc-ddpo", type=str)
    parser.add_argument("--wandb_run_name", default="", type=str)
    parser.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"], type=str)

    parser.add_argument("--log_interval", default=1, type=int)
    parser.add_argument("--save_interval", default=10, type=int)
    parser.add_argument("--sample_progress", action="store_true")
    parser.add_argument("--fixed_sampling_seed", action="store_true")
    parser.add_argument("--debug_prompt_advantages", action="store_true")
    parser.add_argument("--debug_prompt_advantages_limit", default=4, type=int)
    parser.add_argument("--resume_lora_path", default="", type=str)
    return parser


def parse_ddpo_args(argv=None):
    parser = build_ddpo_parser()
    args = parser.parse_args(argv)

    if args.run_name:
        if not args.save_dir:
            args.save_dir = str((DEFAULT_SAVE_ROOT / args.run_name).resolve())
        if not args.wandb_run_name:
            args.wandb_run_name = args.run_name
    elif not args.save_dir:
        parser.error("Either --save_dir or --run_name must be provided.")

    checkpoint_args = _load_checkpoint_args(args.model_path)
    for key in [
        "dataset",
        "data_dir",
        "arch",
        "emb_trans_dec",
        "layers",
        "latent_dim",
        "cond_mask_prob",
        "lambda_rcxyz",
        "lambda_vel",
        "lambda_fc",
        "unconstrained",
        "noise_schedule",
        "diffusion_steps",
        "sigma_small",
    ]:
        if key in checkpoint_args:
            setattr(args, key, checkpoint_args[key])

    if getattr(args, "cond_mask_prob", 0.0) == 0:
        args.guidance_param = 1.0
    if args.actor_num_epochs <= 0:
        args.actor_num_epochs = args.num_inner_epochs
    if args.ft_denoising_steps < 0:
        parser.error("--ft_denoising_steps must be non-negative.")
    if args.reward_assignment == "chunk" and args.algo != "actor_critic":
        parser.error("--reward_assignment=chunk currently requires --algo actor_critic.")
    if args.reward_assignment == "chunk" and args.reward_chunk_size <= 0:
        parser.error("--reward_chunk_size must be positive when --reward_assignment=chunk.")
    if args.train_sampling_mode != "uniform" and not args.failure_cases_file:
        parser.error("--failure_cases_file is required when --train_sampling_mode is not uniform.")
    if args.train_sampling_mode == "mixed":
        if args.failure_batch_size <= 0:
            parser.error("--failure_batch_size must be positive when --train_sampling_mode=mixed.")
        if args.failure_batch_size >= args.prompt_batch_size:
            parser.error("--failure_batch_size must be smaller than --prompt_batch_size.")
    return args
