import os
from pathlib import Path


def _repo_root():
    return Path(__file__).resolve().parents[1]


def _parse_env_line(line):
    line = line.strip()
    if not line or line.startswith('#'):
        return None
    if line.startswith('export '):
        line = line[len('export '):].strip()
    if '=' not in line:
        return None
    key, value = line.split('=', 1)
    key = key.strip()
    value = value.strip().strip('"').strip("'")
    if not key:
        return None
    return key, value


def load_env_file(env_file=None):
    env_path = Path(env_file) if env_file else _repo_root() / '.env'
    if not env_path.exists():
        return env_path

    with env_path.open('r', encoding='utf-8') as handle:
        for line in handle:
            parsed = _parse_env_line(line)
            if parsed is None:
                continue
            key, value = parsed
            os.environ.setdefault(key, value)
    return env_path


def _arg_or_env(args, attr_name, env_name, default=None):
    arg_value = getattr(args, attr_name, None) if args is not None else None
    if arg_value not in (None, ''):
        return arg_value
    return os.environ.get(env_name, default)


def init_wandb_run(save_dir, args=None):
    try:
        import wandb
    except ImportError as exc:
        raise ImportError('wandb is not installed in the active environment.') from exc

    env_file = getattr(args, 'env_file', None) if args is not None else None
    env_path = load_env_file(env_file=env_file)

    api_key = os.environ.get('WANDB_API_KEY')
    if not api_key:
        raise RuntimeError(
            f'WANDB_API_KEY is required when using WandbPlatform. '
            f'Checked env file: {env_path}'
        )

    entity = _arg_or_env(args, 'wandb_entity', 'WANDB_ENTITY')
    project = _arg_or_env(args, 'wandb_project', 'WANDB_PROJECT', default="MDM-Post")
    mode = _arg_or_env(args, 'wandb_mode', 'WANDB_MODE', default='online')
    group = _arg_or_env(args, 'wandb_group', 'WANDB_RUN_GROUP', default=None)
    tags_raw = _arg_or_env(args, 'wandb_tags', 'WANDB_TAGS', default=None)
    run_name = _arg_or_env(args, 'wandb_run_name', 'WANDB_RUN_NAME')
    tags = None
    if tags_raw:
        tags = [tag.strip() for tag in tags_raw.split(',') if tag.strip()]

    if not project:
        raise RuntimeError(
            f'WANDB_PROJECT is required when using WandbPlatform. '
            f'Checked env file: {env_path}'
        )
    if not run_name:
        raise RuntimeError(
            'wandb_run_name is required when using WandbPlatform. '
            'Pass --wandb_run_name explicitly.'
        )

    wandb.login(key=api_key)

    run = wandb.init(
        project=project,
        entity=entity,
        dir=os.path.abspath(save_dir),
        name=run_name,
        mode=mode,
        group=group,
        tags=tags,
        reinit=False,
    )
    return run
