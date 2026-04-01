import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from train.train_ddpo_phc_rl import main as ddpo_rl_main


def _ensure_legacy_defaults(argv):
    argv = list(argv)
    has_algo = any(arg == "--algo" or arg.startswith("--algo=") for arg in argv)
    if not has_algo:
        argv.extend(["--algo", "reinforce"])
    return argv


def main():
    ddpo_rl_main(_ensure_legacy_defaults(sys.argv[1:]))


if __name__ == "__main__":
    main()
