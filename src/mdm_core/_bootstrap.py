from __future__ import annotations

import sys
from pathlib import Path


_TOP_LEVEL_MODULES = (
    "utils",
    "data_loaders",
    "model",
    "diffusion",
    "sample",
    "mdm_talker",
)


def _is_target_module(module_name: str) -> bool:
    for name in _TOP_LEVEL_MODULES:
        if module_name == name or module_name.startswith(f"{name}."):
            return True
    return False


def activate_import_context(repo_root: Path | None = None) -> Path:
    """Expose repo modules under `mdm_core.*` and prioritize this repo on sys.path."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    repo_root = repo_root.resolve()
    repo_root_str = str(repo_root)
    src_root_str = str(repo_root / "src")

    for p in (repo_root_str, src_root_str):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)

    # Allow importing existing repo modules as mdm_core.<module>.
    pkg = sys.modules.get("mdm_core")
    if pkg is not None and hasattr(pkg, "__path__"):
        pkg_path = pkg.__path__
        if repo_root_str not in pkg_path:
            pkg_path.append(repo_root_str)

    for mod_name, mod in list(sys.modules.items()):
        if not _is_target_module(mod_name):
            continue
        mod_file = str(getattr(mod, "__file__", "") or "")
        if repo_root_str and repo_root_str in mod_file:
            continue
        sys.modules.pop(mod_name, None)

    return repo_root
