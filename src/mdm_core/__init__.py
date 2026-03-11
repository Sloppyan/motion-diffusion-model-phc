"""MDM Core namespace package.

This package exposes the original repository layout under `mdm_core.*`
to avoid ambiguous top-level imports like `utils` in host applications.
"""

from ._bootstrap import activate_import_context

activate_import_context()

__all__ = ["activate_import_context"]
