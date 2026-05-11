"""Backward-compat re-export shim for the legacy ``migrated_ops`` module.

The 4800-line implementation that used to live here was split into a
per-layer package at ``c4_release/neural_vm/unified_compiler/ops/``
(2026-05-11). All factory names previously importable from this module
remain importable from this module via the star-import below, so existing
callers do not need to change.

New code should import directly from the per-layer modules:

    from c4_release.neural_vm.unified_compiler.ops.l5_ops import (
        make_layer5_fetch_op,
    )

See ``ops/__init__.py`` for the full list of available submodules.
"""

from .ops import *  # noqa: F401,F403
from .ops.all_core_ops import all_core_ops, all_hybrid_alu_wrap_ops  # noqa: F401

# Explicit re-export of underscore-prefixed helpers (skipped by `import *`).
# These were importable from the legacy module by name; preserve that surface.
from .ops.shared import (  # noqa: F401
    _as_setdim_proxy,
    _bake_post_op_into,
    _make_hybrid_alu_wrap_op,
    _ensure_l11_mul_module,
    _ALUShiftCompositeBuilder,
    _FlattenedDivModBuilder,
    _IO_REQUIRED_DIMS,
)
