"""
Unified Weight Compiler for Neural VM.

Provides a declarative API for generating transformer weights that implement
the C4 VM. Produces identical weights to vm_step.py's set_vm_weights() function.

Usage:
    from neural_vm.unified_compiler import UnifiedVMCompiler, Verifier

    compiler = UnifiedVMCompiler()
    compiler.compile(model)

    # Verify against manual implementation
    verifier = Verifier()
    assert verifier.compare_models(manual_model, compiled_model)
"""

from .primitives import Primitives
from .compiler import UnifiedVMCompiler
from .verification import Verifier
from .layer_compiler import LayerCompiler, Operation, ModelLayout, build_model_from_layout

# Optional imports: builder.py, ir.py, and opcodes.py are not tracked on every
# branch (e.g. fresh worktrees off main). Guard so unified_compiler still imports
# without these modules; downstream code that needs them can import directly.
try:
    from .builder import BuilderConfig, PruningConfig, IRBuilder  # noqa: F401
except ImportError:
    pass

try:
    from .ir import CompilerIR, AttentionOp, FFNOp, LayerSpec  # noqa: F401
except ImportError:
    pass

try:
    from .opcodes import *  # noqa: F401,F403
except ImportError:
    pass

__all__ = [
    'Primitives',
    'UnifiedVMCompiler',
    'Verifier',
    # Phase 0 layer-allocation compiler MVP (2026-05-09)
    'LayerCompiler',
    'Operation',
    'ModelLayout',
    'build_model_from_layout',
]

# Extend __all__ with optionally-imported names if they were successfully bound.
for _name in (
    'BuilderConfig', 'PruningConfig', 'IRBuilder',
    'CompilerIR', 'AttentionOp', 'FFNOp', 'LayerSpec',
):
    if _name in globals():
        __all__.append(_name)
del _name
