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

# IRBuilder/CompilerIR are part of an in-flight Phase 0 redesign that lives on a
# parallel branch. Import them when available; fall back gracefully so this
# package stays importable on branches that don't ship the IR module yet.
try:
    from .builder import BuilderConfig, PruningConfig, IRBuilder  # noqa: F401
    from .ir import CompilerIR, AttentionOp, FFNOp, LayerSpec  # noqa: F401
    _HAS_IR = True
except ImportError:
    _HAS_IR = False

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
if _HAS_IR:
    __all__ += [
        'BuilderConfig',
        'PruningConfig',
        'IRBuilder',
        'CompilerIR',
        'AttentionOp',
        'FFNOp',
        'LayerSpec',
    ]
