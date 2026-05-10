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

# Builder/IR re-exports require optional companion modules (`builder.py`,
# `ir.py`) that are absent from this commit's tree. The package must still
# import cleanly so `compile_full_vm` (the production entry point) is usable;
# guard the optional imports and only re-export them when available.
from .primitives import Primitives
from .compiler import UnifiedVMCompiler
from .verification import Verifier
from .layer_compiler import LayerCompiler, Operation, ModelLayout, build_model_from_layout

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

try:
    from .builder import BuilderConfig, PruningConfig, IRBuilder  # noqa: F401
    from .ir import CompilerIR, AttentionOp, FFNOp, LayerSpec  # noqa: F401
    __all__ += [
        'BuilderConfig', 'PruningConfig', 'IRBuilder',
        'CompilerIR', 'AttentionOp', 'FFNOp', 'LayerSpec',
    ]
except ImportError:
    pass
