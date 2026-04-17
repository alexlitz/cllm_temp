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

__all__ = [
    'Primitives',
    'UnifiedVMCompiler',
    'Verifier',
]
