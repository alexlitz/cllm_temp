"""
Unified Weight Setter Interface

Provides a single entry point for setting VM weights with mode selection:
- HAND_SET: Uses set_vm_weights() - production ready, 2000+ tests passing
- COMPILED: Uses set_compiled_weights() - development, compiler-generated

Both modes must:
1. Pass the same functional tests
2. Pass purity tests (no Python arithmetic in forward pass)
3. Produce equivalent outputs for the same inputs

Usage:
    from neural_vm.weight_setter import WeightMode, set_weights

    # Use hand-set weights (default, production)
    set_weights(model, mode=WeightMode.HAND_SET)

    # Use compiled weights (experimental)
    set_weights(model, mode=WeightMode.COMPILED)

    # Use environment variable
    NEURAL_VM_WEIGHT_MODE=compiled python test_opcodes.py
"""

import os
from enum import Enum
from typing import Optional

import torch


class WeightMode(Enum):
    """Weight setting mode."""
    HAND_SET = "hand_set"      # Manual weight setting (production)
    COMPILED = "compiled"      # Compiler-generated weights (development)


# Default mode from environment or HAND_SET
_DEFAULT_MODE = WeightMode(
    os.environ.get("NEURAL_VM_WEIGHT_MODE", "hand_set")
)


def get_default_mode() -> WeightMode:
    """Get the default weight mode from environment."""
    return _DEFAULT_MODE


def set_default_mode(mode: WeightMode):
    """Set the default weight mode."""
    global _DEFAULT_MODE
    _DEFAULT_MODE = mode


def set_weights(
    model,
    mode: Optional[WeightMode] = None,
    enable_tool_calling: bool = False,
    enable_conversational_io: bool = False,
    alu_mode: str = 'lookup',
    verify_purity: bool = True,
):
    """Set VM weights using the specified mode.

    Args:
        model: AutoregressiveVM instance
        mode: Weight mode (HAND_SET or COMPILED). Defaults to environment/HAND_SET.
        enable_tool_calling: Enable tool calling support
        enable_conversational_io: Enable conversational I/O
        alu_mode: ALU mode ('lookup' or 'efficient')
        verify_purity: If True, verify forward pass purity after setting weights

    Raises:
        NotImplementedError: If COMPILED mode requested but not yet implemented
        PurityViolationError: If purity verification fails
    """
    if mode is None:
        mode = _DEFAULT_MODE

    if mode == WeightMode.HAND_SET:
        _set_hand_weights(
            model,
            enable_tool_calling=enable_tool_calling,
            enable_conversational_io=enable_conversational_io,
            alu_mode=alu_mode,
        )
    elif mode == WeightMode.COMPILED:
        _set_compiled_weights(
            model,
            enable_tool_calling=enable_tool_calling,
            enable_conversational_io=enable_conversational_io,
            alu_mode=alu_mode,
        )
    else:
        raise ValueError(f"Unknown weight mode: {mode}")

    # Verify purity after setting weights
    if verify_purity:
        verify_forward_purity(model)


def _set_hand_weights(
    model,
    enable_tool_calling: bool = False,
    enable_conversational_io: bool = False,
    alu_mode: str = 'lookup',
):
    """Set weights using hand-crafted approach (production)."""
    from .vm_step import set_vm_weights
    set_vm_weights(
        model,
        enable_tool_calling=enable_tool_calling,
        enable_conversational_io=enable_conversational_io,
        alu_mode=alu_mode,
    )


def _set_compiled_weights(
    model,
    enable_tool_calling: bool = False,
    enable_conversational_io: bool = False,
    alu_mode: str = 'lookup',
):
    """Set weights using compiler-generated approach.

    Uses UnifiedVMCompiler to generate all weights for the Neural VM.
    The compiled approach produces weights that:
    1. Pass all tests that hand-set weights pass
    2. Maintain forward pass purity
    3. Support the same features (tool calling, conversational I/O, etc.)

    Args:
        model: AutoregressiveVM instance
        enable_tool_calling: Enable tool calling support
        enable_conversational_io: Enable conversational I/O
        alu_mode: ALU mode ('lookup' or 'efficient')
    """
    from .unified_compiler import UnifiedVMCompiler

    # Create compiler with model parameters
    compiler = UnifiedVMCompiler(
        d_model=model.d_model,
        n_layers=len(model.blocks),
        n_heads=model.blocks[0].attn.num_heads,
        ffn_hidden=model.blocks[0].ffn.W_up.shape[0],
        alu_mode=alu_mode,
    )

    # Compile all weights into the model
    compiler.compile(
        model,
        enable_tool_calling=enable_tool_calling,
        enable_conversational_io=enable_conversational_io,
    )


def verify_forward_purity(model):
    """Verify the model's forward pass is pure (no Python tensor modifications).

    This ensures both weight setting approaches maintain autoregressive purity.
    """
    from .purity_guard import verify_forward_purity as _verify
    return _verify(model)


# =============================================================================
# Convenience functions for testing
# =============================================================================

def with_weight_mode(mode: WeightMode):
    """Context manager for temporarily changing weight mode.

    Usage:
        with with_weight_mode(WeightMode.COMPILED):
            runner = BatchedSpeculativeRunner()  # Uses compiled weights
    """
    class WeightModeContext:
        def __init__(self, new_mode):
            self.new_mode = new_mode
            self.old_mode = None

        def __enter__(self):
            self.old_mode = get_default_mode()
            set_default_mode(self.new_mode)
            return self

        def __exit__(self, *args):
            set_default_mode(self.old_mode)

    return WeightModeContext(mode)


def parametrize_weight_modes():
    """Get list of weight modes for test parametrization.

    Returns modes that are actually implemented and can be tested.
    """
    return [WeightMode.HAND_SET, WeightMode.COMPILED]


# =============================================================================
# Test utilities
# =============================================================================

def compare_weight_outputs(
    model_hand: 'AutoregressiveVM',
    model_compiled: 'AutoregressiveVM',
    test_inputs: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> bool:
    """Compare outputs from hand-set and compiled weight models.

    Used to validate that compiled weights produce equivalent outputs.

    Args:
        model_hand: Model with hand-set weights
        model_compiled: Model with compiled weights
        test_inputs: Input tensor to test
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        True if outputs match within tolerance
    """
    model_hand.eval()
    model_compiled.eval()

    with torch.no_grad():
        out_hand = model_hand(test_inputs)
        out_compiled = model_compiled(test_inputs)

    return torch.allclose(out_hand, out_compiled, rtol=rtol, atol=atol)
