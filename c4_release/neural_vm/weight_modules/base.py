"""
Base classes for modular weight setting.

Provides common interfaces and utilities for weight modules.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import torch


@dataclass
class WeightConfig:
    """Configuration for weight setting."""

    # Model dimensions
    d_model: int = 512
    n_layers: int = 16
    n_heads: int = 8
    ffn_hidden: int = 4096

    # Derived
    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    # Scale factors
    swiglu_scale: float = 100.0  # SwiGLU scale (S)
    attention_scale: float = 15.0  # Attention scale (L)
    alibi_slope: float = 10.0  # ALiBi slope

    # Thresholds
    opcode_threshold: float = 4.0
    carry_threshold: float = 1.5

    # Anti-leakage
    anti_leak_gate: float = -5000.0

    # Features
    enable_tool_calling: bool = False
    enable_conversational_io: bool = False
    alu_mode: str = 'lookup'


@dataclass
class DiagnosticResult:
    """Result of weight module diagnostic."""

    module_name: str
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class WeightModule(ABC):
    """Base class for weight setting modules.

    Each module handles a specific functional concern (embedding, fetch, ALU, etc.)
    and provides methods for:
    - Setting weights
    - Diagnosing issues
    - Testing in isolation
    """

    def __init__(self, config: WeightConfig):
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Module name for diagnostics."""
        pass

    @property
    @abstractmethod
    def layers(self) -> List[int]:
        """Layers this module affects."""
        pass

    @property
    @abstractmethod
    def dimensions(self) -> List[int]:
        """Dimension ranges this module uses."""
        pass

    @abstractmethod
    def set_weights(self, model) -> None:
        """Set weights into the model.

        Args:
            model: AutoregressiveVM instance
        """
        pass

    def diagnose(
        self,
        model,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> DiagnosticResult:
        """Diagnose potential issues with this module.

        Args:
            model: Model to diagnose
            hidden_states: Optional input for testing activations

        Returns:
            DiagnosticResult with findings
        """
        result = DiagnosticResult(module_name=self.name, passed=True)

        # Check layer indices are valid
        for layer in self.layers:
            if layer < 0 or layer >= len(model.blocks):
                result.errors.append(f"Invalid layer index: {layer}")
                result.passed = False

        return result

    def test_isolation(
        self,
        model,
        test_input: torch.Tensor,
    ) -> DiagnosticResult:
        """Test this module in isolation.

        Args:
            model: Model with only this module's weights set
            test_input: Input tensor

        Returns:
            DiagnosticResult with test findings
        """
        result = DiagnosticResult(module_name=self.name, passed=True)

        # Run forward pass
        with torch.no_grad():
            try:
                output = model(test_input)
                result.details["output_shape"] = list(output.shape)
            except Exception as e:
                result.errors.append(f"Forward pass failed: {e}")
                result.passed = False

        return result


def get_dimension_registry():
    """Get the dimension registry for weight setting.

    Returns the _SetDim class from vm_step.py for dimension constants.
    """
    from neural_vm.vm_step import _SetDim
    return _SetDim


def zero_layer_weights(model, layer_idx: int):
    """Zero all weights in a layer for isolation testing."""
    block = model.blocks[layer_idx]

    # Zero attention weights
    block.attn.W_q.zero_()
    block.attn.W_k.zero_()
    block.attn.W_v.zero_()
    block.attn.W_o.zero_()

    # Zero FFN weights
    block.ffn.W_up.zero_()
    block.ffn.W_gate.zero_()
    block.ffn.W_down.zero_()
    if hasattr(block.ffn, 'bias_up'):
        block.ffn.bias_up.zero_()


def copy_layer_weights(src_model, dst_model, layer_idx: int):
    """Copy weights from one model to another for a specific layer."""
    src_block = src_model.blocks[layer_idx]
    dst_block = dst_model.blocks[layer_idx]

    with torch.no_grad():
        # Copy attention weights
        dst_block.attn.W_q.copy_(src_block.attn.W_q)
        dst_block.attn.W_k.copy_(src_block.attn.W_k)
        dst_block.attn.W_v.copy_(src_block.attn.W_v)
        dst_block.attn.W_o.copy_(src_block.attn.W_o)

        # Copy FFN weights
        dst_block.ffn.W_up.copy_(src_block.ffn.W_up)
        dst_block.ffn.W_gate.copy_(src_block.ffn.W_gate)
        dst_block.ffn.W_down.copy_(src_block.ffn.W_down)
        if hasattr(src_block.ffn, 'bias_up'):
            dst_block.ffn.bias_up.copy_(src_block.ffn.bias_up)
