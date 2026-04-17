"""
Modular Weight Setting for Neural VM.

Organizes weight setting by functional concern:
- embedding: Token embeddings
- threshold: Threshold attention (L0-L2)
- carry_forward: Register carry-forward (L3)
- fetch: Instruction fetch (L5)
- alu: Arithmetic/logic operations (L8-L12)
- memory: Memory operations (L14-L15)
- function_calls: JSR/ENT/LEV/LEA (distributed)
- io: I/O operations

Each module provides:
- set_weights(model, config) - Main entry point
- diagnose(model, input) - Diagnostic output
- test_isolation(model) - Test module in isolation
"""

from .base import WeightModule, WeightConfig
from .embedding import EmbeddingWeights
from .threshold import ThresholdWeights
from .carry_forward import CarryForwardWeights
from .fetch import FetchWeights
from .alu import ALUWeights
from .memory import MemoryWeights
from .function_calls import FunctionCallWeights

__all__ = [
    "WeightModule",
    "WeightConfig",
    "EmbeddingWeights",
    "ThresholdWeights",
    "CarryForwardWeights",
    "FetchWeights",
    "ALUWeights",
    "MemoryWeights",
    "FunctionCallWeights",
]
