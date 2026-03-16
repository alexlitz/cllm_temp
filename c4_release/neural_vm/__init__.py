"""
Neural VM - Autoregressive Neural Virtual Machine

Decoder-only transformer that executes VM instructions through set weights.
All computation flows through standard transformer layers.
NO Python arithmetic in forward passes.

Legacy non-autoregressive components are in neural_vm/archive/.
"""

from .embedding import E, Opcode

# Core layers
from .base_layers import PureFFN, PureAttention, bake_weights

# Autoregressive VM model
from .vm_step import (
    AutoregressiveVM, Token, TransformerBlock,
    CausalSelfAttention, AutoregressiveAttention,
    set_vm_weights, _SetDim,
)

# VM Runner (autoregressive generation loop)
from .run_vm import AutoregressiveVMRunner, ToolCall, ToolResponse

# DraftVM for speculative execution
from .speculative import DraftVM

# Ultra-Fast Batch Runner (massive batches + speculation)
from .batch_runner_v2 import UltraBatchRunner, UltraBatchRunnerCached, run_batch_ultra

# softmax1 (ZFOD attention)
from .kv_cache_eviction import softmax1

# Dimension registry & contract validation
from .dim_registry import (
    DimRegistry, DimSlot, LayerIO, ContractValidator,
    build_default_registry, build_default_contracts, validate_default,
)


__all__ = [
    # Core
    'E', 'Opcode',
    'PureFFN', 'PureAttention', 'bake_weights',
    # Autoregressive VM
    'AutoregressiveVM', 'AutoregressiveVMRunner', 'ToolCall', 'ToolResponse',
    'DraftVM',
    'UltraBatchRunner', 'UltraBatchRunnerCached', 'run_batch_ultra',
    'Token', 'TransformerBlock',
    'CausalSelfAttention', 'AutoregressiveAttention',
    'set_vm_weights', '_SetDim',
    # Utilities
    'softmax1',
    # Dim registry
    'DimRegistry', 'DimSlot', 'LayerIO', 'ContractValidator',
    'build_default_registry', 'build_default_contracts', 'validate_default',
]
