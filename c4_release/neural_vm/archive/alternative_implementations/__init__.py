"""
Alternative Implementations for Neural VM.

This folder contains valid alternative implementations that are not the default:
- attention_based_ops.py: Attention-based cross-nibble operations (default is FFN-based)
- rope_binary_io.py: RoPE-based binary addressing (alternative to ALiBi)

These are kept for:
1. Comparison and benchmarking
2. Research into different architectures
3. Potential future use cases where attention may be preferred
"""

from .attention_based_ops import (
    CompareReduceEqAttention,
    CompareReduceNeAttention,
    CmpBroadcastResultAttention,
    BranchConditionAttention,
    BzReduceAttention,
    BnzReduceAttention,
    McmpReduceAttention,
)

__all__ = [
    'CompareReduceEqAttention',
    'CompareReduceNeAttention',
    'CmpBroadcastResultAttention',
    'BranchConditionAttention',
    'BzReduceAttention',
    'BnzReduceAttention',
    'McmpReduceAttention',
]
