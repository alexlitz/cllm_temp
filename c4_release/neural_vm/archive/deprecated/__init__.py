"""
Deprecated Implementations for Neural VM.

This folder contains implementations that should NOT be used:
- impure_ops.py: Operations using .sum(), .clone(), .expand() (not pure neural)

These are kept ONLY for:
1. Historical reference
2. Understanding why they were deprecated
3. Testing/comparison purposes

DO NOT USE IN PRODUCTION.
"""

from .impure_ops import (
    MultiHeadGatherAttention,
    GatherAttentionPure,
)

__all__ = [
    'MultiHeadGatherAttention',
    'GatherAttentionPure',
]
