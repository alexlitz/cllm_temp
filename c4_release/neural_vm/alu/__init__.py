"""
Chunk-generic ALU framework with multi-precision support.

Provides parameterized ALU operations that work at any chunk size
from 1-bit to 32-bit, using the same algorithms with different
thresholds and precision requirements.
"""

from .chunk_config import (
    ChunkConfig, BIT, PAIR, NIBBLE, BYTE, HALFWORD, WORD, ALL_CONFIGS,
)
from .ops import (
    build_add_layers, build_sub_layers, build_div_layers,
    build_mul_layers, build_mod_layers,
    build_lt_layers, build_gt_layers, build_le_layers, build_ge_layers,
    build_cmp_layers,
    build_and_layers, build_or_layers, build_xor_layers,
    build_shl_layers, build_shr_layers,
)
