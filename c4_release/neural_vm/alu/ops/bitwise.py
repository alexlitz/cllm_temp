"""
Chunk-generic bitwise AND/OR/XOR pipeline.

For BIT config (k=1): single layer, NIB_A/NIB_B are already bits.
For other configs: 2 layers:
  Layer 1: Extract all k bits from NIB_A and NIB_B into temp slots.
  Layer 2: Combine bits using AND/OR/XOR logic + clear temp slots.

Bit j extraction uses alternating step functions:
  bit_j = step(v >= 1*2^j) - step(v >= 2*2^j) + step(v >= 3*2^j) - ...
  Number of terms = 2^(k-j) - 1

Supported configs: BIT, PAIR, NIBBLE, BYTE (k <= 8).
HALFWORD/WORD impractical (2^k - 1 step pairs per operand per bit).
"""

import torch
import torch.nn as nn

from ..chunk_config import ChunkConfig
from .common import (
    GenericE, GenericPureFFN, GenericFlattenedFFN,
    bake_clear_pair,
)

# Bit storage slots: use high opcode range (40..40+2k-1)
# Safe since bitwise opcodes are 14 (OR), 15 (XOR), 16 (AND),
# so OP_START+opcode ∈ {21,22,23}, well below 40.
BIT_SLOT_BASE_A = 40
BIT_SLOT_BASE_B = 56  # 40 + 16, enough for k up to 16


def _bit_a_slot(bit_idx):
    return BIT_SLOT_BASE_A + bit_idx


def _bit_b_slot(bit_idx):
    return BIT_SLOT_BASE_B + bit_idx


class BitExtractFFN(nn.Module):
    """Extract all k bits from NIB_A and NIB_B into temp slots.

    For each bit j (0=LSB to k-1=MSB):
      bit_j(v) = sum_{m=1}^{M} (-1)^(m+1) * step(v >= m*2^j)
      where M = 2^(k-j) - 1

    Each step(v >= threshold) is a step pair (2 hidden units).
    Per-position FFN.
    """

    def __init__(self, ge: GenericE, opcode: int):
        super().__init__()
        k = ge.config.chunk_bits
        S = ge.SCALE
        dim = ge.DIM
        dtype = ge.config.torch_dtype

        # Count hidden units: for each bit j, each operand, M step pairs
        total_pairs = 0
        for j in range(k):
            M = (1 << (k - j)) - 1
            total_pairs += M
        hidden_dim = total_pairs * 2 * 2  # * 2 operands * 2 units per pair

        self.ffn = GenericPureFFN(dim, hidden_dim=hidden_dim, dtype=dtype)

        with torch.no_grad():
            h = 0
            for operand, source_slot, slot_fn in [
                ('A', ge.NIB_A, _bit_a_slot),
                ('B', ge.NIB_B, _bit_b_slot),
            ]:
                for j in range(k):
                    M = (1 << (k - j)) - 1
                    step_size = 1 << j  # 2^j
                    dest_slot = slot_fn(j)

                    for m in range(1, M + 1):
                        threshold = m * step_size
                        sign = 1.0 if (m % 2 == 1) else -1.0

                        # Rise at threshold-1
                        self.ffn.W_up[h, source_slot] = S
                        self.ffn.b_up[h] = -S * (threshold - 1.0)
                        self.ffn.W_gate[h, ge.OP_START + opcode] = 1.0
                        self.ffn.W_down[dest_slot, h] = sign / S
                        h += 1

                        # Saturation at threshold
                        self.ffn.W_up[h, source_slot] = S
                        self.ffn.b_up[h] = -S * float(threshold)
                        self.ffn.W_gate[h, ge.OP_START + opcode] = 1.0
                        self.ffn.W_down[dest_slot, h] = -sign / S
                        h += 1

            assert h == hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class BitAndCombineClearFFN(nn.Module):
    """AND combine + clear bit slots.

    AND: result_bit_j = a_bit_j * b_bit_j → RESULT += 2^j * product
    Uses cancel pair per bit. Then clears bit slots.
    """

    def __init__(self, ge: GenericE, opcode: int):
        super().__init__()
        k = ge.config.chunk_bits
        S = ge.SCALE
        dim = ge.DIM
        dtype = ge.config.torch_dtype

        # 2 per bit (AND cancel pair) + 2*2k (clear bit slots)
        hidden_dim = 2 * k + 4 * k

        self.ffn = GenericPureFFN(dim, hidden_dim=hidden_dim, dtype=dtype)

        with torch.no_grad():
            h = 0

            for j in range(k):
                weight = float(1 << j)  # 2^j
                a_slot = _bit_a_slot(j)
                b_slot = _bit_b_slot(j)

                # Cancel pair: a * b * weight → RESULT
                self.ffn.W_up[h, a_slot] = S
                self.ffn.W_gate[h, b_slot] = 1.0
                self.ffn.W_down[ge.RESULT, h] = weight / S
                h += 1

                self.ffn.W_up[h, a_slot] = -S
                self.ffn.W_gate[h, b_slot] = -1.0
                self.ffn.W_down[ge.RESULT, h] = weight / S
                h += 1

            # Clear bit slots
            for j in range(k):
                bake_clear_pair(self.ffn, h, ge.OP_START + opcode, _bit_a_slot(j), S)
                h += 2
            for j in range(k):
                bake_clear_pair(self.ffn, h, ge.OP_START + opcode, _bit_b_slot(j), S)
                h += 2

            assert h == hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class BitOrCombineClearFFN(nn.Module):
    """OR combine + clear bit slots.

    OR: a + b - a*b → RESULT += 2^j * (a + b - a*b)
    Uses: 1 unit for +a, 1 unit for +b, 2 units for -a*b (cancel pair).
    """

    def __init__(self, ge: GenericE, opcode: int):
        super().__init__()
        k = ge.config.chunk_bits
        S = ge.SCALE
        dim = ge.DIM
        dtype = ge.config.torch_dtype

        hidden_dim = 4 * k + 4 * k  # 4 per bit + clearing

        self.ffn = GenericPureFFN(dim, hidden_dim=hidden_dim, dtype=dtype)

        with torch.no_grad():
            h = 0

            for j in range(k):
                weight = float(1 << j)
                a_slot = _bit_a_slot(j)
                b_slot = _bit_b_slot(j)

                # +a: silu(S*opcode) * a_bit * weight/S
                self.ffn.W_up[h, ge.OP_START + opcode] = S
                self.ffn.W_gate[h, a_slot] = 1.0
                self.ffn.W_down[ge.RESULT, h] = weight / S
                h += 1

                # +b: silu(S*opcode) * b_bit * weight/S
                self.ffn.W_up[h, ge.OP_START + opcode] = S
                self.ffn.W_gate[h, b_slot] = 1.0
                self.ffn.W_down[ge.RESULT, h] = weight / S
                h += 1

                # -a*b: cancel pair
                self.ffn.W_up[h, a_slot] = S
                self.ffn.W_gate[h, b_slot] = 1.0
                self.ffn.W_down[ge.RESULT, h] = -weight / S
                h += 1

                self.ffn.W_up[h, a_slot] = -S
                self.ffn.W_gate[h, b_slot] = -1.0
                self.ffn.W_down[ge.RESULT, h] = -weight / S
                h += 1

            # Clear bit slots
            for j in range(k):
                bake_clear_pair(self.ffn, h, ge.OP_START + opcode, _bit_a_slot(j), S)
                h += 2
            for j in range(k):
                bake_clear_pair(self.ffn, h, ge.OP_START + opcode, _bit_b_slot(j), S)
                h += 2

            assert h == hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class BitXorCombineClearFFN(nn.Module):
    """XOR combine + clear bit slots.

    XOR: a + b - 2*a*b → RESULT += 2^j * (a + b - 2*a*b)
    """

    def __init__(self, ge: GenericE, opcode: int):
        super().__init__()
        k = ge.config.chunk_bits
        S = ge.SCALE
        dim = ge.DIM
        dtype = ge.config.torch_dtype

        hidden_dim = 4 * k + 4 * k

        self.ffn = GenericPureFFN(dim, hidden_dim=hidden_dim, dtype=dtype)

        with torch.no_grad():
            h = 0

            for j in range(k):
                weight = float(1 << j)
                a_slot = _bit_a_slot(j)
                b_slot = _bit_b_slot(j)

                # +a
                self.ffn.W_up[h, ge.OP_START + opcode] = S
                self.ffn.W_gate[h, a_slot] = 1.0
                self.ffn.W_down[ge.RESULT, h] = weight / S
                h += 1

                # +b
                self.ffn.W_up[h, ge.OP_START + opcode] = S
                self.ffn.W_gate[h, b_slot] = 1.0
                self.ffn.W_down[ge.RESULT, h] = weight / S
                h += 1

                # -2*a*b: cancel pair with weight -2
                self.ffn.W_up[h, a_slot] = S
                self.ffn.W_gate[h, b_slot] = 1.0
                self.ffn.W_down[ge.RESULT, h] = -2.0 * weight / S
                h += 1

                self.ffn.W_up[h, a_slot] = -S
                self.ffn.W_gate[h, b_slot] = -1.0
                self.ffn.W_down[ge.RESULT, h] = -2.0 * weight / S
                h += 1

            # Clear bit slots
            for j in range(k):
                bake_clear_pair(self.ffn, h, ge.OP_START + opcode, _bit_a_slot(j), S)
                h += 2
            for j in range(k):
                bake_clear_pair(self.ffn, h, ge.OP_START + opcode, _bit_b_slot(j), S)
                h += 2

            assert h == hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


# --- BIT config (k=1) special case: single-layer, no extraction ---

class BitAndDirectFFN(nn.Module):
    """AND for BIT config: a*b → RESULT. 2 hidden units."""

    def __init__(self, ge: GenericE, opcode: int):
        super().__init__()
        S = ge.SCALE
        dtype = ge.config.torch_dtype
        self.ffn = GenericPureFFN(ge.DIM, hidden_dim=2, dtype=dtype)

        with torch.no_grad():
            self.ffn.W_up[0, ge.NIB_A] = S
            self.ffn.W_gate[0, ge.NIB_B] = 1.0
            self.ffn.W_down[ge.RESULT, 0] = 1.0 / S

            self.ffn.W_up[1, ge.NIB_A] = -S
            self.ffn.W_gate[1, ge.NIB_B] = -1.0
            self.ffn.W_down[ge.RESULT, 1] = 1.0 / S

    def forward(self, x):
        return self.ffn(x)


class BitOrDirectFFN(nn.Module):
    """OR for BIT config: a + b - a*b → RESULT. 4 hidden units."""

    def __init__(self, ge: GenericE, opcode: int):
        super().__init__()
        S = ge.SCALE
        dtype = ge.config.torch_dtype
        self.ffn = GenericPureFFN(ge.DIM, hidden_dim=4, dtype=dtype)

        with torch.no_grad():
            # +a
            self.ffn.W_up[0, ge.OP_START + opcode] = S
            self.ffn.W_gate[0, ge.NIB_A] = 1.0
            self.ffn.W_down[ge.RESULT, 0] = 1.0 / S
            # +b
            self.ffn.W_up[1, ge.OP_START + opcode] = S
            self.ffn.W_gate[1, ge.NIB_B] = 1.0
            self.ffn.W_down[ge.RESULT, 1] = 1.0 / S
            # -a*b cancel pair
            self.ffn.W_up[2, ge.NIB_A] = S
            self.ffn.W_gate[2, ge.NIB_B] = 1.0
            self.ffn.W_down[ge.RESULT, 2] = -1.0 / S

            self.ffn.W_up[3, ge.NIB_A] = -S
            self.ffn.W_gate[3, ge.NIB_B] = -1.0
            self.ffn.W_down[ge.RESULT, 3] = -1.0 / S

    def forward(self, x):
        return self.ffn(x)


class BitXorDirectFFN(nn.Module):
    """XOR for BIT config: a + b - 2*a*b → RESULT. 4 hidden units."""

    def __init__(self, ge: GenericE, opcode: int):
        super().__init__()
        S = ge.SCALE
        dtype = ge.config.torch_dtype
        self.ffn = GenericPureFFN(ge.DIM, hidden_dim=4, dtype=dtype)

        with torch.no_grad():
            # +a
            self.ffn.W_up[0, ge.OP_START + opcode] = S
            self.ffn.W_gate[0, ge.NIB_A] = 1.0
            self.ffn.W_down[ge.RESULT, 0] = 1.0 / S
            # +b
            self.ffn.W_up[1, ge.OP_START + opcode] = S
            self.ffn.W_gate[1, ge.NIB_B] = 1.0
            self.ffn.W_down[ge.RESULT, 1] = 1.0 / S
            # -2*a*b cancel pair
            self.ffn.W_up[2, ge.NIB_A] = S
            self.ffn.W_gate[2, ge.NIB_B] = 1.0
            self.ffn.W_down[ge.RESULT, 2] = -2.0 / S

            self.ffn.W_up[3, ge.NIB_A] = -S
            self.ffn.W_gate[3, ge.NIB_B] = -1.0
            self.ffn.W_down[ge.RESULT, 3] = -2.0 / S

    def forward(self, x):
        return self.ffn(x)


# --- Builders ---

def build_and_layers(config: ChunkConfig, opcode: int = 16) -> nn.ModuleList:
    ge = GenericE(config)
    if config.chunk_bits == 1:
        return nn.ModuleList([BitAndDirectFFN(ge, opcode)])
    return nn.ModuleList([
        BitExtractFFN(ge, opcode),
        BitAndCombineClearFFN(ge, opcode),
    ])


def build_or_layers(config: ChunkConfig, opcode: int = 14) -> nn.ModuleList:
    ge = GenericE(config)
    if config.chunk_bits == 1:
        return nn.ModuleList([BitOrDirectFFN(ge, opcode)])
    return nn.ModuleList([
        BitExtractFFN(ge, opcode),
        BitOrCombineClearFFN(ge, opcode),
    ])


def build_xor_layers(config: ChunkConfig, opcode: int = 15) -> nn.ModuleList:
    ge = GenericE(config)
    if config.chunk_bits == 1:
        return nn.ModuleList([BitXorDirectFFN(ge, opcode)])
    return nn.ModuleList([
        BitExtractFFN(ge, opcode),
        BitXorCombineClearFFN(ge, opcode),
    ])
