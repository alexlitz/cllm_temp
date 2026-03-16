"""
Chunk-generic CMP pipeline: 3 layers at any chunk size.

Layer 1: Raw diff + Generate (borrow) + Propagate (per-position)
Layer 2: Borrow lookahead + final borrow → RESULT[0] (flattened)
Layer 3: Clear RAW_SUM/CARRY_IN, optionally invert RESULT[0] (flattened)

LT: swap=False, invert=False  → borrow(A-B) = step(A < B)
GT: swap=True,  invert=False  → borrow(B-A) = step(A > B)
LE: swap=True,  invert=True   → 1 - step(A > B) = step(A ≤ B)
GE: swap=False, invert=True   → 1 - step(A < B) = step(A ≥ B)

For WORD (N=1): same CARRY_OUT-based approach as SUB.
"""

import torch
import torch.nn as nn

from ..chunk_config import ChunkConfig
from .common import (
    GenericE, GenericPureFFN, GenericFlattenedFFN,
    bake_clear_pair,
)


class CmpRawDiffAndGenFFN(nn.Module):
    """Layer 1: RAW_SUM = first-second, G=step(second>first), P=step(first==second).

    7 hidden units, per-position.
    swap=False: first=NIB_A, second=NIB_B (for LT, GE)
    swap=True:  first=NIB_B, second=NIB_A (for GT, LE)
    """

    def __init__(self, ge: GenericE, opcode: int, swap: bool = False):
        super().__init__()
        S = ge.SCALE
        dim = ge.DIM
        dtype = ge.config.torch_dtype

        first = ge.NIB_B if swap else ge.NIB_A
        second = ge.NIB_A if swap else ge.NIB_B

        self.ffn = GenericPureFFN(dim, hidden_dim=7, dtype=dtype)

        with torch.no_grad():
            W_up = self.ffn.W_up
            b_up = self.ffn.b_up
            W_gate = self.ffn.W_gate
            W_down = self.ffn.W_down

            # Units 0-1: RAW_SUM = first - second (cancel pair)
            W_up[0, ge.OP_START + opcode] = S
            W_gate[0, first] = 1.0
            W_gate[0, second] = -1.0
            W_down[ge.RAW_SUM, 0] = 1.0 / S

            W_up[1, ge.OP_START + opcode] = -S
            W_gate[1, first] = -1.0
            W_gate[1, second] = 1.0
            W_down[ge.RAW_SUM, 1] = 1.0 / S

            # Units 2-3: G = step(second > first) = step(second - first >= 1)
            W_up[2, second] = S
            W_up[2, first] = -S
            b_up[2] = 0.0
            W_gate[2, ge.OP_START + opcode] = 1.0
            W_down[ge.CARRY_OUT, 2] = 1.0 / S

            W_up[3, second] = S
            W_up[3, first] = -S
            b_up[3] = -S * 1.0
            W_gate[3, ge.OP_START + opcode] = 1.0
            W_down[ge.CARRY_OUT, 3] = -1.0 / S

            # Units 4-6: P = step(first == second) via 3-unit merged
            W_up[4, first] = S
            W_up[4, second] = -S
            b_up[4] = S * 1.0
            W_gate[4, ge.OP_START + opcode] = 1.0
            W_down[ge.TEMP, 4] = 1.0 / S

            W_up[5, first] = S
            W_up[5, second] = -S
            b_up[5] = 0.0
            W_gate[5, ge.OP_START + opcode] = 1.0
            W_down[ge.TEMP, 5] = -2.0 / S

            W_up[6, first] = S
            W_up[6, second] = -S
            b_up[6] = -S * 1.0
            W_gate[6, ge.OP_START + opcode] = 1.0
            W_down[ge.TEMP, 6] = 1.0 / S

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class CmpBorrowLookaheadFFN(nn.Module):
    """Layer 2: Borrow lookahead + final borrow → RESULT[0].

    For N>1: prefix computation C[1]..C[N-1] → CARRY_IN, C[N] → RESULT[0].
    For N=1: copy CARRY_OUT → RESULT[0], clear TEMP and CARRY_OUT.
    """

    def __init__(self, ge: GenericE, opcode: int):
        super().__init__()
        N = ge.NUM_POSITIONS
        S = ge.SCALE
        dtype = ge.config.torch_dtype

        if N == 1:
            hidden_dim = 5  # 1 AND-gate + 2 clear TEMP + 2 clear CARRY_OUT
        else:
            # carry chain: N*(N-1)/2, final borrow: N, clearing: 4*N
            hidden_dim = N * (N - 1) // 2 + N + 4 * N

        self.flat_ffn = GenericFlattenedFFN(ge, hidden_dim=hidden_dim, dtype=dtype)
        fi = self.flat_ffn._flat_idx

        with torch.no_grad():
            W_up = self.flat_ffn.W_up
            b_up = self.flat_ffn.b_up
            W_gate = self.flat_ffn.W_gate
            W_down = self.flat_ffn.W_down

            h = 0

            if N == 1:
                # C[1] = G[0] → RESULT[0] (AND gate: step(G >= 0.5))
                W_up[h, fi(0, ge.CARRY_OUT)] = S
                b_up[h] = -S * 0.5
                W_gate[h, fi(0, ge.OP_START + opcode)] = 1.0
                W_down[fi(0, ge.RESULT), h] = 2.0 / S
                h += 1

                # Clear TEMP
                bake_clear_pair(self.flat_ffn.ffn, h,
                                fi(0, ge.OP_START + opcode),
                                fi(0, ge.TEMP), S)
                h += 2

                # Clear CARRY_OUT
                bake_clear_pair(self.flat_ffn.ffn, h,
                                fi(0, ge.OP_START + opcode),
                                fi(0, ge.CARRY_OUT), S)
                h += 2
            else:
                # Prefix carry computation C[1]..C[N-1] → CARRY_IN
                for i in range(1, N):
                    # G[i-1] directly
                    W_up[h, fi(i - 1, ge.CARRY_OUT)] = S
                    b_up[h] = -S * 0.5
                    W_gate[h, fi(0, ge.OP_START + opcode)] = 1.0
                    W_down[fi(i, ge.CARRY_IN), h] = 2.0 / S
                    h += 1

                    # Multi-var AND terms: P[i-1]*...*P[j+1]*G[j]
                    for j in range(i - 2, -1, -1):
                        n_vars = (i - 1 - j) + 1
                        for k in range(j + 1, i):
                            W_up[h, fi(k, ge.TEMP)] = S
                        W_up[h, fi(j, ge.CARRY_OUT)] = S
                        b_up[h] = -S * (n_vars - 0.5)
                        W_gate[h, fi(0, ge.OP_START + opcode)] = 1.0
                        W_down[fi(i, ge.CARRY_IN), h] = 2.0 / S
                        h += 1

                # Final borrow C[N] → RESULT[0]
                # G[N-1] directly
                W_up[h, fi(N - 1, ge.CARRY_OUT)] = S
                b_up[h] = -S * 0.5
                W_gate[h, fi(0, ge.OP_START + opcode)] = 1.0
                W_down[fi(0, ge.RESULT), h] = 2.0 / S
                h += 1

                # Multi-var AND terms for final borrow
                for j in range(N - 2, -1, -1):
                    n_vars = (N - 1 - j) + 1
                    for k in range(j + 1, N):
                        W_up[h, fi(k, ge.TEMP)] = S
                    W_up[h, fi(j, ge.CARRY_OUT)] = S
                    b_up[h] = -S * (n_vars - 0.5)
                    W_gate[h, fi(0, ge.OP_START + opcode)] = 1.0
                    W_down[fi(0, ge.RESULT), h] = 2.0 / S
                    h += 1

                # Clear G (CARRY_OUT) and P (TEMP) at all N positions
                for pos in range(N):
                    bake_clear_pair(self.flat_ffn.ffn, h,
                                    fi(pos, ge.OP_START + opcode),
                                    fi(pos, ge.CARRY_OUT), S)
                    h += 2
                    bake_clear_pair(self.flat_ffn.ffn, h,
                                    fi(pos, ge.OP_START + opcode),
                                    fi(pos, ge.TEMP), S)
                    h += 2

            assert h <= hidden_dim, f"Used {h} hidden units, allocated {hidden_dim}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flat_ffn(x)


class CmpClearFFN(nn.Module):
    """Layer 3: Clear RAW_SUM and CARRY_IN, optionally invert RESULT[0].

    For invert (LE/GE): RESULT[0] = 1 - RESULT[0].
    """

    def __init__(self, ge: GenericE, opcode: int, invert: bool = False):
        super().__init__()
        N = ge.NUM_POSITIONS
        S = ge.SCALE
        dtype = ge.config.torch_dtype

        if N == 1:
            hidden_dim = 2 + (4 if invert else 0)
        else:
            hidden_dim = 4 * N + (4 if invert else 0)

        self.flat_ffn = GenericFlattenedFFN(ge, hidden_dim=hidden_dim, dtype=dtype)
        fi = self.flat_ffn._flat_idx

        with torch.no_grad():
            h = 0

            # Clear RAW_SUM at all positions
            for pos in range(N):
                bake_clear_pair(self.flat_ffn.ffn, h,
                                fi(pos, ge.OP_START + opcode),
                                fi(pos, ge.RAW_SUM), S)
                h += 2

            if N > 1:
                # Clear CARRY_IN at all positions
                for pos in range(N):
                    bake_clear_pair(self.flat_ffn.ffn, h,
                                    fi(pos, ge.OP_START + opcode),
                                    fi(pos, ge.CARRY_IN), S)
                    h += 2

            if invert:
                result_idx = fi(0, ge.RESULT)
                opcode_idx = fi(0, ge.OP_START + opcode)

                # Negate: RESULT += -2*RESULT → RESULT = -RESULT
                self.flat_ffn.W_up[h, opcode_idx] = S
                self.flat_ffn.W_gate[h, result_idx] = -2.0
                self.flat_ffn.W_down[result_idx, h] = 1.0 / S
                h += 1

                self.flat_ffn.W_up[h, opcode_idx] = -S
                self.flat_ffn.W_gate[h, result_idx] = 2.0
                self.flat_ffn.W_down[result_idx, h] = 1.0 / S
                h += 1

                # Add 1: opcode * opcode → RESULT
                self.flat_ffn.W_up[h, opcode_idx] = S
                self.flat_ffn.W_gate[h, opcode_idx] = 1.0
                self.flat_ffn.W_down[result_idx, h] = 1.0 / S
                h += 1

                self.flat_ffn.W_up[h, opcode_idx] = -S
                self.flat_ffn.W_gate[h, opcode_idx] = -1.0
                self.flat_ffn.W_down[result_idx, h] = 1.0 / S
                h += 1

            assert h <= hidden_dim, f"Used {h} hidden units, allocated {hidden_dim}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flat_ffn(x)


def build_cmp_layers(config: ChunkConfig, opcode: int,
                     swap: bool = False, invert: bool = False) -> nn.ModuleList:
    """Build 3-layer CMP pipeline for the given chunk config."""
    ge = GenericE(config)
    return nn.ModuleList([
        CmpRawDiffAndGenFFN(ge, opcode, swap=swap),
        CmpBorrowLookaheadFFN(ge, opcode),
        CmpClearFFN(ge, opcode, invert=invert),
    ])


def build_lt_layers(config: ChunkConfig, opcode: int = 19) -> nn.ModuleList:
    return build_cmp_layers(config, opcode, swap=False, invert=False)


def build_gt_layers(config: ChunkConfig, opcode: int = 20) -> nn.ModuleList:
    return build_cmp_layers(config, opcode, swap=True, invert=False)


def build_le_layers(config: ChunkConfig, opcode: int = 21) -> nn.ModuleList:
    return build_cmp_layers(config, opcode, swap=True, invert=True)


def build_ge_layers(config: ChunkConfig, opcode: int = 22) -> nn.ModuleList:
    return build_cmp_layers(config, opcode, swap=False, invert=True)
