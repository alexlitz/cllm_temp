"""
Chunk-generic SUB pipeline: 3 layers at any chunk size.

Layer 1: Raw diff + Generate (borrow) + Propagate (per-position)
Layer 2: Borrow lookahead (FlattenedFFN, cross-position)
Layer 3: Final result = (RAW_SUM - BORROW_IN + base) mod base

Parameterized by ChunkConfig.

For WORD (N=1): same CARRY_OUT-based approach as ADD to avoid
step-pair overflow. Layer 1 computes G = step(B > A) → CARRY_OUT,
layer 2 preserves it, layer 3 reads it to add base when underflow.
"""

import torch
import torch.nn as nn

from ..chunk_config import ChunkConfig
from .common import (
    GenericE, GenericPureFFN, GenericFlattenedFFN,
    bake_clear_pair,
)


class SubRawAndGenFFN(nn.Module):
    """Layer 1: RAW_SUM = A - B, G = step(B > A), P = step(A == B).

    7 hidden units (9 with clear_result), per-position.
    """

    def __init__(self, ge: GenericE, opcode: int, source_b=None, clear_result=False):
        super().__init__()
        S = ge.SCALE
        dim = ge.DIM
        dtype = ge.config.torch_dtype
        if source_b is None:
            source_b = ge.NIB_B

        hdim = 9 if clear_result else 7
        self.ffn = GenericPureFFN(dim, hidden_dim=hdim, dtype=dtype)

        with torch.no_grad():
            W_up = self.ffn.W_up
            b_up = self.ffn.b_up
            W_gate = self.ffn.W_gate
            W_down = self.ffn.W_down

            # Units 0-1: RAW_SUM = A - B (cancel pair)
            W_up[0, ge.OP_START + opcode] = S
            W_gate[0, ge.NIB_A] = 1.0
            W_gate[0, source_b] = -1.0
            W_down[ge.RAW_SUM, 0] = 1.0 / S

            W_up[1, ge.OP_START + opcode] = -S
            W_gate[1, ge.NIB_A] = -1.0
            W_gate[1, source_b] = 1.0
            W_down[ge.RAW_SUM, 1] = 1.0 / S

            # Units 2-3: G = step(B > A) = step(B - A >= 1)
            W_up[2, source_b] = S
            W_up[2, ge.NIB_A] = -S
            b_up[2] = 0.0
            W_gate[2, ge.OP_START + opcode] = 1.0
            W_down[ge.CARRY_OUT, 2] = 1.0 / S

            W_up[3, source_b] = S
            W_up[3, ge.NIB_A] = -S
            b_up[3] = -S * 1.0
            W_gate[3, ge.OP_START + opcode] = 1.0
            W_down[ge.CARRY_OUT, 3] = -1.0 / S

            # Units 4-6: P = step(A == B) via 3-unit merged approach
            W_up[4, ge.NIB_A] = S
            W_up[4, source_b] = -S
            b_up[4] = S * 1.0
            W_gate[4, ge.OP_START + opcode] = 1.0
            W_down[ge.TEMP, 4] = 1.0 / S

            W_up[5, ge.NIB_A] = S
            W_up[5, source_b] = -S
            b_up[5] = 0.0
            W_gate[5, ge.OP_START + opcode] = 1.0
            W_down[ge.TEMP, 5] = -2.0 / S

            W_up[6, ge.NIB_A] = S
            W_up[6, source_b] = -S
            b_up[6] = -S * 1.0
            W_gate[6, ge.OP_START + opcode] = 1.0
            W_down[ge.TEMP, 6] = 1.0 / S

            if clear_result:
                bake_clear_pair(self.ffn, 7, ge.OP_START + opcode, ge.RESULT, S)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class SubBorrowLookaheadFFN(nn.Module):
    """Layer 2: Parallel borrow-lookahead.

    For N=1 (WORD): only clear TEMP, preserve CARRY_OUT for layer 3.
    For N>1: prefix computation + clear both CARRY_OUT and TEMP.
    """

    def __init__(self, ge: GenericE, opcode: int):
        super().__init__()
        N = ge.NUM_POSITIONS
        S = ge.SCALE
        dtype = ge.config.torch_dtype

        if N == 1:
            hidden_dim = 2  # just clear TEMP
        else:
            hidden_dim = ge.config.carry_lookahead_hidden

        self.flat_ffn = GenericFlattenedFFN(ge, hidden_dim=hidden_dim, dtype=dtype)
        fi = self.flat_ffn._flat_idx

        with torch.no_grad():
            W_up = self.flat_ffn.W_up
            b_up = self.flat_ffn.b_up
            W_gate = self.flat_ffn.W_gate
            W_down = self.flat_ffn.W_down

            h = 0
            for i in range(1, N):
                W_up[h, fi(i - 1, ge.CARRY_OUT)] = S
                b_up[h] = -S * 0.5
                W_gate[h, fi(0, ge.OP_START + opcode)] = 1.0
                W_down[fi(i, ge.CARRY_IN), h] = 2.0 / S
                h += 1

                for j in range(i - 2, -1, -1):
                    n_vars = (i - 1 - j) + 1
                    for k in range(j + 1, i):
                        W_up[h, fi(k, ge.TEMP)] = S
                    W_up[h, fi(j, ge.CARRY_OUT)] = S
                    b_up[h] = -S * (n_vars - 0.5)
                    W_gate[h, fi(0, ge.OP_START + opcode)] = 1.0
                    W_down[fi(i, ge.CARRY_IN), h] = 2.0 / S
                    h += 1

            if N == 1:
                bake_clear_pair(self.flat_ffn.ffn, h,
                                fi(0, ge.OP_START + opcode),
                                fi(0, ge.TEMP), S)
                h += 2
            else:
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


class SubFinalResultFFN(nn.Module):
    """Layer 3: RESULT = (RAW_SUM - BORROW_IN + base) mod base.

    For N>1: step pair detects underflow.
    For N=1 (WORD): reads CARRY_OUT (borrow flag) from layer 1.
    """

    def __init__(self, ge: GenericE, opcode: int):
        super().__init__()
        S = ge.SCALE
        base = ge.BASE
        dim = ge.DIM
        dtype = ge.config.torch_dtype
        N = ge.NUM_POSITIONS

        self.ffn = GenericPureFFN(dim, hidden_dim=10, dtype=dtype)

        with torch.no_grad():
            W_up = self.ffn.W_up
            b_up = self.ffn.b_up
            W_gate = self.ffn.W_gate
            W_down = self.ffn.W_down

            # Units 0-1: Copy RAW_SUM to RESULT
            W_up[0, ge.OP_START + opcode] = S
            W_gate[0, ge.RAW_SUM] = 1.0
            W_down[ge.RESULT, 0] = 1.0 / S

            W_up[1, ge.OP_START + opcode] = -S
            W_gate[1, ge.RAW_SUM] = -1.0
            W_down[ge.RESULT, 1] = 1.0 / S

            # Units 2-3: Subtract CARRY_IN (borrow)
            W_up[2, ge.OP_START + opcode] = S
            W_gate[2, ge.CARRY_IN] = -1.0
            W_down[ge.RESULT, 2] = 1.0 / S

            W_up[3, ge.OP_START + opcode] = -S
            W_gate[3, ge.CARRY_IN] = 1.0
            W_down[ge.RESULT, 3] = 1.0 / S

            if N == 1:
                # N=1 (WORD): Read borrow flag from CARRY_OUT (set by layer 1).
                # CARRY_OUT = step(B > A). When 1, add base to correct underflow.
                # Cancel pair: silu(S*opcode) * CARRY_OUT * (base/S)
                # Product = S * base/S = base ≈ 4.3e9, safe in fp64.
                W_up[4, ge.OP_START + opcode] = S
                W_gate[4, ge.CARRY_OUT] = 1.0
                W_down[ge.RESULT, 4] = float(base) / S

                W_up[5, ge.OP_START + opcode] = -S
                W_gate[5, ge.CARRY_OUT] = -1.0
                W_down[ge.RESULT, 5] = float(base) / S
            else:
                # N>1: Step pair for underflow detection.
                # step(CARRY_IN - RAW_SUM >= 1) → add base
                W_up[4, ge.CARRY_IN] = S
                W_up[4, ge.RAW_SUM] = -S
                b_up[4] = 0.0
                W_gate[4, ge.OP_START + opcode] = 1.0
                W_down[ge.RESULT, 4] = float(base) / S

                W_up[5, ge.CARRY_IN] = S
                W_up[5, ge.RAW_SUM] = -S
                b_up[5] = -S * 1.0
                W_gate[5, ge.OP_START + opcode] = 1.0
                W_down[ge.RESULT, 5] = -float(base) / S

            # Units 6-7: Clear RAW_SUM
            bake_clear_pair(self.ffn, 6, ge.OP_START + opcode, ge.RAW_SUM, S)

            if N == 1:
                # Units 8-9: Clear CARRY_OUT (preserved through layer 2)
                bake_clear_pair(self.ffn, 8, ge.OP_START + opcode, ge.CARRY_OUT, S)
            else:
                # Units 8-9: Clear CARRY_IN
                bake_clear_pair(self.ffn, 8, ge.OP_START + opcode, ge.CARRY_IN, S)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


def build_sub_layers(config: ChunkConfig, opcode: int) -> nn.ModuleList:
    """Build 3-layer SUB pipeline for the given chunk config.

    Works at all 6 configs including WORD.
    """
    ge = GenericE(config)
    return nn.ModuleList([
        SubRawAndGenFFN(ge, opcode),
        SubBorrowLookaheadFFN(ge, opcode),
        SubFinalResultFFN(ge, opcode),
    ])
