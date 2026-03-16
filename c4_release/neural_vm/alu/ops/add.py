"""
Chunk-generic ADD pipeline: 3 layers at any chunk size.

Layer 1: Raw sum + Generate + Propagate (per-position PureFFN)
Layer 2: Carry lookahead (FlattenedFFN, cross-position)
Layer 3: Final result = (RAW_SUM + CARRY_IN) mod base

Parameterized by ChunkConfig — same algorithm, different thresholds.

Precision analysis for the step pair in layer 3:
  Step pair computes step(RAW_SUM >= base) × (-base).
  Max silu output ≈ S × base, W_down = base/S, product ≈ base².
  Safe when base² < 2^53 (fp64 limit), i.e. base < 2^26.5.
  All configs except WORD satisfy this.

  For WORD (base=2^32, N=1): layer 1 already computed the overflow flag
  G = step(A+B >= base) → CARRY_OUT. Layer 3 reads CARRY_OUT directly
  instead of recomputing via step pair. Product = S × base/S = base ≈ 4.3e9.
"""

import torch
import torch.nn as nn

from ..chunk_config import ChunkConfig
from .common import (
    GenericE, GenericPureFFN, GenericFlattenedFFN,
    bake_cancel_pair, bake_step_pair, bake_clear_pair,
)


class AddRawAndGenFFN(nn.Module):
    """Layer 1: Compute RAW_SUM = A + B, G = step(A+B >= base), P = step(A+B == base-1).

    6 hidden units, per-position.
    """

    def __init__(self, ge: GenericE, opcode: int):
        super().__init__()
        S = ge.SCALE
        base = ge.BASE
        dim = ge.DIM
        dtype = ge.config.torch_dtype

        self.ffn = GenericPureFFN(dim, hidden_dim=6, dtype=dtype)

        with torch.no_grad():
            W_up = self.ffn.W_up
            b_up = self.ffn.b_up
            W_gate = self.ffn.W_gate
            W_down = self.ffn.W_down

            # Units 0-1: RAW_SUM = A + B (cancel pair)
            W_up[0, ge.NIB_A] = S
            W_up[0, ge.NIB_B] = S
            W_gate[0, ge.OP_START + opcode] = S
            W_down[ge.RAW_SUM, 0] = 1.0 / (S * S)

            W_up[1, ge.NIB_A] = -S
            W_up[1, ge.NIB_B] = -S
            W_gate[1, ge.OP_START + opcode] = -S
            W_down[ge.RAW_SUM, 1] = 1.0 / (S * S)

            # Units 2-3: G = step(A+B >= base) → CARRY_OUT
            W_up[2, ge.NIB_A] = S
            W_up[2, ge.NIB_B] = S
            b_up[2] = -S * (base - 1.0)
            W_gate[2, ge.OP_START + opcode] = 1.0
            W_down[ge.CARRY_OUT, 2] = 1.0 / S

            W_up[3, ge.NIB_A] = S
            W_up[3, ge.NIB_B] = S
            b_up[3] = -S * float(base)
            W_gate[3, ge.OP_START + opcode] = 1.0
            W_down[ge.CARRY_OUT, 3] = -1.0 / S

            # Units 4-5: P = step(A+B == base-1) → TEMP
            # P = step(>=base-1) - step(>=base)
            W_up[4, ge.NIB_A] = S
            W_up[4, ge.NIB_B] = S
            b_up[4] = -S * (base - 2.0)
            W_gate[4, ge.OP_START + opcode] = 1.0
            W_down[ge.TEMP, 4] = 1.0 / S

            W_up[5, ge.NIB_A] = S
            W_up[5, ge.NIB_B] = S
            b_up[5] = -S * (base - 1.0)
            W_gate[5, ge.OP_START + opcode] = 1.0
            W_down[ge.TEMP, 5] = -1.0 / S

            # Also subtract G from TEMP: P = step(>=base-1) - step(>=base) = step(==base-1)
            W_down[ge.TEMP, 2] = -1.0 / S
            W_down[ge.TEMP, 3] = 1.0 / S

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class AddCarryLookaheadFFN(nn.Module):
    """Layer 2: Parallel carry-lookahead using prefix computation.

    C[i] = G[i-1] OR (P[i-1] AND G[i-2]) OR ...
    N*(N-1)/2 AND-gate units + clearing units.

    For N=1 (WORD): no AND-gates, only clear TEMP (preserve CARRY_OUT for layer 3).
    For N>1: clear both CARRY_OUT and TEMP after prefix computation.
    """

    def __init__(self, ge: GenericE, opcode: int):
        super().__init__()
        N = ge.NUM_POSITIONS
        S = ge.SCALE
        dtype = ge.config.torch_dtype

        # For N=1: only clear TEMP (2 units). Preserve CARRY_OUT for final layer.
        # For N>1: N*(N-1)/2 AND-gates + 4*N clearing (CARRY_OUT + TEMP)
        if N == 1:
            hidden_dim = 2  # just clear TEMP at pos 0
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

            # Prefix carry computation (empty for N=1)
            for i in range(1, N):
                # Term 0: G[i-1] directly (1-var AND)
                W_up[h, fi(i - 1, ge.CARRY_OUT)] = S
                b_up[h] = -S * 0.5
                W_gate[h, fi(0, ge.OP_START + opcode)] = 1.0
                W_down[fi(i, ge.CARRY_IN), h] = 2.0 / S
                h += 1

                # Multi-var AND terms: P[i-1]*...*P[j+1]*G[j]
                for j in range(i - 2, -1, -1):
                    n_vars = (i - 1 - j) + 1
                    for k in range(j + 1, i):
                        W_up[h, fi(k, ge.TEMP)] = S  # P[k]
                    W_up[h, fi(j, ge.CARRY_OUT)] = S  # G[j]
                    b_up[h] = -S * (n_vars - 0.5)
                    W_gate[h, fi(0, ge.OP_START + opcode)] = 1.0
                    W_down[fi(i, ge.CARRY_IN), h] = 2.0 / S
                    h += 1

            if N == 1:
                # Only clear TEMP; preserve CARRY_OUT for final layer
                bake_clear_pair(self.flat_ffn.ffn, h,
                                fi(0, ge.OP_START + opcode),
                                fi(0, ge.TEMP), S)
                h += 2
            else:
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


class AddFinalResultFFN(nn.Module):
    """Layer 3: RESULT = (RAW_SUM + CARRY_IN) mod base.

    For N>1: step pair detects overflow (product ≈ base², safe when base < 2^26).
    For N=1 (WORD): reads CARRY_OUT from layer 1 (product ≈ base, always safe).
    """

    def __init__(self, ge: GenericE, opcode: int):
        super().__init__()
        S = ge.SCALE
        base = ge.BASE
        dim = ge.DIM
        dtype = ge.config.torch_dtype
        N = ge.NUM_POSITIONS

        # N=1 needs 2 extra units to clear CARRY_OUT (replaces clearing CARRY_IN)
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

            # Units 2-3: Add CARRY_IN to RESULT
            W_up[2, ge.OP_START + opcode] = S
            W_gate[2, ge.CARRY_IN] = 1.0
            W_down[ge.RESULT, 2] = 1.0 / S

            W_up[3, ge.OP_START + opcode] = -S
            W_gate[3, ge.CARRY_IN] = -1.0
            W_down[ge.RESULT, 3] = 1.0 / S

            if N == 1:
                # N=1 (WORD): Read overflow flag from CARRY_OUT (set by layer 1).
                # Cancel pair: silu(S*opcode) * CARRY_OUT * (-base/S) → RESULT
                # Product = S * base/S = base ≈ 4.3e9, well within fp64.
                W_up[4, ge.OP_START + opcode] = S
                W_gate[4, ge.CARRY_OUT] = 1.0
                W_down[ge.RESULT, 4] = -float(base) / S

                W_up[5, ge.OP_START + opcode] = -S
                W_gate[5, ge.CARRY_OUT] = -1.0
                W_down[ge.RESULT, 5] = -float(base) / S
            else:
                # N>1: Step pair detects overflow from RAW_SUM + CARRY_IN.
                # Product ≈ base², safe for base < 2^26 (all configs except WORD).
                W_up[4, ge.RAW_SUM] = S
                W_up[4, ge.CARRY_IN] = S
                b_up[4] = -S * (base - 1.0)
                W_gate[4, ge.OP_START + opcode] = 1.0
                W_down[ge.RESULT, 4] = -float(base) / S

                W_up[5, ge.RAW_SUM] = S
                W_up[5, ge.CARRY_IN] = S
                b_up[5] = -S * float(base)
                W_gate[5, ge.OP_START + opcode] = 1.0
                W_down[ge.RESULT, 5] = float(base) / S

            # Units 6-7: Clear RAW_SUM
            bake_clear_pair(self.ffn, 6, ge.OP_START + opcode, ge.RAW_SUM, S)

            if N == 1:
                # Units 8-9: Clear CARRY_OUT (was preserved through layer 2)
                bake_clear_pair(self.ffn, 8, ge.OP_START + opcode, ge.CARRY_OUT, S)
            else:
                # Units 8-9: Clear CARRY_IN
                bake_clear_pair(self.ffn, 8, ge.OP_START + opcode, ge.CARRY_IN, S)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


def build_add_layers(config: ChunkConfig, opcode: int) -> nn.ModuleList:
    """Build 3-layer ADD pipeline for the given chunk config.

    Works at all 6 configs including WORD (32-bit, 1 position).
    """
    ge = GenericE(config)
    return nn.ModuleList([
        AddRawAndGenFFN(ge, opcode),
        AddCarryLookaheadFFN(ge, opcode),
        AddFinalResultFFN(ge, opcode),
    ])
