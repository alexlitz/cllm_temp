"""
Chunk-generic DIV pipeline: nibble-level long division (no fp64).

Replaces the previous fp64 MAGIC-floor approach with direct schoolbook long
division on nibble vectors. See ``divmod_longdiv.py`` for the algorithm.

Layer count:
  Layer 1: ClearDivSlotsFFN — clear scratch slots
  Layer 2: LongDivisionModule — 8 outer iterations of bring-down + trial
           multiply + compare + subtract (24 sub-FFN-equivalent operations)
  Layer 3: EmitDivResultModule — copy SLOT_QUOTIENT[*] → RESULT[*]
"""

import torch
import torch.nn as nn

from ..chunk_config import ChunkConfig
from .common import GenericE, GenericPureFFN, GenericFlattenedFFN, bake_clear_pair

MAGIC = 3.0 * 2**51  # 6755399441055744.0 — fp64 ULP = 1 at this scale
SOFTMAX1_SCALE = 60.0


class ClearDivSlotsFFN(nn.Module):
    """Clear division temp slots at position 0."""

    def __init__(self, ge: GenericE, opcode: int):
        super().__init__()
        slots = [ge.SLOT_DIVIDEND, ge.SLOT_DIVISOR, ge.SLOT_REMAINDER, ge.SLOT_QUOTIENT]
        dtype = ge.config.torch_dtype

        self.flat_ffn = GenericFlattenedFFN(ge, hidden_dim=len(slots) * 2, dtype=dtype)
        fi = self.flat_ffn._flat_idx
        S = ge.DIV_SCALE

        with torch.no_grad():
            for i, slot in enumerate(slots):
                bake_clear_pair(self.flat_ffn.ffn, i * 2,
                                fi(0, ge.OP_START + opcode),
                                fi(0, slot), S)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flat_ffn(x)


class GatherScalarFFN(nn.Module):
    """Gather N chunk positions into a scalar at position 0."""

    def __init__(self, ge: GenericE, source_slot: int, dest_slot: int, opcode: int):
        super().__init__()
        N = ge.NUM_POSITIONS
        S = ge.DIV_SCALE
        base = ge.BASE
        dtype = ge.config.torch_dtype

        self.flat_ffn = GenericFlattenedFFN(ge, hidden_dim=N + 1, dtype=dtype)
        fi = self.flat_ffn._flat_idx

        with torch.no_grad():
            W_up = self.flat_ffn.W_up
            W_gate = self.flat_ffn.W_gate
            W_down = self.flat_ffn.W_down
            dst_idx = fi(0, dest_slot)

            for i in range(N):
                h = i
                src_idx = fi(i, source_slot)
                W_up[h, fi(0, ge.OP_START + opcode)] = S
                W_gate[h, src_idx] = float(base ** i)
                W_down[dst_idx, h] = 1.0 / S

            # Saturation cancel
            h = N
            W_up[h, fi(0, ge.OP_START + opcode)] = -S
            W_gate[h, dst_idx] = -1.0  # gate on result to cancel leakage
            W_down[dst_idx, h] = 0.0   # no output — just soak up the negative arm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flat_ffn(x)


class MultiplyReciprocalFFN(nn.Module):
    """Multiply: Q_float = dividend × reciprocal → SLOT_REMAINDER."""

    def __init__(self, ge: GenericE, opcode: int, subtract_one: bool = False):
        super().__init__()
        S = ge.DIV_SCALE
        dtype = ge.config.torch_dtype

        self.flat_ffn = GenericFlattenedFFN(ge, hidden_dim=2, dtype=dtype)
        fi = self.flat_ffn._flat_idx

        with torch.no_grad():
            dividend_idx = fi(0, ge.SLOT_DIVIDEND)
            reciprocal_idx = fi(0, ge.SLOT_QUOTIENT)
            result_idx = fi(0, ge.SLOT_REMAINDER)

            self.flat_ffn.W_up[0, dividend_idx] = S
            if subtract_one:
                self.flat_ffn.b_up[0] = -S
            self.flat_ffn.W_gate[0, reciprocal_idx] = 1.0
            self.flat_ffn.W_down[result_idx, 0] = 1.0 / S

            self.flat_ffn.W_up[1, dividend_idx] = -S
            if subtract_one:
                self.flat_ffn.b_up[1] = S
            self.flat_ffn.W_gate[1, reciprocal_idx] = -1.0
            self.flat_ffn.W_down[result_idx, 1] = 1.0 / S

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flat_ffn(x)


class FloorExtractionFP32FFN(nn.Module):
    """Extract floor(Q/base^j) for j=0..N-1 via window detection - FP32 SAFE.

    WARNING: This implementation has fundamental limitations for large quotients.
    For 32-bit division, floor(Q/base^j) for positions 0-3 can exceed 2^24,
    which is beyond fp32's exact integer range. The window detection approach
    cannot enumerate all possible floor values efficiently.

    This implementation only works correctly when Q < base^5 (i.e., quotient < 65536).
    For full 32-bit division support, use the fp64 MAGIC approach instead.

    For each position j and nibble value k=0..base-1:
      Detect if floor(Q/base^j) == k using step pairs
      Output k to RESULT[j] when detected

    RESULT[j] = sum_{k=0}^{base-1} k * step(floor(Q/base^j) == k)

    Hidden units: N * base * 4 (step pairs) + 2*N (clear RESULT).
    """

    def __init__(self, ge: GenericE, opcode: int):
        super().__init__()
        N = ge.NUM_POSITIONS
        base = ge.BASE
        S = ge.SCALE
        dtype = ge.config.torch_dtype

        # 4 units per (position, nibble_value) + 2*N for clearing
        hidden_dim = N * base * 4 + 2 * N

        self.flat_ffn = GenericFlattenedFFN(ge, hidden_dim=hidden_dim, dtype=dtype)
        fi = self.flat_ffn._flat_idx

        eps = 0.001  # Small shift to avoid exact integer boundaries

        with torch.no_grad():
            W_up = self.flat_ffn.W_up
            b_up = self.flat_ffn.b_up
            W_gate = self.flat_ffn.W_gate
            W_down = self.flat_ffn.W_down

            opcode_idx = fi(0, ge.OP_START + opcode)
            q_float_idx = fi(0, ge.SLOT_REMAINDER)

            h = 0

            # Floor extraction via window detection
            for j in range(N):
                scale_j = 1.0 / float(base ** j)
                result_j_idx = fi(j, ge.RESULT)

                for k in range(base):
                    # Detect: floor(Q/base^j) == k
                    # Equivalently: Q/base^j is in [k, k+1)
                    # Step pair for step(Q*scale_j >= k-eps) - step(Q*scale_j >= k+1-eps)
                    # Output: k when detected

                    threshold_lo = k - eps
                    threshold_hi = k + 1 - eps

                    # Step pair for step(scaled_Q >= threshold_lo)
                    W_up[h, q_float_idx] = S * scale_j
                    b_up[h] = -S * (threshold_lo - 0.5)
                    W_gate[h, opcode_idx] = 1.0
                    W_down[result_j_idx, h] = float(k) / S
                    h += 1

                    W_up[h, q_float_idx] = S * scale_j
                    b_up[h] = -S * (threshold_lo + 0.5)
                    W_gate[h, opcode_idx] = 1.0
                    W_down[result_j_idx, h] = -float(k) / S
                    h += 1

                    # Step pair for step(scaled_Q >= threshold_hi) - subtract
                    W_up[h, q_float_idx] = S * scale_j
                    b_up[h] = -S * (threshold_hi - 0.5)
                    W_gate[h, opcode_idx] = 1.0
                    W_down[result_j_idx, h] = -float(k) / S
                    h += 1

                    W_up[h, q_float_idx] = S * scale_j
                    b_up[h] = -S * (threshold_hi + 0.5)
                    W_gate[h, opcode_idx] = 1.0
                    W_down[result_j_idx, h] = float(k) / S
                    h += 1

            # Clear old RESULT values
            for pos in range(N):
                result_pos_idx = fi(pos, ge.RESULT)

                W_up[h, opcode_idx] = S
                W_gate[h, result_pos_idx] = -1.0
                W_down[result_pos_idx, h] = 1.0 / S
                h += 1

                W_up[h, opcode_idx] = -S
                W_gate[h, result_pos_idx] = 1.0
                W_down[result_pos_idx, h] = 1.0 / S
                h += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flat_ffn(x)


class ChunkSubtractFFN(nn.Module):
    """Convert floor values to chunk digits: RESULT[j] -= base * RESULT[j+1].

    Hidden units: 2*(N-1).
    """

    def __init__(self, ge: GenericE, opcode: int):
        super().__init__()
        N = ge.NUM_POSITIONS
        if N <= 1:
            # Single position — nothing to subtract
            self.flat_ffn = None
            return

        S = ge.SCALE
        base = ge.BASE
        dtype = ge.config.torch_dtype
        hidden_dim = 2 * (N - 1)

        self.flat_ffn = GenericFlattenedFFN(ge, hidden_dim=hidden_dim, dtype=dtype)
        fi = self.flat_ffn._flat_idx

        with torch.no_grad():
            opcode_idx = fi(0, ge.OP_START + opcode)

            for j in range(N - 1):
                result_j_idx = fi(j, ge.RESULT)
                result_jp1_idx = fi(j + 1, ge.RESULT)

                h = j * 2
                self.flat_ffn.W_up[h, opcode_idx] = S
                self.flat_ffn.W_gate[h, result_jp1_idx] = 1.0
                self.flat_ffn.W_down[result_j_idx, h] = -float(base) / S

                h = j * 2 + 1
                self.flat_ffn.W_up[h, opcode_idx] = -S
                self.flat_ffn.W_gate[h, result_jp1_idx] = -1.0
                self.flat_ffn.W_down[result_j_idx, h] = -float(base) / S

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.flat_ffn is None:
            return x
        return self.flat_ffn(x)


def build_div_layers(config: ChunkConfig, opcode: int, fp32_floor: bool = False) -> nn.ModuleList:
    """Build DIV pipeline using nibble-level long division (no fp64).

    All arithmetic is fp32. The ``fp32_floor`` argument is retained for
    backward compatibility but is now ignored — the long-division pipeline
    has no floor-extraction stage to swap.
    """
    # Defer import to avoid cycles (mod.py also re-exports from div.py).
    from .divmod_longdiv import build_div_layers_longdiv
    return build_div_layers_longdiv(config, opcode)


def build_div_layers_fp32(config: ChunkConfig, opcode: int) -> nn.ModuleList:
    """Build DIV pipeline using nibble-level long division.

    Backward-compat alias; identical to ``build_div_layers``.
    """
    return build_div_layers(config, opcode)
