"""
Chunk-generic DIV pipeline: 4 layers.

Layer 1: Clear + Gather + Softmax1 reciprocal (merged)
Layer 2: Multiply Q_float = dividend × reciprocal
Layer 3: Floor extraction via fp64 MAGIC trick
Layer 4: Chunk subtraction: chunk_j = floor_j - base*floor_{j+1}

Parameterized by ChunkConfig. Floor extraction generalizes the MAGIC trick:
  eps_j = 2^(-20 - chunk_bits*j)
  scale_j = 1/base^j

For single-position configs (WORD), layers 3-4 simplify to direct floor.
"""

import math
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


class Softmax1ReciprocalModule(nn.Module):
    """Compute 1/divisor via softmax1 nibble construction.

    Generalizes to any base: score_j = log(base^j * d_j).
    """

    def __init__(self, ge: GenericE, opcode: int):
        super().__init__()
        self.ge = ge
        self.opcode = opcode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ge = self.ge
        B, N, D = x.shape
        opcode_w = x[:, 0, ge.OP_START + self.opcode]

        S = SOFTMAX1_SCALE
        base = ge.BASE
        num_pos = ge.NUM_POSITIONS

        scores = torch.full((B, num_pos), -S, device=x.device, dtype=x.dtype)
        for j in range(num_pos):
            d_j = x[:, j, ge.NIB_B]
            active = d_j > 0.5
            val = (d_j * float(base ** j)).clamp(min=0.5)
            scores[:, j] = torch.where(active, torch.log(val), scores[:, j])

        mx = scores.max(dim=-1, keepdim=True).values
        ex = torch.exp(scores - mx)
        sum_ex = ex.sum(dim=-1)
        exp_neg_mx = torch.exp(-mx.squeeze(-1))
        reciprocal = exp_neg_mx / sum_ex.clamp(min=1e-30)

        delta = torch.zeros_like(x)
        delta[:, 0, ge.SLOT_QUOTIENT] = opcode_w * reciprocal
        return x + delta


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


class FloorExtractionFFN(nn.Module):
    """Extract floor(Q/base^j) for j=0..N-1 via fp64 MAGIC trick.

    Generalized: eps_j = 2^(-20 - chunk_bits*j), scale_j = 1/base^j.

    Hidden units: N (floor) + 1 (MAGIC cancel) + 2*N (clear RESULT) = 3*N + 1.
    REQUIRES fp64 precision.
    """

    def __init__(self, ge: GenericE, opcode: int):
        super().__init__()
        N = ge.NUM_POSITIONS
        hidden_dim = 3 * N + 1

        # Always use fp64 for MAGIC trick
        self.flat_ffn = GenericFlattenedFFN(ge, hidden_dim=hidden_dim, dtype=torch.float64)
        fi = self.flat_ffn._flat_idx

        S = ge.SCALE
        base = ge.BASE
        chunk_bits = ge.config.chunk_bits
        C = float(2**20)

        with torch.no_grad():
            W_up = self.flat_ffn.W_up
            b_up = self.flat_ffn.b_up
            W_gate = self.flat_ffn.W_gate
            W_down = self.flat_ffn.W_down

            opcode_idx = fi(0, ge.OP_START + opcode)
            q_float_idx = fi(0, ge.SLOT_REMAINDER)

            # h=0..N-1: Floor extraction via MAGIC trick
            for j in range(N):
                h = j
                eps_j = 2.0 ** (-20 - chunk_bits * j)
                scale_j = 1.0 / float(base ** j)
                offset_j = -(0.5 - eps_j)
                result_j_idx = fi(j, ge.RESULT)

                W_up[h, q_float_idx] = scale_j
                W_up[h, opcode_idx] = offset_j
                b_up[h] = MAGIC
                W_gate[h, opcode_idx] = 1.0
                W_down[result_j_idx, h] = 1.0

            # h=N: MAGIC cancellation unit
            h = N
            W_up[h, opcode_idx] = C
            W_gate[h, opcode_idx] = 1.0
            for j in range(N):
                result_j_idx = fi(j, ge.RESULT)
                W_down[result_j_idx, h] = -MAGIC / C

            # h=N+1..3N: Clear old RESULT values
            for pos in range(N):
                result_pos_idx = fi(pos, ge.RESULT)
                h = N + 1 + pos * 2
                W_up[h, opcode_idx] = S
                W_gate[h, result_pos_idx] = -1.0
                W_down[result_pos_idx, h] = 1.0 / S

                h = N + 1 + pos * 2 + 1
                W_up[h, opcode_idx] = -S
                W_gate[h, result_pos_idx] = 1.0
                W_down[result_pos_idx, h] = 1.0 / S

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to fp64 for MAGIC trick, then back
        orig_dtype = x.dtype
        x64 = x.double()
        y64 = self.flat_ffn(x64)
        return y64.to(orig_dtype)


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


class DivMergedLayer1(nn.Module):
    """Layer 1: Clear + Gather + Reciprocal (merged — independent output slots)."""

    def __init__(self, ge: GenericE, opcode: int):
        super().__init__()
        self.clear = ClearDivSlotsFFN(ge, opcode)
        self.gather = GatherScalarFFN(ge, ge.NIB_A, ge.SLOT_DIVIDEND, opcode)
        self.reciprocal = Softmax1ReciprocalModule(ge, opcode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # These write to independent slots, so deltas sum correctly
        d1 = self.clear(x) - x
        d2 = self.gather(x) - x
        d3 = self.reciprocal(x) - x
        return x + d1 + d2 + d3


def build_div_layers(config: ChunkConfig, opcode: int) -> nn.ModuleList:
    """Build 4-layer DIV pipeline for the given chunk config.

    Works at all chunk sizes. Floor extraction uses fp64 MAGIC trick
    internally regardless of config precision.
    """
    ge = GenericE(config)
    layers = [
        DivMergedLayer1(ge, opcode),
        MultiplyReciprocalFFN(ge, opcode),
        FloorExtractionFFN(ge, opcode),
    ]
    if config.num_positions > 1:
        layers.append(ChunkSubtractFFN(ge, opcode))
    return nn.ModuleList(layers)
