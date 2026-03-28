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
    Pure neural: no Python loops, all tensor operations.
    Uses fp64 for precision to avoid off-by-one errors.
    """

    def __init__(self, ge: GenericE, opcode: int):
        super().__init__()
        self.ge = ge
        self.opcode = opcode
        self.S = SOFTMAX1_SCALE
        # Pre-compute base powers as a buffer (no loops in forward)
        base_powers = torch.tensor(
            [float(ge.BASE ** j) for j in range(ge.NUM_POSITIONS)],
            dtype=torch.float64
        )
        self.register_buffer('base_powers', base_powers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ge = self.ge
        B, N, D = x.shape
        opcode_w = x[:, 0, ge.OP_START + self.opcode]

        # Extract divisor nibbles: [B, num_pos]
        d = x[:, :ge.NUM_POSITIONS, ge.NIB_B].double()  # Use fp64 for precision

        # Compute scores = log(base^j * d_j) where d_j > 0, else -S
        val = (d * self.base_powers).clamp(min=0.5)
        active = d > 0.5
        scores = torch.where(active, torch.log(val), torch.full_like(val, -self.S))

        # Softmax1 reciprocal computation in fp64
        mx = scores.max(dim=-1, keepdim=True).values
        ex = torch.exp(scores - mx)
        sum_ex = ex.sum(dim=-1)
        exp_neg_mx = torch.exp(-mx.squeeze(-1))
        reciprocal = exp_neg_mx / sum_ex.clamp(min=1e-30)

        # Write result back in original dtype
        delta = torch.zeros_like(x)
        delta[:, 0, ge.SLOT_QUOTIENT] = opcode_w * reciprocal.to(x.dtype)
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

    Uses direct tensor operations (not SwiGLU) for batch-stable precision.
    The MAGIC trick: floor(x) = (x - 0.5 + eps + MAGIC) - MAGIC
    where MAGIC = 3 * 2^51 forces fp64 to round to nearest integer.

    This is a pure neural implementation - all operations are tensor ops,
    no Python arithmetic on extracted values.
    """

    def __init__(self, ge: GenericE, opcode: int):
        super().__init__()
        self.ge = ge
        self.opcode = opcode
        N = ge.NUM_POSITIONS
        base = ge.BASE
        chunk_bits = ge.config.chunk_bits

        # Pre-compute scale factors as buffers (no Python math in forward)
        scales = torch.tensor([1.0 / float(base ** j) for j in range(N)], dtype=torch.float64)
        eps_vals = torch.tensor([2.0 ** (-20 - chunk_bits * j) for j in range(N)], dtype=torch.float64)
        offsets = -(0.5 - eps_vals)  # offset_j for each position

        self.register_buffer('scales', scales)
        self.register_buffer('offsets', offsets)
        self.register_buffer('magic', torch.tensor(MAGIC, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract floor values using direct tensor operations with MAGIC trick."""
        ge = self.ge
        B, N, D = x.shape
        orig_dtype = x.dtype

        # Get opcode weight and Q_float in fp64
        opcode_w = x[:, 0, ge.OP_START + self.opcode].double()  # [B]
        q_float = x[:, 0, ge.SLOT_REMAINDER].double()  # [B]

        # Compute scaled values for all positions at once: [B, N]
        # scaled[b, j] = q_float[b] * scales[j] + offsets[j]
        scaled = q_float[:, None] * self.scales[None, :] + self.offsets[None, :]

        # Apply MAGIC trick for floor: floor(x) = (x + MAGIC) - MAGIC
        # This works because at MAGIC scale, fp64 can only represent integers
        floored = (scaled + self.magic) - self.magic

        # Gate by opcode and write to RESULT
        # First clear old RESULT values, then write new ones
        delta = torch.zeros_like(x)

        # Clear RESULT at all positions (multiply old value by -opcode_w, add back)
        old_results = x[:, :ge.NUM_POSITIONS, ge.RESULT]
        delta[:, :ge.NUM_POSITIONS, ge.RESULT] = -old_results * opcode_w[:, None]

        # Write floored values gated by opcode
        delta[:, :ge.NUM_POSITIONS, ge.RESULT] += (floored * opcode_w[:, None]).to(orig_dtype)

        return x + delta


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


def build_div_layers(config: ChunkConfig, opcode: int, fp32_floor: bool = False) -> nn.ModuleList:
    """Build 4-layer DIV pipeline for the given chunk config.

    Args:
        config: Chunk configuration
        opcode: DIV opcode number
        fp32_floor: If True, use fp32-safe floor extraction (more params but no fp64).
                   If False (default), use MAGIC fp64 trick (fewer params but requires fp64).

    Works at all chunk sizes.
    """
    ge = GenericE(config)

    if fp32_floor:
        floor_layer = FloorExtractionFP32FFN(ge, opcode)
    else:
        floor_layer = FloorExtractionFFN(ge, opcode)

    layers = [
        DivMergedLayer1(ge, opcode),
        MultiplyReciprocalFFN(ge, opcode),
        floor_layer,
    ]
    if config.num_positions > 1:
        layers.append(ChunkSubtractFFN(ge, opcode))
    return nn.ModuleList(layers)


def build_div_layers_fp32(config: ChunkConfig, opcode: int) -> nn.ModuleList:
    """Build DIV pipeline using fp32-safe floor extraction.

    WARNING: Only works correctly for quotients < 65536 (16-bit results).
    For full 32-bit division, use the default fp64 MAGIC approach.

    Convenience wrapper for build_div_layers with fp32_floor=True.
    """
    return build_div_layers(config, opcode, fp32_floor=True)
