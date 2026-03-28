"""
Chunk-generic MOD pipeline: DIV + MUL + SUB + correction.

MOD = dividend - floor((dividend-1)/divisor) * divisor, then correct if >= divisor.

The x-1 trick ensures the quotient is never too high:
  floor((x-1)/n) <= floor(x/n), with gap from (x-1)/n to next int >= 1/n.

Supported configs: NIBBLE, BYTE.
(Requires both DIV and MUL support; BIT/PAIR overflow fp16 scalar gather,
HALFWORD/WORD impractical for MUL step pairs.)

Pipeline:
  Layer 1: ModDivScalarModule — gather + reciprocal + (x-1)*reciprocal in fp64
  Layer 2: FloorExtractionFFN — MAGIC trick (fp64)
  Layer 3: ChunkSubtractFFN — extract chunk digits
  Layer 4: Fp64MulSubModule — MUL(quotient*divisor) + SUB(dividend-product) in fp64
  Layer 5: ModCorrectionModule (if remainder >= divisor, subtract)

The MUL+SUB stages are run in fp64 to avoid fp32 accumulation errors
in the carry extraction step pairs (which sum O(100) large terms).
"""

import torch
import torch.nn as nn

from ..chunk_config import ChunkConfig
from .common import GenericE, magic_floor32
from .div import FloorExtractionFFN, ChunkSubtractFFN, SOFTMAX1_SCALE
from .mul import build_mul_layers
from .sub import SubRawAndGenFFN, SubBorrowLookaheadFFN, SubFinalResultFFN


class ModDivScalarModule(nn.Module):
    """Compute Q_float = (dividend-1) * (1/divisor) in fp64.

    Replaces DivMergedLayer1 + MultiplyReciprocalFFN for MOD.
    The entire scalar computation happens in fp64 to avoid
    precision loss from fp32 gather (scalar up to 4.3e9 > 2^24).

    Writes Q_float to SLOT_REMAINDER at position 0.
    Pure neural: no Python loops in forward, all tensor operations.
    """

    def __init__(self, ge: GenericE, opcode: int):
        super().__init__()
        self.ge = ge
        self.opcode = opcode
        num_pos = ge.NUM_POSITIONS
        base = ge.BASE

        # Pre-compute base powers as buffer (no loops in forward)
        base_powers = torch.tensor(
            [float(base ** j) for j in range(num_pos)],
            dtype=torch.float64
        )
        self.register_buffer('base_powers', base_powers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ge = self.ge
        B, N, D = x.shape
        opcode_w = x[:, 0, ge.OP_START + self.opcode]
        num_pos = ge.NUM_POSITIONS
        S = SOFTMAX1_SCALE

        # Gather dividend: sum(nib_a[j] * base^j) - vectorized
        nib_a = x[:, :num_pos, ge.NIB_A].double()  # [B, num_pos]
        dividend = (nib_a * self.base_powers).sum(dim=-1)  # [B]

        # Gather divisor nibbles and compute softmax1 reciprocal
        nib_b = x[:, :num_pos, ge.NIB_B].double()  # [B, num_pos]

        # Compute scores = log(base^j * d_j) where d_j > 0, else -S
        val = (nib_b * self.base_powers).clamp(min=0.5)
        active = nib_b > 0.5
        scores = torch.where(active, torch.log(val), torch.full_like(val, -S))

        # Softmax1 reciprocal
        mx = scores.max(dim=-1, keepdim=True).values
        ex = torch.exp(scores - mx)
        sum_ex = ex.sum(dim=-1)
        exp_neg_mx = torch.exp(-mx.squeeze(-1))
        reciprocal = exp_neg_mx / sum_ex.clamp(min=1e-30)

        # (x-1) multiply in fp64
        q_float = (dividend - 1.0) * reciprocal

        # Write to SLOT_REMAINDER at position 0
        delta = torch.zeros_like(x)
        delta[:, 0, ge.SLOT_REMAINDER] = (opcode_w * q_float).to(x.dtype)
        return x + delta


class Fp64MulSubModule(nn.Module):
    """Run MUL + SUB stages in fp64 to avoid accumulation errors.

    Wraps MUL layers (quotient * divisor) and SUB layers (dividend - product)
    with fp64 upcast/downcast.
    """

    def __init__(self, ge: GenericE, opcode: int, config: ChunkConfig):
        super().__init__()
        # Build MUL and SUB layers in fp64
        from ..chunk_config import ChunkConfig as CC
        fp64_config = CC(chunk_bits=config.chunk_bits,
                         total_bits=config.total_bits,
                         precision="fp64")
        ge64 = GenericE(fp64_config)

        self.mul_layers = build_mul_layers(fp64_config, opcode, source_a=ge64.RESULT)
        self.sub_layers = nn.ModuleList([
            SubRawAndGenFFN(ge64, opcode, source_b=ge64.RESULT, clear_result=True),
            SubBorrowLookaheadFFN(ge64, opcode),
            SubFinalResultFFN(ge64, opcode),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.double()
        for layer in self.mul_layers:
            x = layer(x)
        for layer in self.sub_layers:
            x = layer(x)
        return x.to(orig_dtype)


class ModCorrectionModule(nn.Module):
    """If remainder >= divisor, subtract divisor once.

    After x-1 trick, the quotient is at most 1 too low, so
    remainder is in [0, 2*divisor-1]. One correction suffices.

    Pure neural: no Python loops in forward, all tensor operations.
    Uses MAGIC floor trick for chunk extraction.
    """

    def __init__(self, ge: GenericE, opcode: int):
        super().__init__()
        self.ge = ge
        self.opcode = opcode
        num_pos = ge.NUM_POSITIONS
        base = ge.BASE

        # Pre-compute base powers and inverse powers as buffers
        base_powers = torch.tensor(
            [float(base ** j) for j in range(num_pos)],
            dtype=torch.float64
        )
        inv_base_powers = torch.tensor(
            [1.0 / float(base ** j) for j in range(num_pos)],
            dtype=torch.float64
        )
        self.register_buffer('base_powers', base_powers)
        self.register_buffer('inv_base_powers', inv_base_powers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ge = self.ge
        B, N, D = x.shape
        opcode_w = x[:, 0, ge.OP_START + self.opcode]
        num_pos = ge.NUM_POSITIONS
        base = ge.BASE

        # Gather remainder and divisor as scalars - vectorized
        result_nib = x[:, :num_pos, ge.RESULT].double()  # [B, num_pos]
        nib_b = x[:, :num_pos, ge.NIB_B].double()  # [B, num_pos]

        remainder = (result_nib * self.base_powers).sum(dim=-1)  # [B]
        divisor = (nib_b * self.base_powers).sum(dim=-1)  # [B]

        # Round to nearest integer to eliminate fp32 residuals from MUL+SUB
        remainder = torch.round(remainder)

        # Conditional subtract
        needs_correction = (remainder >= divisor).to(torch.float64)
        corrected = remainder - needs_correction * divisor

        # Extract chunks using MAGIC floor trick - vectorized
        # floor(corrected / base^j) for all j at once
        scaled = corrected[:, None] * self.inv_base_powers  # [B, num_pos]
        floored = magic_floor32(scaled)  # [B, num_pos]

        # Convert cumulative floors to individual chunks:
        # chunk[j] = floor[j] - base * floor[j+1]
        # Use shift and subtract (no loop)
        floored_shifted = torch.cat([
            floored[:, 1:],
            torch.zeros(B, 1, device=x.device, dtype=floored.dtype)
        ], dim=-1)  # [B, num_pos] shifted left by 1
        chunks = (floored - float(base) * floored_shifted).clamp(0, float(ge.CHUNK_MAX))

        # Write to RESULT
        delta = torch.zeros_like(x)
        old_result = x[:, :num_pos, ge.RESULT]
        delta[:, :num_pos, ge.RESULT] = (opcode_w[:, None] * (chunks - old_result.double())).to(x.dtype)

        return x + delta


def build_mod_layers(config: ChunkConfig, opcode: int = 29) -> nn.ModuleList:
    """Build MOD pipeline for NIBBLE/BYTE configs.

    Pipeline: fp64 scalar div → floor extraction → chunk subtract →
              fp64 MUL+SUB → correction.
    """
    ge = GenericE(config)
    layers = []

    # 1. Scalar DIV phase in fp64: (dividend-1) * (1/divisor) → SLOT_REMAINDER
    layers.append(ModDivScalarModule(ge, opcode))

    # 2-3. Floor extraction + chunk subtraction (from DIV pipeline)
    layers.append(FloorExtractionFFN(ge, opcode))
    if config.num_positions > 1:
        layers.append(ChunkSubtractFFN(ge, opcode))

    # 4. MUL + SUB in fp64: quotient*divisor, then dividend-product
    layers.append(Fp64MulSubModule(ge, opcode, config))

    # 5. Correction: if remainder >= divisor, subtract divisor
    layers.append(ModCorrectionModule(ge, opcode))

    return nn.ModuleList(layers)
