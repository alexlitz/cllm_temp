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
from .common import GenericE
from .div import FloorExtractionFFN, ChunkSubtractFFN, SOFTMAX1_SCALE
from .mul import build_mul_layers
from .sub import SubRawAndGenFFN, SubBorrowLookaheadFFN, SubFinalResultFFN


class ModDivScalarModule(nn.Module):
    """Compute Q_float = (dividend-1) * (1/divisor) in fp64.

    Replaces DivMergedLayer1 + MultiplyReciprocalFFN for MOD.
    The entire scalar computation happens in fp64 to avoid
    precision loss from fp32 gather (scalar up to 4.3e9 > 2^24).

    Writes Q_float to SLOT_REMAINDER at position 0.
    """

    def __init__(self, ge: GenericE, opcode: int):
        super().__init__()
        self.ge = ge
        self.opcode = opcode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ge = self.ge
        B, N, D = x.shape
        opcode_w = x[:, 0, ge.OP_START + self.opcode]
        base = ge.BASE
        num_pos = ge.NUM_POSITIONS

        # Work in fp64 for precision
        S = SOFTMAX1_SCALE

        # Gather dividend
        dividend = torch.zeros(B, device=x.device, dtype=torch.float64)
        for j in range(num_pos):
            dividend = dividend + x[:, j, ge.NIB_A].double() * float(base ** j)

        # Compute reciprocal via softmax1 (in fp64)
        scores = torch.full((B, num_pos), -S, device=x.device, dtype=torch.float64)
        for j in range(num_pos):
            d_j = x[:, j, ge.NIB_B].double()
            active = d_j > 0.5
            val = (d_j * float(base ** j)).clamp(min=0.5)
            scores[:, j] = torch.where(active, torch.log(val), scores[:, j])

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

    Uses direct PyTorch operations (gather, compare, subtract, extract).
    """

    def __init__(self, ge: GenericE, opcode: int):
        super().__init__()
        self.ge = ge
        self.opcode = opcode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ge = self.ge
        B, N, D = x.shape
        opcode_w = x[:, 0, ge.OP_START + self.opcode]
        base = ge.BASE
        num_pos = ge.NUM_POSITIONS

        # Gather remainder and divisor as scalars (fp64 for precision)
        remainder = torch.zeros(B, device=x.device, dtype=torch.float64)
        divisor = torch.zeros(B, device=x.device, dtype=torch.float64)
        for j in range(num_pos):
            pw = float(base ** j)
            remainder = remainder + x[:, j, ge.RESULT].double() * pw
            divisor = divisor + x[:, j, ge.NIB_B].double() * pw

        # Round to nearest integer to eliminate fp32 residuals from MUL+SUB
        remainder = torch.round(remainder)

        # Conditional subtract
        needs_correction = (remainder >= divisor).to(torch.float64)
        corrected = remainder - needs_correction * divisor

        # Extract corrected chunks back to RESULT
        delta = torch.zeros_like(x)
        val = corrected
        for j in range(num_pos - 1, -1, -1):
            pw = float(base ** j)
            chunk = torch.floor(val / pw).clamp(0, float(ge.CHUNK_MAX))
            delta[:, j, ge.RESULT] = opcode_w * (chunk - x[:, j, ge.RESULT]).to(x.dtype)
            val = val - chunk * pw

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
