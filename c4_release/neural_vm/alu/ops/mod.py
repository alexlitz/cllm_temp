"""
Chunk-generic MOD pipeline: nibble-level long division (no fp64).

MOD shares the long-division core with DIV: 8 outer iterations produce both
quotient and remainder simultaneously. The MOD entry point emits the
remainder; the DIV entry point emits the quotient. See
``divmod_longdiv.py`` for algorithm details.

Pipeline:
  Layer 1: ClearDivSlotsFFN — clear scratch slots
  Layer 2: LongDivisionModule — long division → SLOT_REMAINDER, SLOT_QUOTIENT
  Layer 3: EmitDivResultModule — copy SLOT_REMAINDER[*] → RESULT[*]

The legacy fp64 modules (``ModDivScalarModule``, ``Fp64MulSubModule``,
``ModCorrectionModule``) are retained DEPRECATED below in case any
external code still imports them. ``build_mod_layers`` no longer uses them.
"""

import torch
import torch.nn as nn

from ..chunk_config import ChunkConfig
from .common import GenericE, magic_floor32
from .div import FloorExtractionFFN, ChunkSubtractFFN, SOFTMAX1_SCALE
from .mul import build_mul_layers
from .sub import SubRawAndGenFFN, SubBorrowLookaheadFFN, SubFinalResultFFN


class ModDivScalarModule(nn.Module):
    """DEPRECATED — fp64 scalar (dividend-1) * (1/divisor) stage.

    Was used by the previous MOD pipeline. The new long-division pipeline
    in ``divmod_longdiv.py`` does not use this module. Kept only for
    backward compatibility.

    Compute Q_float = (dividend-1) * (1/divisor) in fp64.

    Writes Q_float to SLOT_REMAINDER at position 0.
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
    """DEPRECATED — fp64 MUL + SUB stages from the previous MOD pipeline.

    No longer used by ``build_mod_layers`` (the new long-division pipeline
    in ``divmod_longdiv.py`` computes the remainder directly during long
    division and never invokes this module). Retained only for backward
    compatibility with any external caller that still imports the name —
    it has no remaining call sites in this repository.

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
    """DEPRECATED — final remainder correction for the previous MOD pipeline.

    Not used by the new long-division pipeline in ``divmod_longdiv.py``
    (which produces an exact remainder directly). Kept only for backward
    compatibility.

    If remainder >= divisor, subtract divisor once.
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
    """Build MOD pipeline using nibble-level long division (no fp64).

    All arithmetic is fp32. See ``divmod_longdiv.py``.
    """
    from .divmod_longdiv import build_mod_layers_longdiv
    return build_mod_layers_longdiv(config, opcode)
