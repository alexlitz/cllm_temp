"""
Efficient SHIFT for BYTE chunks using byte-level u32-safe arithmetic.

For 32-bit values stored as 4 byte lanes, shifts decompose into:
  - a byte-aligned component q = n // 8 (whole-byte realignment)
  - a sub-byte component r = n % 8 (bit shift inside each byte, with
    carry across the byte boundary)

This keeps every intermediate inside a single byte (or two bytes when
combining sub-byte carries), so no value ever needs an int64 / "long"
intermediate -- the u32-everywhere invariant holds throughout.

SHL: result_byte[i] = ((a[i-q] << r) | (a[i-q-1] >> (8-r))) & 0xFF
SHR: result_byte[i] = ((a[i+q] >> r) | (a[i+q+1] << (8-r))) & 0xFF
                      (with out-of-range a[k] treated as 0)
"""

import torch
import torch.nn as nn

from ..chunk_config import ChunkConfig, BYTE
from .common import GenericE, GenericFlattenedFFN, bake_clear_pair


def _u32_select(shift_amount: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Index-select ``table[shift_amount]`` using an int32 shift index.

    ``shift_amount`` is the per-batch shift count in 0..31 (clamped by
    caller). ``table`` is a 1-D buffer with 32 entries. Returns the
    selected entry, broadcastable against per-byte tensors.
    """
    idx = shift_amount.to(torch.int32).clamp(0, 31)
    return table[idx]


class EfficientSHLFFN(nn.Module):
    """Efficient left shift implemented directly on the per-byte lanes.

    SHL by n = 8q + r (0 <= q < 4, 0 <= r < 8):
        result[i] = ((a[i-q] << r) | (a[i-q-1] >> (8-r))) & 0xFF

    Every intermediate value fits in 16 bits (a single shifted byte is at
    most 255 << 7 = 32640 < 2^15), so the computation stays in int32 --
    no int64 widening is ever required.
    """

    def __init__(self, ge: GenericE, opcode: int):
        super().__init__()
        N = ge.NUM_POSITIONS
        base = ge.BASE  # 256 for BYTE

        self.ge = ge
        self.opcode = opcode
        self.N = N
        self.base = base

        # Two 32-entry int32 lookup tables: q = n // 8, r = n % 8 for each
        # shift amount in 0..31. These are buffers (not learnable params)
        # so the static u32 verifier sees no fp64 / no widening.
        q_table = torch.tensor([n // 8 for n in range(32)], dtype=torch.int32)
        r_table = torch.tensor([n % 8 for n in range(32)], dtype=torch.int32)
        self.register_buffer("q_table", q_table)
        self.register_buffer("r_table", r_table)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SHL operation, staying in int32 throughout."""
        B, N, D = x.shape
        ge = self.ge

        op_active = x[:, 0, ge.OP_START + self.opcode]

        # Stack byte lanes into an int32 tensor of shape [B, N].
        bytes_i32 = torch.stack(
            [x[:, pos, ge.NIB_A].to(torch.int32) & 0xFF for pos in range(N)],
            dim=1,
        )  # [B, N], each lane in 0..255

        shift_amount = x[:, 0, ge.NIB_B].to(torch.int32).clamp(0, 31)
        q = _u32_select(shift_amount, self.q_table)  # [B]
        r = _u32_select(shift_amount, self.r_table)  # [B]

        # Precompute 1 << r and 1 << (8 - r). When r == 0 the second
        # shift collapses (the carry term is zero). Use a mask to keep
        # the byte-wise arithmetic regular.
        lo_shift = (torch.ones_like(r) << r) & 0xFF   # int32, in 0..255 (when r<8)
        # For r == 0 we want hi_shift to multiply zero (no carry-in).
        # Compute (1 << (8 - r)) when r > 0, else 0.
        r_pos = (r > 0).to(torch.int32)
        hi_shift = r_pos * (torch.ones_like(r) << (8 - r * r_pos))  # 0 when r==0, else 1<<(8-r)
        # We will use ``rcomp = 8 - r`` only when r > 0. The `r * r_pos`
        # idiom keeps the shift exponent non-negative when r == 0.

        # Build the result lanes one by one.
        result = x.clone()
        # We accumulate result lanes in int32 then cast back to x.dtype.
        out_lanes = torch.zeros((B, N), dtype=torch.int32, device=x.device)
        for i in range(N):
            # src1 = a[i - q] when i - q in [0, N); else 0.
            # src2 = a[i - q - 1] when i - q - 1 in [0, N); else 0.
            src1 = torch.zeros(B, dtype=torch.int32, device=x.device)
            src2 = torch.zeros(B, dtype=torch.int32, device=x.device)
            for k in range(N):
                mask1 = (q == (i - k)).to(torch.int32)
                src1 = src1 + mask1 * bytes_i32[:, k]
                mask2 = (q == (i - k - 1)).to(torch.int32)
                src2 = src2 + mask2 * bytes_i32[:, k]
            # Apply sub-byte shift. lo_shift = 1 << r, hi_shift = 1<<(8-r)
            # for r > 0 and 0 for r == 0 (so the high half carries nothing).
            lo = (src1 * lo_shift) & 0xFF
            # When r == 0 the hi_shift mask is 0, so src2 // hi_shift would
            # divide by zero. Guard by using the r_pos mask to gate the carry.
            carry = torch.where(
                r_pos.bool(),
                (src2 // torch.clamp(hi_shift, min=1)) & 0xFF,
                torch.zeros_like(src2),
            )
            out_lanes[:, i] = (lo | carry) & 0xFF

        for pos in range(N):
            result[:, pos, ge.RESULT] = out_lanes[:, pos].to(x.dtype) * op_active

        return result


class EfficientSHRFFN(nn.Module):
    """Efficient right shift implemented directly on the per-byte lanes.

    SHR by n = 8q + r (0 <= q < 4, 0 <= r < 8):
        result[i] = ((a[i+q] >> r) | (a[i+q+1] << (8-r))) & 0xFF

    All intermediates fit in 16 bits (single shifted byte stays below
    2^15), so the computation stays in int32 -- no int64 widening.
    """

    def __init__(self, ge: GenericE, opcode: int):
        super().__init__()
        N = ge.NUM_POSITIONS

        self.ge = ge
        self.opcode = opcode
        self.N = N

        q_table = torch.tensor([n // 8 for n in range(32)], dtype=torch.int32)
        r_table = torch.tensor([n % 8 for n in range(32)], dtype=torch.int32)
        self.register_buffer("q_table", q_table)
        self.register_buffer("r_table", r_table)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SHR operation, staying in int32 throughout."""
        B, N, D = x.shape
        ge = self.ge

        op_active = x[:, 0, ge.OP_START + self.opcode]

        bytes_i32 = torch.stack(
            [x[:, pos, ge.NIB_A].to(torch.int32) & 0xFF for pos in range(N)],
            dim=1,
        )  # [B, N]

        shift_amount = x[:, 0, ge.NIB_B].to(torch.int32).clamp(0, 31)
        q = _u32_select(shift_amount, self.q_table)
        r = _u32_select(shift_amount, self.r_table)

        # lo_div = 1 << r (always valid as r in 0..7), hi_shift = 1 << (8 - r)
        # for the carry from the next byte (zero when r == 0).
        lo_div = (torch.ones_like(r) << r) & 0xFF  # 1..128
        r_pos = (r > 0).to(torch.int32)
        hi_shift = r_pos * (torch.ones_like(r) << (8 - r * r_pos))

        result = x.clone()
        out_lanes = torch.zeros((B, N), dtype=torch.int32, device=x.device)
        for i in range(N):
            src1 = torch.zeros(B, dtype=torch.int32, device=x.device)
            src2 = torch.zeros(B, dtype=torch.int32, device=x.device)
            for k in range(N):
                mask1 = (q == (k - i)).to(torch.int32)
                src1 = src1 + mask1 * bytes_i32[:, k]
                mask2 = (q == (k - i - 1)).to(torch.int32)
                src2 = src2 + mask2 * bytes_i32[:, k]
            lo = (src1 // torch.clamp(lo_div, min=1)) & 0xFF
            carry = torch.where(
                r_pos.bool(),
                (src2 * hi_shift) & 0xFF,
                torch.zeros_like(src2),
            )
            out_lanes[:, i] = (lo | carry) & 0xFF

        for pos in range(N):
            result[:, pos, ge.RESULT] = out_lanes[:, pos].to(x.dtype) * op_active

        return result


def build_efficient_shl_layers(config: ChunkConfig, opcode: int = 23) -> nn.ModuleList:
    """Build efficient SHL layer."""
    ge = GenericE(config)
    return nn.ModuleList([EfficientSHLFFN(ge, opcode)])


def build_efficient_shr_layers(config: ChunkConfig, opcode: int = 24) -> nn.ModuleList:
    """Build efficient SHR layer."""
    ge = GenericE(config)
    return nn.ModuleList([EfficientSHRFFN(ge, opcode)])


def count_efficient_shift_params():
    """Count parameters for efficient SHIFT implementations."""
    shl_layers = build_efficient_shl_layers(BYTE, opcode=23)
    shr_layers = build_efficient_shr_layers(BYTE, opcode=24)

    def count(layers):
        return sum(sum((p != 0).sum().item() for p in layer.parameters()) for layer in layers)

    shl_params = count(shl_layers)
    shr_params = count(shr_layers)

    print("Efficient BYTE SHIFT (multiply by powers of 2):")
    print(f"  SHL: {shl_params:,} params")
    print(f"  SHR: {shr_params:,} params")
    print(f"  Total: {shl_params + shr_params:,} params")

    return shl_params, shr_params


if __name__ == '__main__':
    print("="*70)
    print("Efficient SHIFT with Power-of-2 Multiplication")
    print("="*70)
    count_efficient_shift_params()

    # Test
    ge = GenericE(BYTE)

    # Test SHL: 1 << 4 = 16
    print("\nTest SHL: 1 << 4 = 16")
    x = torch.zeros(1, 4, ge.DIM)
    x[0, 0, ge.NIB_A] = 1.0
    x[0, 0, ge.NIB_B] = 4.0
    for pos in range(4):
        x[0, pos, ge.OP_START + 23] = 1.0

    shl = EfficientSHLFFN(ge, opcode=23)
    with torch.no_grad():
        result = shl(x)
    print(f"  Result byte 0: {result[0, 0, ge.RESULT].item()}")

    # Test SHR: 256 >> 4 = 16
    print("\nTest SHR: 256 >> 4 = 16")
    x2 = torch.zeros(1, 4, ge.DIM)
    x2[0, 0, ge.NIB_A] = 0.0
    x2[0, 1, ge.NIB_A] = 1.0  # 256 = 0x100
    x2[0, 0, ge.NIB_B] = 4.0
    for pos in range(4):
        x2[0, pos, ge.OP_START + 24] = 1.0

    shr = EfficientSHRFFN(ge, opcode=24)
    with torch.no_grad():
        result2 = shr(x2)
    print(f"  Result byte 0: {result2[0, 0, ge.RESULT].item()}")
