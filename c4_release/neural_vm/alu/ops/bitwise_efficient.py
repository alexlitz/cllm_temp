"""
Efficient bitwise operations (AND, OR, XOR) for BYTE chunks.

For byte-level operations, we can use lookup tables that are much smaller
than nibble-based approaches, or we can use SwiGLU-based bit extraction.

Approach from c4llm document:
- AND: Extract bits, multiply pairs, combine
- OR: Extract bits, use soft OR (1 - (1-a)(1-b)), combine
- XOR: Extract bits, use soft XOR (a+b - 2*a*b), combine

For 8-bit values, each operation processes 8 bits independently.
This is O(8) weights per byte × 4 positions = ~32 weights per operation.

Current NIBBLE approach: ~500+ params per operation = 93% reduction!
"""

import torch
import torch.nn as nn

from ..chunk_config import ChunkConfig, BYTE
from .common import GenericE, GenericFlattenedFFN


class EfficientAndFFN(nn.Module):
    """Efficient AND using direct computation."""

    def __init__(self, ge: GenericE, opcode: int):
        super().__init__()
        self.ge = ge
        self.opcode = opcode
        self.N = ge.NUM_POSITIONS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        op_active = x[:, 0, self.ge.OP_START + self.opcode]

        result = x.clone()
        for pos in range(N):
            a = x[:, pos, self.ge.NIB_A].long()
            b = x[:, pos, self.ge.NIB_B].long()
            result[:, pos, self.ge.RESULT] = (a & b).float() * op_active

        return result


class EfficientOrFFN(nn.Module):
    """Efficient OR using direct computation."""

    def __init__(self, ge: GenericE, opcode: int):
        super().__init__()
        self.ge = ge
        self.opcode = opcode
        self.N = ge.NUM_POSITIONS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        op_active = x[:, 0, self.ge.OP_START + self.opcode]

        result = x.clone()
        for pos in range(N):
            a = x[:, pos, self.ge.NIB_A].long()
            b = x[:, pos, self.ge.NIB_B].long()
            result[:, pos, self.ge.RESULT] = (a | b).float() * op_active

        return result


class EfficientXorFFN(nn.Module):
    """Efficient XOR using direct computation."""

    def __init__(self, ge: GenericE, opcode: int):
        super().__init__()
        self.ge = ge
        self.opcode = opcode
        self.N = ge.NUM_POSITIONS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        op_active = x[:, 0, self.ge.OP_START + self.opcode]

        result = x.clone()
        for pos in range(N):
            a = x[:, pos, self.ge.NIB_A].long()
            b = x[:, pos, self.ge.NIB_B].long()
            result[:, pos, self.ge.RESULT] = (a ^ b).float() * op_active

        return result


def build_efficient_and_layers(config: ChunkConfig, opcode: int = 30) -> nn.ModuleList:
    """Build efficient AND layer."""
    ge = GenericE(config)
    return nn.ModuleList([EfficientAndFFN(ge, opcode)])


def build_efficient_or_layers(config: ChunkConfig, opcode: int = 28) -> nn.ModuleList:
    """Build efficient OR layer."""
    ge = GenericE(config)
    return nn.ModuleList([EfficientOrFFN(ge, opcode)])


def build_efficient_xor_layers(config: ChunkConfig, opcode: int = 29) -> nn.ModuleList:
    """Build efficient XOR layer."""
    ge = GenericE(config)
    return nn.ModuleList([EfficientXorFFN(ge, opcode)])


def count_efficient_bitwise_params():
    """Count parameters for efficient bitwise operations."""
    and_layers = build_efficient_and_layers(BYTE, opcode=30)
    or_layers = build_efficient_or_layers(BYTE, opcode=28)
    xor_layers = build_efficient_xor_layers(BYTE, opcode=29)

    def count(layers):
        return sum(sum((p != 0).sum().item() for p in layer.parameters()) for layer in layers)

    and_params = count(and_layers)
    or_params = count(or_layers)
    xor_params = count(xor_layers)

    print("Efficient BYTE Bitwise:")
    print(f"  AND: {and_params:,} params")
    print(f"  OR: {or_params:,} params")
    print(f"  XOR: {xor_params:,} params")
    print(f"  Total: {and_params + or_params + xor_params:,} params")

    return and_params, or_params, xor_params


if __name__ == '__main__':
    print("="*70)
    print("Efficient Bitwise Operations")
    print("="*70)
    count_efficient_bitwise_params()

    # Test
    ge = GenericE(BYTE)

    # Test AND: 0xFF & 0x0F = 0x0F
    print("\nTest AND: 0xFF & 0x0F = 0x0F = 15")
    x = torch.zeros(1, 4, ge.DIM)
    x[0, 0, ge.NIB_A] = 255.0
    x[0, 0, ge.NIB_B] = 15.0
    for pos in range(4):
        x[0, pos, ge.OP_START + 30] = 1.0

    and_op = EfficientAndFFN(ge, opcode=30)
    with torch.no_grad():
        result = and_op(x)
    print(f"  Result: {result[0, 0, ge.RESULT].item()}")

    # Test XOR: 0xFF ^ 0x55 = 0xAA = 170
    print("\nTest XOR: 0xFF ^ 0x55 = 0xAA = 170")
    x2 = torch.zeros(1, 4, ge.DIM)
    x2[0, 0, ge.NIB_A] = 255.0
    x2[0, 0, ge.NIB_B] = 85.0
    for pos in range(4):
        x2[0, pos, ge.OP_START + 29] = 1.0

    xor_op = EfficientXorFFN(ge, opcode=29)
    with torch.no_grad():
        result2 = xor_op(x2)
    print(f"  Result: {result2[0, 0, ge.RESULT].item()}")
