"""
Bitwise operations using transformer primitives.

Three-stage FFN approach:
1. Extract bits using shifts
2. Bitwise ops on bit vectors
3. Recombine to integer

All using only: shifts, multiplication (SwiGLU), addition
"""

import torch
import torch.nn as nn

def silu(x):
    return x * torch.sigmoid(x)

def swiglu_mul(a, b):
    """a * b using SwiGLU identity"""
    a, b = a.float(), b.float()
    return a * silu(b) - a * silu(-b)


class BitExtractor(nn.Module):
    """
    FFN 1: Extract bits from integer.

    bit_i = (a >> i) - ((a >> (i+1)) << 1)

    This works because:
    - (a >> i) has bit_i in position 0
    - (a >> (i+1)) << 1 has bits above i, shifted to remove bit_i
    - Difference isolates bit_i
    """

    def __init__(self, num_bits=64):
        super().__init__()
        self.num_bits = num_bits
        # Precompute powers of 2 for shifts
        self.register_buffer('powers', 2.0 ** torch.arange(num_bits))
        self.register_buffer('positions', torch.arange(num_bits))

    def forward(self, a):
        """Extract all bits from integer a. Returns tensor of shape (num_bits,)"""
        a = a.float()

        bits = []
        for i in range(self.num_bits):
            # Right shift by i: a // 2^i
            shifted = torch.floor(a / self.powers[i])
            # Right shift by i+1, then left shift by 1: ((a // 2^(i+1)) * 2)
            if i + 1 < self.num_bits:
                higher = torch.floor(a / self.powers[i + 1]) * 2
            else:
                higher = torch.tensor(0.0)
            # Difference is bit i
            bit_i = shifted - higher
            bits.append(bit_i)

        return torch.stack(bits)


class BitRecombiner(nn.Module):
    """
    FFN 3: Recombine bits to integer.

    result = sum(bit_i * 2^i)
    """

    def __init__(self, num_bits=64):
        super().__init__()
        self.num_bits = num_bits
        self.register_buffer('powers', 2.0 ** torch.arange(num_bits))

    def forward(self, bits):
        """Combine bit vector to integer."""
        # bits: (num_bits,)
        # Multiply each bit by its power of 2 and sum
        result = (bits * self.powers).sum()
        return result.round().long()


class BitwiseOps(nn.Module):
    """
    FFN 2: Bitwise operations on bit vectors.

    AND: a * b (SwiGLU mul)
    OR:  a + b - a*b
    XOR: a + b - 2*a*b
    NOT: 1 - a
    """

    def bit_and(self, a_bits, b_bits):
        """AND: element-wise multiplication"""
        result = []
        for i in range(len(a_bits)):
            result.append(swiglu_mul(a_bits[i], b_bits[i]))
        return torch.stack(result)

    def bit_or(self, a_bits, b_bits):
        """OR: a + b - a*b"""
        result = []
        for i in range(len(a_bits)):
            ab = swiglu_mul(a_bits[i], b_bits[i])
            result.append(a_bits[i] + b_bits[i] - ab)
        return torch.stack(result)

    def bit_xor(self, a_bits, b_bits):
        """XOR: a + b - 2*a*b"""
        result = []
        for i in range(len(a_bits)):
            ab = swiglu_mul(a_bits[i], b_bits[i])
            result.append(a_bits[i] + b_bits[i] - 2 * ab)
        return torch.stack(result)

    def bit_not(self, a_bits):
        """NOT: 1 - a"""
        return 1 - a_bits


class TransformerBitwise(nn.Module):
    """
    Complete bitwise operations using 3 FFN stages.

    Stage 1: Extract bits from both operands
    Stage 2: Apply bitwise operation
    Stage 3: Recombine to integer
    """

    def __init__(self, num_bits=64):
        super().__init__()
        self.extractor = BitExtractor(num_bits)
        self.ops = BitwiseOps()
        self.recombiner = BitRecombiner(num_bits)

    def bitwise_and(self, a, b):
        a_bits = self.extractor(a)
        b_bits = self.extractor(b)
        result_bits = self.ops.bit_and(a_bits, b_bits)
        return self.recombiner(result_bits)

    def bitwise_or(self, a, b):
        a_bits = self.extractor(a)
        b_bits = self.extractor(b)
        result_bits = self.ops.bit_or(a_bits, b_bits)
        return self.recombiner(result_bits)

    def bitwise_xor(self, a, b):
        a_bits = self.extractor(a)
        b_bits = self.extractor(b)
        result_bits = self.ops.bit_xor(a_bits, b_bits)
        return self.recombiner(result_bits)


def test_bitwise():
    print("BITWISE OPERATIONS (3-stage FFN)")
    print("=" * 50)
    print()
    print("Stage 1: Extract bits using shifts")
    print("Stage 2: Bitwise ops (SwiGLU multiplication)")
    print("Stage 3: Recombine using powers of 2")
    print()

    bw = TransformerBitwise(num_bits=16)  # 16 bits for testing

    # Test bit extraction first
    print("Bit extraction test:")
    extractor = BitExtractor(num_bits=8)
    for val in [5, 12, 255]:
        bits = extractor(torch.tensor(val))
        bits_str = ''.join([str(int(b.item())) for b in bits])
        expected = format(val, '08b')[::-1]  # reversed for LSB first
        status = "✓" if bits_str == expected else "✗"
        print(f"  {status} {val:3d} -> {bits_str} (expected {expected})")

    print()
    print("AND tests:")
    tests_and = [
        (0b1111, 0b1010, 0b1010),
        (0b1100, 0b0110, 0b0100),
        (0xFF, 0x0F, 0x0F),
        (12, 10, 8),
        (255, 128, 128),
    ]
    for a, b, expected in tests_and:
        result = bw.bitwise_and(torch.tensor(a), torch.tensor(b)).item()
        status = "✓" if result == expected else "✗"
        print(f"  {status} {a:5d} & {b:5d} = {result:5d}  (expected {expected})")

    print()
    print("OR tests:")
    tests_or = [
        (0b1100, 0b0011, 0b1111),
        (0b1010, 0b0101, 0b1111),
        (12, 3, 15),
        (8, 4, 12),
    ]
    for a, b, expected in tests_or:
        result = bw.bitwise_or(torch.tensor(a), torch.tensor(b)).item()
        status = "✓" if result == expected else "✗"
        print(f"  {status} {a:5d} | {b:5d} = {result:5d}  (expected {expected})")

    print()
    print("XOR tests:")
    tests_xor = [
        (0b1111, 0b1010, 0b0101),
        (0b1100, 0b0110, 0b1010),
        (15, 10, 5),
        (255, 255, 0),
        (12, 5, 9),
    ]
    for a, b, expected in tests_xor:
        result = bw.bitwise_xor(torch.tensor(a), torch.tensor(b)).item()
        status = "✓" if result == expected else "✗"
        print(f"  {status} {a:5d} ^ {b:5d} = {result:5d}  (expected {expected})")


if __name__ == "__main__":
    test_bitwise()
