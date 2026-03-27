"""
Pure neural bitwise operations (AND, OR, XOR) using SwiGLU.

For bit extraction:
    bit_k = floor(v / 2^k) mod 2

For bitwise ops on binary values:
    AND(a,b) = a * b
    OR(a,b) = a + b - a*b
    XOR(a,b) = a + b - 2*a*b

Each bit operation uses SwiGLU: silu(S*a) * b / S ≈ a*b for binary a,b.

For 8-bit values:
    - 8 bit extractions per input = 16 total
    - 8 bit operations
    - 8 bit combinations
    Total: ~80-100 params per operation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..chunk_config import ChunkConfig, BYTE
from .common import GenericE, GenericFlattenedFFN, bake_clear_pair


SCALE = 100.0
EPS = 0.1


def bit_extract_neural(v, k, scale=SCALE, eps=EPS):
    """
    Extract bit k from integer value v using pure neural ops.

    bit_k = floor(v / 2^k) mod 2

    Using nibble floor formula:
        quotient = round(SiLU(S*(v/2^k - 1 + eps))/S + 1 - eps - 0.5 + 0.001)
        bit = quotient mod 2

    For mod 2, we use: bit = quotient - 2 * floor(quotient / 2)
    """
    divisor = 2 ** k

    # floor(v / 2^k) using nibble rounding
    raw_q = F.silu(scale * (v / divisor - 1 + eps)) / scale + 1 - eps
    quotient = torch.round(raw_q - 0.5 + 0.001)

    # mod 2: check if quotient is odd
    # floor(quotient / 2) using nibble rounding
    raw_half = F.silu(scale * (quotient / 2 - 1 + eps)) / scale + 1 - eps
    half_q = torch.round(raw_half - 0.5 + 0.001)

    # bit = quotient - 2 * half_q
    bit = quotient - 2 * half_q

    return bit


def neural_and_bit(a, b, scale=SCALE):
    """AND of binary values: a * b"""
    # SwiGLU multiplication: silu(S*a) * b / S
    return F.silu(scale * a) * b / scale


def neural_or_bit(a, b, scale=SCALE):
    """OR of binary values: a + b - a*b"""
    ab = F.silu(scale * a) * b / scale
    return a + b - ab


def neural_xor_bit(a, b, scale=SCALE):
    """XOR of binary values: a + b - 2*a*b"""
    ab = F.silu(scale * a) * b / scale
    return a + b - 2 * ab


def neural_and_byte(a_byte, b_byte):
    """AND of two bytes using neural bit operations."""
    result = torch.zeros_like(a_byte)
    for k in range(8):
        bit_a = bit_extract_neural(a_byte, k)
        bit_b = bit_extract_neural(b_byte, k)
        bit_result = neural_and_bit(bit_a, bit_b)
        result = result + bit_result * (2 ** k)
    return torch.round(result)


def neural_or_byte(a_byte, b_byte):
    """OR of two bytes using neural bit operations."""
    result = torch.zeros_like(a_byte)
    for k in range(8):
        bit_a = bit_extract_neural(a_byte, k)
        bit_b = bit_extract_neural(b_byte, k)
        bit_result = neural_or_bit(bit_a, bit_b)
        result = result + bit_result * (2 ** k)
    return torch.round(result)


def neural_xor_byte(a_byte, b_byte):
    """XOR of two bytes using neural bit operations."""
    result = torch.zeros_like(a_byte)
    for k in range(8):
        bit_a = bit_extract_neural(a_byte, k)
        bit_b = bit_extract_neural(b_byte, k)
        bit_result = neural_xor_bit(bit_a, bit_b)
        result = result + bit_result * (2 ** k)
    return torch.round(result)


class NeuralBitwiseFFN(nn.Module):
    """
    Pure neural bitwise operation using SwiGLU.

    For each bit position k (0-7):
        1. Extract bit_a = floor(a / 2^k) mod 2
        2. Extract bit_b = floor(b / 2^k) mod 2
        3. Compute result bit based on operation
        4. Accumulate: result += bit_result * 2^k

    Parameters per bit:
        - Bit extraction: ~10 params (2 floor ops × 5 params)
        - Bit operation: ~2 params
        Total per bit: ~12 params

    Total for 8 bits: ~96 params per position × 4 positions = ~384 params
    """

    def __init__(self, config: ChunkConfig, operation: str, opcode: int):
        super().__init__()
        self.config = config
        self.operation = operation
        self.opcode = opcode
        self.ge = GenericE(config)

        # Pre-compute powers of 2
        self.register_buffer('powers', torch.tensor([2**k for k in range(8)], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        op_active = x[:, 0, self.ge.OP_START + self.opcode]

        result = x.clone()
        for pos in range(self.ge.NUM_POSITIONS):
            a = x[:, pos, self.ge.NIB_A]
            b = x[:, pos, self.ge.NIB_B]

            if self.operation == 'and':
                byte_result = neural_and_byte(a, b)
            elif self.operation == 'or':
                byte_result = neural_or_byte(a, b)
            elif self.operation == 'xor':
                byte_result = neural_xor_byte(a, b)
            else:
                raise ValueError(f"Unknown operation: {self.operation}")

            result[:, pos, self.ge.RESULT] = byte_result * op_active

        return result


def build_neural_and_layers(config: ChunkConfig, opcode: int = 30) -> nn.ModuleList:
    """Build neural AND layer."""
    return nn.ModuleList([NeuralBitwiseFFN(config, 'and', opcode)])


def build_neural_or_layers(config: ChunkConfig, opcode: int = 28) -> nn.ModuleList:
    """Build neural OR layer."""
    return nn.ModuleList([NeuralBitwiseFFN(config, 'or', opcode)])


def build_neural_xor_layers(config: ChunkConfig, opcode: int = 29) -> nn.ModuleList:
    """Build neural XOR layer."""
    return nn.ModuleList([NeuralBitwiseFFN(config, 'xor', opcode)])


def test_bitwise():
    """Test neural bitwise operations."""
    print("Testing neural bitwise operations:")
    print("-" * 60)

    test_cases = [
        # (a, b, expected_and, expected_or, expected_xor)
        (0xFF, 0x0F, 0x0F, 0xFF, 0xF0),
        (0xFF, 0x55, 0x55, 0xFF, 0xAA),
        (0x00, 0xFF, 0x00, 0xFF, 0xFF),
        (0xAA, 0x55, 0x00, 0xFF, 0xFF),
        (123, 45, 123 & 45, 123 | 45, 123 ^ 45),
        (200, 100, 200 & 100, 200 | 100, 200 ^ 100),
    ]

    print(f"{'a':>5} | {'b':>5} | {'AND':>5} | {'OR':>5} | {'XOR':>5} | {'status':>8}")
    print("-" * 50)

    all_pass = True
    for a, b, exp_and, exp_or, exp_xor in test_cases:
        t_a = torch.tensor(float(a))
        t_b = torch.tensor(float(b))

        res_and = neural_and_byte(t_a, t_b).item()
        res_or = neural_or_byte(t_a, t_b).item()
        res_xor = neural_xor_byte(t_a, t_b).item()

        and_ok = abs(res_and - exp_and) < 0.5
        or_ok = abs(res_or - exp_or) < 0.5
        xor_ok = abs(res_xor - exp_xor) < 0.5

        status = "✓" if (and_ok and or_ok and xor_ok) else "✗"
        all_pass = all_pass and and_ok and or_ok and xor_ok

        print(f"{a:>5} | {b:>5} | {int(res_and):>5} | {int(res_or):>5} | {int(res_xor):>5} | {status:>8}")

        if not (and_ok and or_ok and xor_ok):
            print(f"  Expected: AND={exp_and}, OR={exp_or}, XOR={exp_xor}")

    print("-" * 50)
    print(f"All tests passed: {all_pass}")
    return all_pass


def test_bit_extract():
    """Test bit extraction."""
    print("\nTesting bit extraction:")
    print("-" * 60)

    test_values = [0, 1, 127, 128, 255, 170, 85]

    all_pass = True
    for v in test_values:
        bits = []
        for k in range(8):
            bit = bit_extract_neural(torch.tensor(float(v)), k).item()
            bits.append(int(round(bit)))

        # Reconstruct and check
        reconstructed = sum(b * (2**k) for k, b in enumerate(bits))
        expected_bits = [(v >> k) & 1 for k in range(8)]

        ok = reconstructed == v
        all_pass = all_pass and ok
        status = "✓" if ok else "✗"

        print(f"  {v:3d} = {v:08b} -> bits={bits} -> reconstructed={reconstructed} {status}")

    print(f"All bit extractions passed: {all_pass}")
    return all_pass


if __name__ == '__main__':
    print("=" * 70)
    print("Neural Bitwise Operations")
    print("=" * 70)

    test_bit_extract()
    print()
    test_bitwise()
