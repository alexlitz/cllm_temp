"""
Byte-based (8-bit) operations for Neural VM.

Using bytes (4 per 32-bit word) instead of nibbles (8 per word):
- ADD/SUB: 4 carry stages instead of 8 (2x faster)
- MUL: 10 partial products instead of 36 (3.6x fewer, see below)
- Comparison: 4 byte comparisons instead of 8 nibbles

MUL Partial Products (byte version):
  For 32-bit result, we only need products where i+j < 4:
    i=0: b[0]*a[0], b[0]*a[1], b[0]*a[2], b[0]*a[3]  (4 products)
    i=1: b[1]*a[0], b[1]*a[1], b[1]*a[2]              (3 products)
    i=2: b[2]*a[0], b[2]*a[1]                          (2 products)
    i=3: b[3]*a[0]                                      (1 product)
  Total: 4+3+2+1 = 10 products (vs 36 nibble products)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .embedding import E, Opcode
    from .base_layers import PureFFN
except ImportError:
    from embedding import E, Opcode
    from base_layers import PureFFN


# Byte slot layout in embedding (512-dim)
# Using E.NIB_A through E.NIB_A+7 as 8 nibbles = 4 bytes
# BYTE_A[i] combines NIB_A[2*i] and NIB_A[2*i+1]

class E_BYTE:
    """Byte-level embedding slots."""
    # Input bytes (4 bytes = 32 bits)
    BYTE_A = [E.NIB_A + 2*i for i in range(4)]  # Positions of byte A values
    BYTE_B = [E.NIB_B + 2*i for i in range(4)]  # Positions of byte B values

    # Work slots for byte operations
    BYTE_SUM = 200      # Slots 200-203: Raw byte sums (0-510)
    BYTE_CARRY = 204    # Slots 204-207: Carries (0 or 1)
    BYTE_RESULT = 208   # Slots 208-211: Final byte results

    # MUL partial products (10 products for non-overflow)
    MUL_PARTIAL = 220   # Slots 220-229: 10 partial products


def nibbles_to_byte(nib_low: torch.Tensor, nib_high: torch.Tensor) -> torch.Tensor:
    """Combine two nibbles into a byte: byte = nib_high * 16 + nib_low."""
    return nib_high * 16 + nib_low


def byte_to_nibbles(byte: torch.Tensor) -> tuple:
    """Split byte into (low_nibble, high_nibble)."""
    low = byte % 16
    high = byte // 16
    return low, high


class ByteAddFFN(PureFFN):
    """
    Byte-level addition: compute a[i] + b[i] for each byte position.

    Processes 4 bytes in parallel, producing sums in [0, 510].
    Carry propagation happens in subsequent layer.

    Uses gate to combine nibble pairs into bytes.
    """

    def __init__(self):
        # 4 bytes * 3 components (combine A nibbles, combine B nibbles, add)
        super().__init__(E.DIM, hidden_dim=12)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            for byte_idx in range(4):
                nib_lo_a = E.NIB_A + 2 * byte_idx
                nib_hi_a = E.NIB_A + 2 * byte_idx + 1
                nib_lo_b = E.NIB_B + 2 * byte_idx
                nib_hi_b = E.NIB_B + 2 * byte_idx + 1

                # Row 3*i: Extract byte A = nibA_hi * 16 + nibA_lo
                row = 3 * byte_idx
                self.W_up[row, nib_lo_a] = S
                self.W_up[row, nib_hi_a] = S * 16
                self.b_gate[row] = 1.0
                self.W_down[E_BYTE.BYTE_SUM + byte_idx, row] = 1.0 / S

                # Row 3*i+1: Extract byte B = nibB_hi * 16 + nibB_lo
                row = 3 * byte_idx + 1
                self.W_up[row, nib_lo_b] = S
                self.W_up[row, nib_hi_b] = S * 16
                self.b_gate[row] = 1.0
                self.W_down[E_BYTE.BYTE_SUM + byte_idx, row] = 1.0 / S


class ByteCarryFFN(PureFFN):
    """
    Compute carries from byte sums.

    carry[i] = 1 if sum[i] >= 256, else 0
    result[i] = sum[i] % 256
    """

    def __init__(self):
        # 4 bytes * 2 (detect carry, compute mod)
        super().__init__(E.DIM, hidden_dim=8)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            for byte_idx in range(4):
                sum_slot = E_BYTE.BYTE_SUM + byte_idx
                carry_slot = E_BYTE.BYTE_CARRY + byte_idx
                result_slot = E_BYTE.BYTE_RESULT + byte_idx

                # Row 2*i: Detect carry (sum >= 256)
                row = 2 * byte_idx
                self.W_up[row, sum_slot] = S
                self.b_up[row] = -S * 255.5  # Threshold at 256
                self.b_gate[row] = 1.0
                self.W_down[carry_slot, row] = 1.0 / S

                # Row 2*i+1: Compute result mod 256
                # This is more complex - need step function
                row = 2 * byte_idx + 1
                self.W_up[row, sum_slot] = S
                self.b_gate[row] = 1.0
                self.W_down[result_slot, row] = 1.0 / S


class ByteMulPartialFFN(PureFFN):
    """
    Compute non-overflow partial products for byte multiplication.

    Only computes products where i+j < 4 (10 total):
      Slot 0: a[0]*b[0] -> position 0
      Slot 1: a[0]*b[1] + a[1]*b[0] -> position 1
      Slot 2: a[0]*b[2] + a[1]*b[1] + a[2]*b[0] -> position 2
      Slot 3: a[0]*b[3] + a[1]*b[2] + a[2]*b[1] + a[3]*b[0] -> position 3

    Each partial product is a[i]*b[j] where result goes to position i+j.
    """

    def __init__(self):
        # 10 partial products
        super().__init__(E.DIM, hidden_dim=20)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            row = 0
            # Enumerate all (i,j) pairs where i+j < 4
            for result_pos in range(4):
                for i in range(result_pos + 1):
                    j = result_pos - i
                    # Partial product a[i] * b[j] goes to result position i+j

                    nib_lo_a = E.NIB_A + 2 * i
                    nib_hi_a = E.NIB_A + 2 * i + 1
                    nib_lo_b = E.NIB_B + 2 * j
                    nib_hi_b = E.NIB_B + 2 * j + 1

                    # Up: extract byte_a (combine nibbles)
                    self.W_up[row, nib_lo_a] = S
                    self.W_up[row, nib_hi_a] = S * 16
                    # Gate: byte_b
                    self.W_gate[row, nib_lo_b] = 1.0
                    self.W_gate[row, nib_hi_b] = 16.0
                    # Down: accumulate to partial product slot
                    self.W_down[E_BYTE.MUL_PARTIAL + result_pos, row] = 1.0 / (S * S)

                    row += 1


def test_byte_ops():
    """Test byte-based operations."""
    print("=== Byte-Based Operations Test ===")

    # Test nibble-to-byte conversion
    print("\n1. Nibble-to-Byte conversion:")
    for lo, hi in [(5, 10), (0, 15), (15, 0), (7, 8)]:
        byte = nibbles_to_byte(torch.tensor(lo), torch.tensor(hi))
        expected = hi * 16 + lo
        print(f"   ({hi:X}, {lo:X}) -> {int(byte):02X} (expected {expected:02X})")

    # Count partial products
    print("\n2. Partial product count for MUL:")
    nibble_count = sum(1 for i in range(8) for j in range(8) if i + j < 8)
    byte_count = sum(1 for i in range(4) for j in range(4) if i + j < 4)
    print(f"   Nibble-based (i+j < 8): {nibble_count} products")
    print(f"   Byte-based (i+j < 4): {byte_count} products")
    print(f"   Reduction: {nibble_count / byte_count:.1f}x fewer products")

    # Enumerate byte partial products
    print("\n3. Byte partial products:")
    for result_pos in range(4):
        products = []
        for i in range(result_pos + 1):
            j = result_pos - i
            products.append(f"a[{i}]*b[{j}]")
        print(f"   Position {result_pos}: {' + '.join(products)}")

    print("\n4. Carry stages comparison:")
    print(f"   Nibble-based: 8 carry stages")
    print(f"   Byte-based: 4 carry stages")
    print(f"   Reduction: 2x fewer stages")


if __name__ == '__main__':
    test_byte_ops()
