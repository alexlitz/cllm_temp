"""
Transformer-native implementations of arithmetic operations.

All operations use only:
- Linear projections (matmul)
- SiLU activation
- Addition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def silu(x):
    """SiLU activation: x * sigmoid(x)"""
    return x * torch.sigmoid(x)


class TransformerMul(nn.Module):
    """
    Multiplication using SwiGLU structure.

    a * b = a * SiLU(b) - a * SiLU(-b)

    Proof:
      SiLU(b) - SiLU(-b) = b * sigmoid(b) + b * sigmoid(-b)
                         = b * (sigmoid(b) + sigmoid(-b))
                         = b * 1  (since sigmoid(x) + sigmoid(-x) = 1)
                         = b

      Therefore: a * SiLU(b) - a * SiLU(-b) = a * b
    """

    def __init__(self):
        super().__init__()
        # Wired weights to extract a and b from input [a, b]
        # W_up extracts a, W_gate extracts b
        self.W_up = nn.Parameter(torch.tensor([[1.0], [0.0]]))      # extracts a
        self.W_gate = nn.Parameter(torch.tensor([[0.0], [1.0]]))    # extracts b
        self.W_gate_neg = nn.Parameter(torch.tensor([[0.0], [-1.0]]))  # extracts -b
        self.W_down = nn.Parameter(torch.tensor([[1.0], [-1.0]]))   # combines: +term1 - term2

    def forward(self, a, b):
        """Compute a * b using SwiGLU structure."""
        x = torch.stack([a.float(), b.float()], dim=-1)  # (*, 2)

        up = x @ self.W_up           # a
        gate_pos = x @ self.W_gate   # b
        gate_neg = x @ self.W_gate_neg  # -b

        term1 = up * silu(gate_pos)  # a * SiLU(b)
        term2 = up * silu(gate_neg)  # a * SiLU(-b)

        # Combine: term1 - term2 = a * b
        result = torch.cat([term1, term2], dim=-1) @ self.W_down

        return result.squeeze(-1).long()


class TransformerMulDirect(nn.Module):
    """Simpler version - just show the math works."""

    def forward(self, a, b):
        a, b = a.float(), b.float()
        result = a * silu(b) - a * silu(-b)
        return result.round().long()


def test_mul():
    print("TRANSFORMER MULTIPLICATION")
    print("=" * 50)
    print()
    print("Using: a * b = a * SiLU(b) - a * SiLU(-b)")
    print()

    mul = TransformerMulDirect()

    tests = [
        (3, 4, 12),
        (5, 7, 35),
        (10, 10, 100),
        (0, 5, 0),
        (5, 0, 0),
        (-3, 4, -12),
        (-3, -4, 12),
        (100, 50, 5000),
        (7, 8, 56),
    ]

    passed = 0
    for a, b, expected in tests:
        a_t = torch.tensor(a)
        b_t = torch.tensor(b)
        result = mul(a_t, b_t).item()
        ok = result == expected
        passed += ok
        status = "✓" if ok else "✗"
        print(f"  {status} {a:4d} * {b:4d} = {result:6d}  (expected {expected})")

    print()
    print(f"Passed: {passed}/{len(tests)}")
    print()

    # Show the components
    print("Breakdown for 3 * 4:")
    a, b = torch.tensor(3.0), torch.tensor(4.0)
    term1 = a * silu(b)
    term2 = a * silu(-b)
    print(f"  a * SiLU(b)  = {a.item()} * SiLU({b.item()}) = {a.item()} * {silu(b).item():.4f} = {term1.item():.4f}")
    print(f"  a * SiLU(-b) = {a.item()} * SiLU({-b.item()}) = {a.item()} * {silu(-b).item():.4f} = {term2.item():.4f}")
    print(f"  term1 - term2 = {term1.item():.4f} - {term2.item():.4f} = {(term1 - term2).item():.4f}")
    print()

    print("Why it works:")
    print(f"  sigmoid({b.item()}) = {torch.sigmoid(b).item():.4f}")
    print(f"  sigmoid({-b.item()}) = {torch.sigmoid(-b).item():.4f}")
    print(f"  sum = {(torch.sigmoid(b) + torch.sigmoid(-b)).item():.4f}  (always 1!)")


class TransformerShiftSwiGLU(nn.Module):
    """
    Shift using ONLY SiLU (no sigmoid).

    a << b = a * 2^b

    Structure:
      For each possible shift amount i in [0, 63]:
        - gate_i activates when b == i (using SiLU-based pulse)
        - up_i = a * 2^i (precomputed weight)
        - output += up_i * gate_i

    The gating uses SiLU differences to create soft one-hot:
      threshold(x) = (SiLU(s*(x+0.5)) - SiLU(s*(x-0.5))) / s
      pulse(x) = threshold(x + 0.5) * threshold(-x + 0.5)
               ≈ 1 when |x| < 0.5, ≈ 0 otherwise
    """

    def __init__(self, max_shift=64):
        super().__init__()
        self.max_shift = max_shift

        # W_up: projects a to [a*2^0, a*2^1, ..., a*2^63]
        W_up = torch.zeros(1, max_shift)
        W_up[0, :] = torch.tensor([2.0**i for i in range(max_shift)])
        self.register_buffer('W_up', W_up)

        # Positions for gating
        self.register_buffer('positions', torch.arange(max_shift).float())

        # Scale for sharp gating
        self.scale = 20.0

    def _silu_threshold(self, x):
        """
        Soft threshold using SiLU differences.
        Returns ≈1 when x > 0, ≈0 when x < 0.

        threshold(x) = (SiLU(s*(x+0.5)) - SiLU(s*(x-0.5))) / s
        """
        s = self.scale
        upper = silu(s * (x + 0.5))
        lower = silu(s * (x - 0.5))
        return (upper - lower) / s

    def _silu_pulse(self, x):
        """
        Soft pulse (one-hot) using SiLU.
        Returns ≈1 when |x| < 0.5, ≈0 otherwise.

        pulse(x) = threshold(x + 0.5) * threshold(-x + 0.5)

        Multiplication done via SwiGLU: a*b = a*SiLU(b) - a*SiLU(-b)
        """
        # Two thresholds that overlap only near x=0
        t1 = self._silu_threshold(x + 0.5)   # ≈1 when x > -0.5
        t2 = self._silu_threshold(-x + 0.5)  # ≈1 when x < 0.5

        # Multiply using SwiGLU identity
        pulse = t1 * silu(t2) - t1 * silu(-t2)
        return pulse

    def _soft_onehot(self, b):
        """
        Create soft one-hot vector using only SiLU.
        gate_i ≈ 1 when i == b, ≈ 0 otherwise.
        """
        b = b.float()
        diff = b - self.positions  # (max_shift,)
        gates = self._silu_pulse(diff)
        return gates

    def left_shift(self, a, b):
        """a << b using only SiLU ops"""
        a, b = a.float(), b.float()

        # up = [a*1, a*2, a*4, a*8, ...]
        up = a * self.W_up.squeeze(0)

        # gates via SiLU pulse
        gates = self._soft_onehot(b)

        # Gated sum
        result = (up * gates).sum()

        return result.round().long()

    def right_shift(self, a, b):
        """a >> b = a // 2^b using only SiLU ops"""
        a, b = a.float(), b.float()

        # down = [a/1, a/2, a/4, ...]
        divisors = self.W_up.squeeze(0)
        down = a / divisors

        # gates via SiLU pulse
        gates = self._soft_onehot(b)

        # Gated sum
        result = (down * gates).sum()

        return torch.floor(result).long()


def test_shift():
    print()
    print("TRANSFORMER SHIFT (SwiGLU only)")
    print("=" * 50)
    print()
    print("Using:")
    print("  up_i = a * 2^i           (wired W_up)")
    print("  gate_i = soft_onehot(b)  (SiLU diff pulse)")
    print("  result = sum(up_i * gate_i)")
    print()

    shift = TransformerShiftSwiGLU()

    print("LEFT SHIFT (a << b):")
    tests_left = [
        (1, 0, 1),
        (1, 1, 2),
        (1, 4, 16),
        (3, 2, 12),
        (5, 3, 40),
        (7, 4, 112),
        (1, 10, 1024),
    ]

    for a, b, expected in tests_left:
        result = shift.left_shift(torch.tensor(a), torch.tensor(b)).item()
        status = "✓" if result == expected else "✗"
        print(f"  {status} {a:3d} << {b:2d} = {result:6d}  (expected {expected})")

    print()
    print("RIGHT SHIFT (a >> b):")
    tests_right = [
        (16, 0, 16),
        (16, 1, 8),
        (16, 2, 4),
        (16, 4, 1),
        (100, 2, 25),
        (1024, 10, 1),
        (255, 4, 15),
    ]

    for a, b, expected in tests_right:
        result = shift.right_shift(torch.tensor(a), torch.tensor(b)).item()
        status = "✓" if result == expected else "✗"
        print(f"  {status} {a:4d} >> {b:2d} = {result:5d}  (expected {expected})")


class TransformerCompare(nn.Module):
    """
    Comparisons using SiLU differences.

    Key insight: SiLU(x+c) - SiLU(x-c) creates a smooth step function.

    For integers, with proper shifts:
      SiLU(x + 0.5) - SiLU(x - 0.5) ≈ 1 when x > 0
                                    ≈ 0 when x < 0
                                    ≈ 0.5 when x = 0
    """

    def __init__(self, scale=10.0):
        super().__init__()
        self.scale = scale

    def _threshold(self, x, offset=0.0):
        """
        Soft threshold: ≈1 when x > offset, ≈0 when x < offset

        Using SiLU differences:
          gate = SiLU(scale*(x - offset + 0.5)) - SiLU(scale*(x - offset - 0.5))
        """
        x = x.float()
        diff = self.scale * (x - offset)

        term1 = silu(diff + 0.5 * self.scale)
        term2 = silu(diff - 0.5 * self.scale)

        # Normalize to [0, 1] range
        # When diff >> 0: term1 - term2 ≈ scale (the gap)
        # When diff << 0: term1 - term2 ≈ 0
        gate = (term1 - term2) / self.scale

        return gate

    def gt(self, a, b):
        """a > b: returns 1 if true, 0 if false"""
        # a > b means (a - b) > 0, i.e., (a - b) >= 1 for integers
        # Threshold at 0.5 so that a-b=1 gives 1, a-b=0 gives 0
        return self._threshold(a - b, offset=0.5).round().long()

    def lt(self, a, b):
        """a < b"""
        return self.gt(b, a)

    def ge(self, a, b):
        """a >= b: threshold at -0.5"""
        return self._threshold(a - b, offset=-0.5).round().long()

    def le(self, a, b):
        """a <= b"""
        return self.ge(b, a)

    def eq(self, a, b):
        """a == b: 1 when |a-b| < 0.5"""
        # Use pulse: gate at -0.5 AND gate at +0.5
        # eq = threshold(a-b, -0.5) * threshold(-(a-b), -0.5)
        #    = ge(a, b) * le(a, b)
        diff = (a - b).float()
        upper = self._threshold(diff, offset=-0.5)  # a >= b
        lower = self._threshold(-diff, offset=-0.5)  # a <= b
        # Both must be true for equality
        return (upper * lower).round().long()

    def ne(self, a, b):
        """a != b"""
        return 1 - self.eq(a, b)


def test_compare():
    print()
    print("TRANSFORMER COMPARISONS (SwiGLU only)")
    print("=" * 50)
    print()
    print("Using: gate = SiLU(scale*(x+0.5)) - SiLU(scale*(x-0.5))")
    print()

    cmp = TransformerCompare(scale=10.0)

    print("GT (a > b):")
    tests_gt = [
        (5, 3, 1), (3, 5, 0), (5, 5, 0),
        (0, -1, 1), (-1, 0, 0), (100, 50, 1),
    ]
    for a, b, expected in tests_gt:
        result = cmp.gt(torch.tensor(a), torch.tensor(b)).item()
        status = "✓" if result == expected else "✗"
        print(f"  {status} {a:4d} >  {b:4d} = {result}  (expected {expected})")

    print()
    print("LT (a < b):")
    tests_lt = [
        (3, 5, 1), (5, 3, 0), (5, 5, 0),
    ]
    for a, b, expected in tests_lt:
        result = cmp.lt(torch.tensor(a), torch.tensor(b)).item()
        status = "✓" if result == expected else "✗"
        print(f"  {status} {a:4d} <  {b:4d} = {result}  (expected {expected})")

    print()
    print("EQ (a == b):")
    tests_eq = [
        (5, 5, 1), (5, 3, 0), (0, 0, 1), (-3, -3, 1), (3, -3, 0),
    ]
    for a, b, expected in tests_eq:
        result = cmp.eq(torch.tensor(a), torch.tensor(b)).item()
        status = "✓" if result == expected else "✗"
        print(f"  {status} {a:4d} == {b:4d} = {result}  (expected {expected})")

    print()
    print("NE (a != b):")
    tests_ne = [
        (5, 5, 0), (5, 3, 1), (0, 1, 1),
    ]
    for a, b, expected in tests_ne:
        result = cmp.ne(torch.tensor(a), torch.tensor(b)).item()
        status = "✓" if result == expected else "✗"
        print(f"  {status} {a:4d} != {b:4d} = {result}  (expected {expected})")

    print()
    print("GE (a >= b):")
    tests_ge = [
        (5, 3, 1), (5, 5, 1), (3, 5, 0),
    ]
    for a, b, expected in tests_ge:
        result = cmp.ge(torch.tensor(a), torch.tensor(b)).item()
        status = "✓" if result == expected else "✗"
        print(f"  {status} {a:4d} >= {b:4d} = {result}  (expected {expected})")

    print()
    print("LE (a <= b):")
    tests_le = [
        (3, 5, 1), (5, 5, 1), (5, 3, 0),
    ]
    for a, b, expected in tests_le:
        result = cmp.le(torch.tensor(a), torch.tensor(b)).item()
        status = "✓" if result == expected else "✗"
        print(f"  {status} {a:4d} <= {b:4d} = {result}  (expected {expected})")


if __name__ == "__main__":
    test_mul()
    test_shift()
    test_compare()
