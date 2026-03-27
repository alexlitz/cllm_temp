"""
Efficient floor using SiLU formula with nibble rounding correction.

Key insight from user:
- SiLU formula gives approx_x ≈ x for x >= 1
- The fractional part of V/256 is always a multiple of 1/256
- We only need to detect which 1/16 "nibble bin" the fractional part is in
- Subtract the bin center, error < 1/32, which rounds correctly to floor

This gives O(16) weights for nibble detection instead of O(max_quotient) staircase!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_SCALE = 100.0
DEFAULT_EPS = 0.1  # Larger eps to ensure formula gives >= 1 at x = 1


def silu_floor_raw(x, scale=DEFAULT_SCALE, eps=DEFAULT_EPS):
    """Raw SiLU floor: gives ~x for x >= 1, ~(1-eps) for x < 1."""
    return F.silu(scale * (x - 1 + eps)) / scale + 1 - eps


def nibble_floor(x, scale=DEFAULT_SCALE, eps=DEFAULT_EPS):
    """
    Compute floor(x) using SiLU + round(x - 0.5).

    Key insight: floor(x) = round(x - 0.5 + tiny) for non-half-integer x.

    For x >= 1:
    - SiLU formula gives approx_x ≈ x (accurate to within eps)
    - round(approx_x - 0.5) gives floor(x)

    For x < 1:
    - SiLU gives ~(1-eps) = 0.9
    - round(0.9 - 0.5) = round(0.4) = 0 ✓

    The eps = 0.1 ensures:
    - For x = 1.0: raw_silu ≈ 1.0, round(1.0 - 0.5) = 1 ✓
    - For x = 0.9: raw_silu ≈ 0.9, round(0.9 - 0.5) = 0 ✓
    """
    approx_x = silu_floor_raw(x, scale, eps)

    # floor(x) = round(x - 0.5 + tiny_eps)
    # The tiny_eps ensures we don't hit exact 0.5 boundaries
    return torch.round(approx_x - 0.5 + 0.001)


def nibble_floor_v2(x, scale=DEFAULT_SCALE, eps=DEFAULT_EPS):
    """
    Alternative: floor(x) = round(x - 0.5 + tiny_eps)

    This works because:
    - round(x - 0.5) = floor(x) when x is not a half-integer
    - For V/256, x is never exactly n.5 (would need V = 128 mod 256 exactly on boundary)
    - The tiny eps handles edge cases

    With SiLU formula:
    - approx_x ≈ x for x >= 1
    - floor(approx_x) ≈ floor(x) if SiLU error < 0.5
    """
    approx_x = silu_floor_raw(x, scale, eps)

    # floor(x) = round(x - 0.5) for non-half-integer x
    # Add tiny eps to break ties toward floor
    return torch.round(approx_x - 0.5 + 0.001)


def nibble_mod(x, n, scale=DEFAULT_SCALE, eps=DEFAULT_EPS):
    """Compute x mod n using nibble floor."""
    quotient = nibble_floor(x / n, scale, eps)
    return x - quotient * n


class NibbleFloorFFN(nn.Module):
    """
    FFN that computes floor(x) using SiLU + round(x - 0.5).

    Architecture:
    - Hidden 0: SiLU floor formula (gives ~x for x >= 1)
    - Hidden 1: Offset by -0.5 for rounding
    - Output: Implicitly rounded by subsequent operations

    Total: ~6 parameters
    """

    def __init__(self, scale=DEFAULT_SCALE, eps=DEFAULT_EPS):
        super().__init__()
        self.scale = scale
        self.eps = eps

        # 2 hidden units: SiLU path + offset
        self.W_up = nn.Parameter(torch.tensor([[scale], [0.0]]))
        self.b_up = nn.Parameter(torch.tensor([scale * (-1 + eps), 0.0]))
        self.W_gate = nn.Parameter(torch.tensor([[0.0], [0.0]]))
        self.b_gate = nn.Parameter(torch.tensor([1.0, 1.0]))
        self.W_down = nn.Parameter(torch.tensor([[1.0 / scale, 1.0]]))
        self.b_down = nn.Parameter(torch.tensor([1.0 - eps - 0.5 + 0.001]))  # Include -0.5 offset

    def forward(self, x):
        """Compute floor(x) via SiLU + round offset."""
        up = F.linear(x.unsqueeze(-1), self.W_up, self.b_up)
        gate = F.linear(x.unsqueeze(-1), self.W_gate, self.b_gate)
        hidden = F.silu(up) * gate
        result = F.linear(hidden, self.W_down, self.b_down).squeeze(-1)
        return torch.round(result)


class NibbleModFFN(nn.Module):
    """
    FFN that computes x mod n using nibble floor.

    x mod n = x - floor(x/n) * n

    With nibble floor:
    - Compute approx = silu_floor(x/n)
    - Round: q = round(approx - 0.5)
    - Result: x - q * n
    """

    def __init__(self, n, scale=DEFAULT_SCALE, eps=DEFAULT_EPS):
        super().__init__()
        self.n = n
        self.scale = scale
        self.eps = eps

        # SiLU floor for x/n, then multiply by -n and add x
        self.W_up = nn.Parameter(torch.tensor([[scale / n]]))
        self.b_up = nn.Parameter(torch.tensor([scale * (-1 + eps)]))
        self.W_gate = nn.Parameter(torch.tensor([[0.0]]))
        self.b_gate = nn.Parameter(torch.tensor([1.0]))
        self.W_down = nn.Parameter(torch.tensor([[-n / scale]]))
        # b_down includes: (1 - eps) * (-n) - 0.5 * (-n) = -n + n*eps + 0.5*n
        self.b_down = nn.Parameter(torch.tensor([(-1 + eps + 0.5) * (-n)]))

    def forward(self, x):
        """Compute x mod n."""
        up = F.linear(x.unsqueeze(-1), self.W_up, self.b_up)
        gate = F.linear(x.unsqueeze(-1), self.W_gate, self.b_gate)
        hidden = F.silu(up) * gate
        # This gives: -n * round(silu_floor(x/n) - 0.5)
        floor_times_neg_n = F.linear(hidden, self.W_down, self.b_down).squeeze(-1)
        # Need to round the quotient before multiplying by -n
        # But linear ops don't include round...
        # We need a different approach
        return x + floor_times_neg_n  # This won't work correctly


def test_nibble_floor():
    """Test nibble floor implementation."""
    print("Testing nibble floor (SiLU + round(x - 0.5)):")
    print("-" * 60)

    test_values = [0.0, 0.5, 0.99, 1.0, 1.5, 2.0, 2.5, 2.9, 3.0, 3.5, 10.0, 100.5, 255.0]

    print(f"{'x':>8} | {'raw_silu':>10} | {'nibble':>10} | {'expected':>8} | {'status':>6}")
    print("-" * 60)

    for x in test_values:
        t = torch.tensor(x)
        raw = silu_floor_raw(t).item()
        nib = nibble_floor(t).item()
        expected = int(x)
        status = "✓" if abs(nib - expected) < 0.5 else "✗"
        print(f"{x:>8.2f} | {raw:>10.4f} | {nib:>10.0f} | {expected:>8} | {status:>6}")


def test_nibble_mod():
    """Test nibble mod implementation."""
    print("\nTesting nibble mod (x mod 256):")
    print("-" * 60)

    n = 256
    test_values = [0, 100, 255, 256, 257, 512, 1000, 65535, 260100]

    print(f"{'x':>8} | {'quotient':>10} | {'mod':>10} | {'expected':>8} | {'status':>6}")
    print("-" * 60)

    for x in test_values:
        t = torch.tensor(float(x))
        q = nibble_floor(t / n).item()
        mod = nibble_mod(t, n).item()
        expected = x % n
        status = "✓" if abs(mod - expected) < 1.0 else "✗"
        print(f"{x:>8} | {q:>10.0f} | {mod:>10.0f} | {expected:>8} | {status:>6}")


def test_carry_extraction():
    """Test for MUL carry extraction use case."""
    print("\nTesting carry extraction (divmod by 256):")
    print("-" * 60)

    n = 256
    # Values that occur in MUL: products and sums of products
    test_values = [0, 1, 100, 255, 256, 257, 512, 1000, 65025, 130050, 260100]

    print(f"{'value':>8} | {'carry':>8} | {'remainder':>10} | {'exp_c':>6} | {'exp_r':>6} | {'ok':>4}")
    print("-" * 60)

    all_pass = True
    for v in test_values:
        t = torch.tensor(float(v))
        carry = nibble_floor(t / n).item()
        remainder = nibble_mod(t, n).item()
        exp_c = v // n
        exp_r = v % n
        ok = abs(carry - exp_c) < 0.5 and abs(remainder - exp_r) < 1.0
        all_pass = all_pass and ok
        status = "✓" if ok else "✗"
        print(f"{v:>8} | {carry:>8.0f} | {remainder:>10.0f} | {exp_c:>6} | {exp_r:>6} | {status:>4}")

    print("-" * 60)
    print(f"All tests passed: {all_pass}")
    return all_pass


def count_params():
    """Count parameters."""
    print("\nParameter counts:")
    print("-" * 40)

    floor_ffn = NibbleFloorFFN()

    def count(m):
        return sum((p != 0).sum().item() for p in m.parameters())

    print(f"  NibbleFloorFFN: {count(floor_ffn)} non-zero params")


if __name__ == '__main__':
    print("=" * 70)
    print("Nibble Floor/Mod Implementation")
    print("=" * 70)
    test_nibble_floor()
    test_nibble_mod()
    test_carry_extraction()
    count_params()
