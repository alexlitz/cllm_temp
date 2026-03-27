"""
Efficient floor and mod operations using SwiGLU.

From c4llm document:
    floor(x) ≈ SiLU(SCALE*(x-1+eps))/SCALE + 1 - eps
    x mod N = x - floor(x/N)*N

This gives O(1) weights for floor, not O(max_quotient) step pairs!

Key insight: SiLU(x) ≈ 0 for x<0, SiLU(x) ≈ x for x>0.
The formula uses this threshold behavior to approximate floor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Default parameters - eps=0.1 for nibble rounding compatibility
DEFAULT_SCALE = 100.0
DEFAULT_EPS = 0.1


def silu_floor_raw(x, scale=DEFAULT_SCALE, eps=DEFAULT_EPS):
    """
    Raw SiLU floor approximation (before rounding).

    Returns ~x for x >= 1, ~(1-eps) for x < 1.
    """
    return F.silu(scale * (x - 1 + eps)) / scale + 1 - eps


def silu_floor(x, scale=DEFAULT_SCALE, eps=DEFAULT_EPS):
    """
    Compute floor(x) using SiLU + nibble rounding.

    floor(x) = round(raw - 0.5 + 0.001) where raw ≈ x for x >= 1.
    """
    raw = silu_floor_raw(x, scale, eps)
    return torch.round(raw - 0.5 + 0.001)


def silu_mod(x, n, scale=DEFAULT_SCALE, eps=DEFAULT_EPS):
    """
    Compute x mod n using SiLU floor with nibble rounding.

    x mod n = x - floor(x/n) * n
    """
    quotient = silu_floor(x / n, scale, eps)
    return x - quotient * n


def silu_divmod(x, n, scale=DEFAULT_SCALE, eps=DEFAULT_EPS):
    """
    Compute both floor(x/n) and x mod n.
    """
    quotient = silu_floor(x / n, scale, eps)
    remainder = x - quotient * n
    return quotient, remainder


class FloorFFN(nn.Module):
    """
    FFN that computes floor(x) using SwiGLU + nibble rounding.

    Architecture:
        hidden = silu(W_up @ x + b_up) * (W_gate @ x + b_gate)
        output = W_down @ hidden + b_down
        floor(x) = round(output)

    For floor(x):
        W_up = SCALE, b_up = SCALE*(-1+eps)
        W_gate = 0, b_gate = 1 (constant gate)
        W_down = 1/SCALE, b_down = (1-eps) - 0.5 + 0.001

    The round() in forward() snaps to exact floor.
    """

    def __init__(self, scale=DEFAULT_SCALE, eps=DEFAULT_EPS):
        super().__init__()
        self.scale = scale
        self.eps = eps

        # Single hidden unit for floor
        self.W_up = nn.Parameter(torch.tensor([[scale]]))
        self.b_up = nn.Parameter(torch.tensor([scale * (-1 + eps)]))
        self.W_gate = nn.Parameter(torch.tensor([[0.0]]))
        self.b_gate = nn.Parameter(torch.tensor([1.0]))
        self.W_down = nn.Parameter(torch.tensor([[1.0 / scale]]))
        # Include -0.5 + 0.001 offset for nibble rounding
        self.b_down = nn.Parameter(torch.tensor([1.0 - eps - 0.5 + 0.001]))

    def forward(self, x):
        """Compute floor(x)."""
        up = F.linear(x.unsqueeze(-1), self.W_up, self.b_up)
        gate = F.linear(x.unsqueeze(-1), self.W_gate, self.b_gate)
        hidden = F.silu(up) * gate
        raw = F.linear(hidden, self.W_down, self.b_down).squeeze(-1)
        return torch.round(raw)


class ModFFN(nn.Module):
    """
    FFN that computes x mod n using SwiGLU floor.

    x mod n = x - floor(x/n) * n

    Architecture integrates division into the floor computation:
        W_up = SCALE/n (scales input by 1/n first)
        Then applies floor formula
        Then computes x - floor(x/n) * n
    """

    def __init__(self, n, scale=DEFAULT_SCALE, eps=DEFAULT_EPS):
        super().__init__()
        self.n = n
        self.scale = scale
        self.eps = eps

        # Hidden unit for floor(x/n)
        # Input scaling: x -> x/n is done via W_up = scale/n
        self.W_up = nn.Parameter(torch.tensor([[scale / n]]))
        self.b_up = nn.Parameter(torch.tensor([scale * (-1 + eps)]))
        self.W_gate = nn.Parameter(torch.tensor([[0.0]]))
        self.b_gate = nn.Parameter(torch.tensor([1.0]))

        # Output: floor(x/n) = silu(...)/scale + 1 - eps
        # But we want: x - floor(x/n) * n
        # So: x - (silu(...)/scale + 1 - eps) * n
        # = x - silu(...)*n/scale - (1-eps)*n
        # W_down = -n/scale contributes -silu(...)*n/scale
        # b_down = -(1-eps)*n contributes the constant term
        self.W_down = nn.Parameter(torch.tensor([[-n / scale]]))
        self.b_down = nn.Parameter(torch.tensor([-(1 - eps) * n]))

    def forward(self, x):
        """Compute x mod n."""
        up = F.linear(x.unsqueeze(-1), self.W_up, self.b_up)
        gate = F.linear(x.unsqueeze(-1), self.W_gate, self.b_gate)
        hidden = F.silu(up) * gate
        mod_term = F.linear(hidden, self.W_down, self.b_down).squeeze(-1)
        return x + mod_term  # x + (- floor(x/n) * n)


class DivModFFN(nn.Module):
    """
    FFN that computes both floor(x/n) and x mod n.

    Returns (quotient, remainder).

    Uses shared hidden computation for efficiency.
    """

    def __init__(self, n, scale=DEFAULT_SCALE, eps=DEFAULT_EPS):
        super().__init__()
        self.n = n
        self.scale = scale
        self.eps = eps

        # Shared hidden unit for floor(x/n)
        self.W_up = nn.Parameter(torch.tensor([[scale / n]]))
        self.b_up = nn.Parameter(torch.tensor([scale * (-1 + eps)]))
        self.W_gate = nn.Parameter(torch.tensor([[0.0]]))
        self.b_gate = nn.Parameter(torch.tensor([1.0]))

        # Quotient output
        self.W_down_q = nn.Parameter(torch.tensor([[1.0 / scale]]))
        self.b_down_q = nn.Parameter(torch.tensor([1.0 - eps]))

        # Remainder output: x - floor(x/n)*n
        self.W_down_r = nn.Parameter(torch.tensor([[-n / scale]]))
        self.b_down_r = nn.Parameter(torch.tensor([-(1 - eps) * n]))

    def forward(self, x):
        """Compute (floor(x/n), x mod n)."""
        up = F.linear(x.unsqueeze(-1), self.W_up, self.b_up)
        gate = F.linear(x.unsqueeze(-1), self.W_gate, self.b_gate)
        hidden = F.silu(up) * gate

        quotient = F.linear(hidden, self.W_down_q, self.b_down_q).squeeze(-1)
        remainder = x + F.linear(hidden, self.W_down_r, self.b_down_r).squeeze(-1)

        return quotient, remainder


def test_floor():
    """Test the floor implementations."""
    print("Testing floor(x) using SwiGLU formula:")
    print("-" * 50)

    test_values = [0.0, 0.5, 1.0, 1.5, 2.0, 2.9, 3.0, 10.0, 100.0, 255.0, 1000.0]

    for x in test_values:
        exact = int(x)
        approx = silu_floor(torch.tensor(x)).item()
        error = abs(approx - exact)
        status = "✓" if error < 0.5 else "✗"
        print(f"  {status} floor({x:6.1f}) = {approx:8.3f} (exact: {exact}, error: {error:.4f})")


def test_mod():
    """Test the mod implementations."""
    print("\nTesting x mod 256 using SwiGLU formula:")
    print("-" * 50)

    n = 256
    test_values = [0, 100, 255, 256, 257, 512, 1000, 65535, 260100]

    for x in test_values:
        exact = x % n
        approx = silu_mod(torch.tensor(float(x)), n).item()
        error = abs(approx - exact)
        status = "✓" if error < 1.0 else "✗"
        print(f"  {status} {x:6d} mod {n} = {approx:8.2f} (exact: {exact}, error: {error:.2f})")


def test_divmod():
    """Test divmod for carry extraction."""
    print("\nTesting divmod for carry extraction (x / 256):")
    print("-" * 50)

    n = 256
    test_values = [0, 100, 255, 256, 512, 1000, 65025, 130050, 260100]

    for x in test_values:
        exact_q = x // n
        exact_r = x % n
        approx_q, approx_r = silu_divmod(torch.tensor(float(x)), n)
        approx_q, approx_r = approx_q.item(), approx_r.item()
        error_q = abs(approx_q - exact_q)
        error_r = abs(approx_r - exact_r)
        status = "✓" if error_q < 1.0 and error_r < 1.0 else "✗"
        print(f"  {status} {x:6d} / {n} = {approx_q:8.2f} (exact: {exact_q:4d}), "
              f"mod = {approx_r:8.2f} (exact: {exact_r:3d})")


def count_params():
    """Count parameters for floor/mod operations."""
    print("\nParameter counts:")
    print("-" * 50)

    floor_ffn = FloorFFN()
    mod_ffn = ModFFN(n=256)
    divmod_ffn = DivModFFN(n=256)

    def count(m):
        return sum((p != 0).sum().item() for p in m.parameters())

    print(f"  FloorFFN: {count(floor_ffn)} non-zero params")
    print(f"  ModFFN: {count(mod_ffn)} non-zero params")
    print(f"  DivModFFN: {count(divmod_ffn)} non-zero params")


if __name__ == '__main__':
    print("="*70)
    print("SiLU Floor/Mod Implementation")
    print("="*70)
    test_floor()
    test_mod()
    test_divmod()
    count_params()
