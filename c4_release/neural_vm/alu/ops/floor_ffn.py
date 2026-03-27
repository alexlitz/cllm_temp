"""
Floor operation as pure FFN (no buffers, no torch.round).

Staircase floor using SwiGLU FFN:
    floor(x) = Σ sigmoid(scale * (x - k + eps)) for k = 1..max

Each threshold k becomes one hidden unit:
    W_up[k] = scale
    b_up[k] = scale * (-k + eps)
    W_down[k] = 1 (sum all units)

Total weights: ~3*M for M thresholds (W_up, b_up, W_down)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StaircaseFloorFFN(nn.Module):
    """
    FFN that computes floor(x) using staircase of sigmoid steps.

    For floor in range [0, max_val]:
        - M = max_val hidden units
        - Each unit k detects if x >= k
        - Sum of all units = floor(x)

    Weights:
        W_up: (M, 1) all = scale
        b_up: (M,) = scale * (-k + eps) for k = 1..M
        W_down: (1, M) all = 1
        b_down: (1,) = 0

    Total: 3*M + 1 weights
    """

    def __init__(self, max_val=32, scale=10000.0, eps=0.002):
        super().__init__()
        self.max_val = max_val
        self.scale = scale
        self.eps = eps

        M = max_val

        # W_up: all weights = scale (detects x - threshold)
        self.W_up = nn.Parameter(torch.full((M, 1), scale))

        # b_up: encodes thresholds 1, 2, ..., M
        # b_up[k] = scale * (-k - 1 + eps) so that:
        # W_up @ x + b_up[k] = scale * x + scale * (-k-1 + eps) = scale * (x - k - 1 + eps)
        thresholds = torch.arange(1, M + 1, dtype=torch.float32)
        self.b_up = nn.Parameter(scale * (-thresholds + eps))

        # W_down: sum all hidden units
        self.W_down = nn.Parameter(torch.ones(1, M))

        # b_down: no bias needed
        self.b_down = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """Compute floor(x)."""
        # x: any shape
        orig_shape = x.shape
        x_flat = x.view(-1, 1)  # (batch, 1)

        # hidden = sigmoid(W_up @ x + b_up)
        # Shape: (batch, M)
        pre_act = F.linear(x_flat, self.W_up, self.b_up)
        hidden = torch.sigmoid(pre_act)

        # output = W_down @ hidden + b_down = sum of hidden units
        output = F.linear(hidden, self.W_down, self.b_down)

        return output.view(orig_shape)


class StaircaseDivModFFN(nn.Module):
    """
    FFN that computes floor(x/n) and x mod n.

    Uses staircase for floor, then:
        mod = x - floor(x/n) * n
    """

    def __init__(self, n, max_quotient=32, scale=10000.0, eps=0.002):
        super().__init__()
        self.n = n
        self.scale = scale
        self.eps = eps

        M = max_quotient

        # Floor FFN weights (operating on x/n)
        # W_up includes the 1/n scaling
        self.W_up = nn.Parameter(torch.full((M, 1), scale / n))

        thresholds = torch.arange(1, M + 1, dtype=torch.float32)
        self.b_up = nn.Parameter(scale * (-thresholds + eps))

        self.W_down_floor = nn.Parameter(torch.ones(1, M))
        self.b_down_floor = nn.Parameter(torch.zeros(1))

        # For mod: need to compute x - floor * n
        # This requires a second stage or combined output

    def forward(self, x):
        """Compute (floor(x/n), x mod n)."""
        orig_shape = x.shape
        x_flat = x.view(-1, 1)

        # Floor(x/n) via staircase
        pre_act = F.linear(x_flat, self.W_up, self.b_up)
        hidden = torch.sigmoid(pre_act)
        floor_q = F.linear(hidden, self.W_down_floor, self.b_down_floor)

        # Mod = x - floor * n
        mod = x_flat - floor_q * self.n

        return floor_q.view(orig_shape), mod.view(orig_shape)


def count_floor_params(max_val):
    """Count parameters for floor FFN."""
    ffn = StaircaseFloorFFN(max_val=max_val)
    total = sum(p.numel() for p in ffn.parameters())
    nonzero = sum((p != 0).sum().item() for p in ffn.parameters())
    return total, nonzero


def test_floor_ffn():
    """Test staircase floor FFN."""
    print("Testing StaircaseFloorFFN:")
    print("-" * 60)

    ffn = StaircaseFloorFFN(max_val=64, scale=10000.0, eps=0.002)

    test_values = [0.0, 0.5, 0.999, 1.0, 1.5, 2.0, 2.9, 3.0, 10.5, 31.0, 63.9]

    print(f"{'x':>8} | {'floor_ffn':>10} | {'expected':>8} | {'status':>6}")
    print("-" * 45)

    all_pass = True
    with torch.no_grad():
        for x in test_values:
            t = torch.tensor(x)
            result = ffn(t).item()
            expected = int(x)
            ok = abs(result - expected) < 0.5
            all_pass = all_pass and ok
            status = "✓" if ok else "✗"
            print(f"{x:>8.3f} | {result:>10.2f} | {expected:>8} | {status:>6}")

    print(f"\nAll tests passed: {all_pass}")
    return all_pass


def test_divmod_ffn():
    """Test divmod FFN."""
    print("\nTesting StaircaseDivModFFN (x / 256):")
    print("-" * 60)

    ffn = StaircaseDivModFFN(n=256, max_quotient=1024, scale=10000.0, eps=0.002)

    test_values = [0, 100, 255, 256, 257, 512, 1000, 65025, 260100]

    print(f"{'x':>8} | {'quotient':>10} | {'mod':>10} | {'exp_q':>6} | {'exp_m':>6} | {'ok':>4}")
    print("-" * 65)

    all_pass = True
    with torch.no_grad():
        for x in test_values:
            t = torch.tensor(float(x))
            q, m = ffn(t)
            q, m = q.item(), m.item()
            exp_q = x // 256
            exp_m = x % 256
            ok = abs(q - exp_q) < 0.5 and abs(m - exp_m) < 1.0
            all_pass = all_pass and ok
            status = "✓" if ok else "✗"
            print(f"{x:>8} | {q:>10.2f} | {m:>10.2f} | {exp_q:>6} | {exp_m:>6} | {status:>4}")

    print(f"\nAll tests passed: {all_pass}")
    return all_pass


def show_param_counts():
    """Show parameter counts for various max values."""
    print("\nParameter counts:")
    print("-" * 40)

    for max_val in [16, 32, 64, 256, 1024]:
        total, nonzero = count_floor_params(max_val)
        print(f"  max_val={max_val:4d}: {total:,} total, {nonzero:,} non-zero")

    print()
    print("For MUL carry extraction (max_quotient=1024):")
    ffn = StaircaseDivModFFN(n=256, max_quotient=1024)
    total = sum(p.numel() for p in ffn.parameters())
    print(f"  StaircaseDivModFFN: {total:,} params")
    print(f"  vs Original lookup: ~41,000 params")
    print(f"  Reduction: {100*(1 - total/41000):.1f}%")


if __name__ == '__main__':
    print("=" * 70)
    print("Staircase Floor as Pure FFN")
    print("=" * 70)

    test_floor_ffn()
    test_divmod_ffn()
    show_param_counts()
