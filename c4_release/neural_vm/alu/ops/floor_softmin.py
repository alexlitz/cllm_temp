"""
Neural floor using soft-min over candidates.

Instead of torch.round(), we:
1. Generate integer candidates: k = 0, 1, 2, ...
2. Compute differences: d_k = |x - (k + 0.5)| for floor
3. Use soft-min to find the closest candidate
4. The +0.5 offset converts rounding to flooring

For floor(x):
    - Candidate k is correct if k <= x < k+1
    - Equivalently: k + 0.5 is closest to x among {0.5, 1.5, 2.5, ...}
    - floor(x) = argmin_k |x - (k + 0.5)|

This is fully neural - no torch.round() needed!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


SCALE = 10000.0  # Very high scale for 1/256 precision
EPS = 0.1


def silu_floor_raw(x, scale=SCALE, eps=EPS):
    """Raw SiLU approximation: ~x for x >= 1, ~0.9 for x < 1."""
    return F.silu(scale * (x - 1 + eps)) / scale + 1 - eps


def staircase_floor(x, max_val=32, scale=SCALE):
    """
    Compute floor(x) using sum of step functions (staircase).

    floor(x) = Σ_{k=1}^{max} sigmoid(scale * (x - k + eps))

    Each step adds 1 when x crosses integer k.
    This correctly handles the discontinuity at integers.

    Using SiLU for steps: step(x >= k) ≈ sigmoid(scale * (x - k + eps))
    """
    # Create thresholds at each integer: 1, 2, 3, ..., max_val
    thresholds = torch.arange(1, max_val + 1, dtype=x.dtype, device=x.device)

    # eps must be small enough for 1/256 precision
    # But large enough that x = k gives step ≈ 1
    # With scale=10000, eps=0.001 gives sigmoid(10) ≈ 0.9999
    # And x = 0.996: sigmoid(10000 * (-0.003)) = sigmoid(-30) ≈ 0
    eps = 0.001

    if x.dim() == 0:
        x = x.unsqueeze(0)

    # For each threshold k, compute sigmoid(scale * (x - k + eps))
    # This is ~1 if x >= k, ~0 if x < k
    diffs = x.unsqueeze(-1) - thresholds + eps  # (batch, thresholds)
    steps = torch.sigmoid(scale * diffs)

    # Sum of steps = floor(x)
    floor_x = steps.sum(dim=-1)

    return floor_x.squeeze()


def softmin_floor(x, max_candidates=32, scale=SCALE):
    """Alias for staircase_floor for compatibility."""
    return staircase_floor(x, max_val=max_candidates, scale=scale)


def softmin_floor_v2(x, max_val, scale=SCALE):
    """
    Compute floor(x) for x in [0, max_val] using adaptive candidates.

    Generates only as many candidates as needed based on max_val.
    """
    num_candidates = int(max_val) + 2  # +2 for safety margin
    return softmin_floor(x, max_candidates=num_candidates, scale=scale)


def hierarchical_softmin_floor(x, scale=SCALE):
    """
    Compute floor(x) using hierarchical soft-min.

    For large x, use two-stage approach:
    1. Find which 16-wide bin x falls into (coarse)
    2. Find which integer within the bin (fine)

    This reduces O(max_val) to O(max_val/16 + 16) candidates.
    """
    # Stage 1: Find coarse bin (which multiple of 16)
    x_coarse = x / 16
    coarse_candidates = torch.arange(128, dtype=x.dtype, device=x.device)  # Up to 2048
    coarse_centers = coarse_candidates + 0.5

    if x_coarse.dim() == 0:
        x_coarse = x_coarse.unsqueeze(0)
    coarse_distances = torch.abs(x_coarse.unsqueeze(-1) - coarse_centers)
    coarse_weights = F.softmax(-scale * coarse_distances, dim=-1)
    coarse_floor = (coarse_weights * coarse_candidates).sum(dim=-1)

    # Stage 2: Find fine offset within bin (0-15)
    x_fine = x - coarse_floor.unsqueeze(-1) * 16 if x.dim() > 0 else x - coarse_floor * 16
    fine_candidates = torch.arange(16, dtype=x.dtype, device=x.device)
    fine_centers = fine_candidates + 0.5

    if x_fine.dim() == 0:
        x_fine = x_fine.unsqueeze(0)
    # Reshape for broadcasting
    x_fine_expanded = x_fine.view(-1, 1) if x_fine.dim() == 1 else x_fine.unsqueeze(-1)
    fine_distances = torch.abs(x_fine_expanded - fine_centers)
    fine_weights = F.softmax(-scale * fine_distances, dim=-1)
    fine_floor = (fine_weights * fine_candidates).sum(dim=-1)

    # Combine
    result = coarse_floor * 16 + fine_floor
    return result.squeeze()


def softmin_mod(x, n, scale=SCALE):
    """
    Compute x mod n using soft-min floor.

    x mod n = x - floor(x/n) * n
    """
    max_quotient = int(x.max().item() / n) + 2 if x.dim() > 0 else int(x.item() / n) + 2
    quotient = softmin_floor(x / n, max_candidates=max_quotient, scale=scale)
    return x - quotient * n


class SoftminFloorFFN(nn.Module):
    """
    FFN that computes floor(x) using soft-min over candidates.

    For floor in range [0, max_val], uses max_val+1 candidates.

    Parameters:
        - Candidate values stored as buffer (not learned)
        - Scale parameter for softmax sharpness
    """

    def __init__(self, max_val=256, scale=SCALE):
        super().__init__()
        self.scale = scale
        self.max_val = max_val

        # Store candidates as buffer (not parameters)
        candidates = torch.arange(max_val + 2, dtype=torch.float32)
        self.register_buffer('candidates', candidates)
        self.register_buffer('centers', candidates + 0.5)

    def forward(self, x):
        """Compute floor(x) using soft-min."""
        orig_shape = x.shape
        x_flat = x.view(-1, 1)

        # Distances to candidate centers
        distances = torch.abs(x_flat - self.centers)

        # Soft-min weights
        weights = F.softmax(-self.scale * distances, dim=-1)

        # Weighted sum
        floor_x = (weights * self.candidates).sum(dim=-1)

        return floor_x.view(orig_shape)


class SoftminDivModFFN(nn.Module):
    """
    FFN that computes floor(x/n) and x mod n using soft-min.
    """

    def __init__(self, n, max_quotient=256, scale=SCALE):
        super().__init__()
        self.n = n
        self.scale = scale

        candidates = torch.arange(max_quotient + 2, dtype=torch.float32)
        self.register_buffer('candidates', candidates)
        self.register_buffer('centers', candidates + 0.5)

    def forward(self, x):
        """Compute (floor(x/n), x mod n)."""
        orig_shape = x.shape
        x_scaled = (x / self.n).view(-1, 1)

        # Soft-min for quotient
        distances = torch.abs(x_scaled - self.centers)
        weights = F.softmax(-self.scale * distances, dim=-1)
        quotient = (weights * self.candidates).sum(dim=-1)

        # Remainder
        remainder = x.view(-1) - quotient * self.n

        return quotient.view(orig_shape), remainder.view(orig_shape)


def test_softmin_floor():
    """Test soft-min floor implementation."""
    print("Testing soft-min floor:")
    print("-" * 60)

    test_values = [0.0, 0.5, 0.9, 1.0, 1.5, 2.0, 2.9, 3.0, 10.5, 15.9, 31.0]

    print(f"{'x':>8} | {'softmin':>10} | {'expected':>8} | {'status':>6}")
    print("-" * 45)

    all_pass = True
    for x in test_values:
        t = torch.tensor(x)
        result = softmin_floor(t, max_candidates=64).item()
        expected = int(x)
        ok = abs(result - expected) < 0.5
        all_pass = all_pass and ok
        status = "✓" if ok else "✗"
        print(f"{x:>8.1f} | {result:>10.2f} | {expected:>8} | {status:>6}")

    print(f"\nAll floor tests passed: {all_pass}")
    return all_pass


def test_softmin_mod():
    """Test soft-min mod implementation."""
    print("\nTesting soft-min mod (x mod 256):")
    print("-" * 60)

    n = 256
    test_values = [0, 100, 255, 256, 257, 512, 1000]

    print(f"{'x':>8} | {'quotient':>10} | {'mod':>10} | {'expected':>8} | {'status':>6}")
    print("-" * 60)

    all_pass = True
    for x in test_values:
        t = torch.tensor(float(x))
        q = softmin_floor(t / n, max_candidates=16).item()
        mod = softmin_mod(t, n).item()
        expected = x % n
        ok = abs(mod - expected) < 1.0
        all_pass = all_pass and ok
        status = "✓" if ok else "✗"
        print(f"{x:>8} | {q:>10.2f} | {mod:>10.2f} | {expected:>8} | {status:>6}")

    print(f"\nAll mod tests passed: {all_pass}")
    return all_pass


def test_large_values():
    """Test with large values (for MUL carry extraction)."""
    print("\nTesting large values (carry extraction):")
    print("-" * 60)

    n = 256
    # Values from MUL: products can reach 260100
    test_values = [65025, 130050, 260100]

    print(f"{'x':>10} | {'q_expected':>10} | {'q_softmin':>10} | {'mod':>6} | {'status':>6}")
    print("-" * 60)

    all_pass = True
    for x in test_values:
        t = torch.tensor(float(x))
        expected_q = x // n
        expected_mod = x % n

        # Need enough candidates for large quotients
        q = hierarchical_softmin_floor(t / n).item()
        mod = x - int(round(q)) * n

        ok = abs(q - expected_q) < 0.5 and abs(mod - expected_mod) < 1
        all_pass = all_pass and ok
        status = "✓" if ok else "✗"
        print(f"{x:>10} | {expected_q:>10} | {q:>10.2f} | {int(mod):>6} | {status:>6}")

    print(f"\nAll large value tests passed: {all_pass}")
    return all_pass


def count_params():
    """Count parameters (mostly buffers, minimal learned params)."""
    print("\nParameter counts:")
    print("-" * 40)

    floor_ffn = SoftminFloorFFN(max_val=32)
    divmod_ffn = SoftminDivModFFN(n=256, max_quotient=1024)

    # Count actual parameters (not buffers)
    floor_params = sum(p.numel() for p in floor_ffn.parameters())
    divmod_params = sum(p.numel() for p in divmod_ffn.parameters())

    # Count buffer size
    floor_buffer = sum(b.numel() for b in floor_ffn.buffers())
    divmod_buffer = sum(b.numel() for b in divmod_ffn.buffers())

    print(f"  SoftminFloorFFN(max=32): {floor_params} params, {floor_buffer} buffer")
    print(f"  SoftminDivModFFN(max_q=1024): {divmod_params} params, {divmod_buffer} buffer")
    print()
    print("  Note: Candidates stored as buffers, not learned parameters")
    print("  The computation is fully neural (softmax over distances)")


if __name__ == '__main__':
    print("=" * 70)
    print("Soft-min Floor Implementation")
    print("(No torch.round() - fully neural!)")
    print("=" * 70)

    test_softmin_floor()
    test_softmin_mod()
    test_large_values()
    count_params()
