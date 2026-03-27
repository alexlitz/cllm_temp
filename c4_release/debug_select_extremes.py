#!/usr/bin/env python3
"""Debug SELECT at extreme values (0 and 1)."""

import torch
import torch.nn.functional as F

S = 100.0

for cond_val, expected_source in [(0.0, 'b'), (1.0, 'a')]:
    a_val = 42.0
    b_val = 99.0

    print(f"\nTesting cond={cond_val} (should select {expected_source}):")

    # Units 0-1: step(cond >= 0.5) * a
    up0 = 2 * S * cond_val + 0
    hidden0 = F.silu(torch.tensor(up0)).item() * a_val
    contrib0 = hidden0 / S

    up1 = 2 * S * cond_val - S
    hidden1 = F.silu(torch.tensor(up1)).item() * (-a_val)
    contrib1 = hidden1 / S

    a_contribution = contrib0 + contrib1
    print(f"  a contribution: {a_contribution:.4f}")

    # Units 2-3: step(cond < 0.5) * b
    up2 = -2 * S * cond_val + S
    hidden2 = F.silu(torch.tensor(up2)).item() * b_val
    contrib2 = hidden2 / S

    up3 = -2 * S * cond_val + 0
    hidden3 = F.silu(torch.tensor(up3)).item() * (-b_val)
    contrib3 = hidden3 / S

    b_contribution = contrib2 + contrib3
    print(f"  b contribution: {b_contribution:.4f}")

    total = a_contribution + b_contribution
    expected = a_val if expected_source == 'a' else b_val
    print(f"  Total: {total:.4f} (expected {expected})")
