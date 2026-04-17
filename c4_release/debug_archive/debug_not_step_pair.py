#!/usr/bin/env python3
"""Debug NOT step pair to understand the threshold issue."""

import torch
import torch.nn.functional as F

S = 100.0

# Test step(a >= 0.5) using step pair pattern
for a_val in [0.0, 0.5, 1.0, 2.0]:
    # Unit 2: step((a - 0.5) >= 0)
    up2 = S * a_val + 0.5 * S  # = S*a + 0.5*S
    hidden2 = F.silu(torch.tensor(up2)).item()

    # Unit 3: -step((a - 0.5) >= 1)
    up3 = S * a_val - 0.5 * S  # = S*a - 0.5*S
    hidden3 = F.silu(torch.tensor(up3)).item()

    # Step pair result
    step_result = (hidden2 - hidden3) / S

    print(f"a={a_val}:")
    print(f"  Unit 2: up={up2:.1f}, silu={hidden2:.2f}")
    print(f"  Unit 3: up={up3:.1f}, silu={hidden3:.2f}")
    print(f"  step(a >= 0.5) = {step_result:.4f} (expected: {1.0 if a_val >= 0.5 else 0.0})")
    print()

print("=" * 60)
print("Now test the full NOT operation: NOT(a) = 1 - step(a >= 0.5)")
print("=" * 60)

for a_val in [0.0, 0.5, 1.0, 2.0]:
    # Units 0-1: constant +1
    up0 = S
    hidden0 = F.silu(torch.tensor(up0)).item() * 1.0  # * gate
    contrib0 = hidden0 / S

    up1 = -S
    hidden1 = F.silu(torch.tensor(up1)).item() * (-1.0)  # * gate
    contrib1 = hidden1 / S

    const_part = contrib0 + contrib1

    # Units 2-3: step(a >= 0.5)
    up2 = S * a_val + 0.5 * S
    hidden2 = F.silu(torch.tensor(up2)).item()
    contrib2 = -(hidden2 / S)  # Negative W_down

    up3 = S * a_val - 0.5 * S
    hidden3 = F.silu(torch.tensor(up3)).item()
    contrib3 = hidden3 / S  # Positive W_down

    step_part = contrib2 + contrib3

    result = const_part + step_part
    expected = 0.0 if a_val >= 0.5 else 1.0

    print(f"\na={a_val}:")
    print(f"  Constant part: {const_part:.4f}")
    print(f"  Step part: {step_part:.4f}")
    print(f"  Total: {result:.4f} (expected: {expected})")
