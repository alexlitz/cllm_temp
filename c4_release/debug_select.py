#!/usr/bin/env python3
"""Debug SELECT with fractional cond value."""

import torch
import torch.nn.functional as F

S = 100.0
cond_val = 0.2
a_val = 42.0
b_val = 99.0

print(f"Testing SELECT({cond_val}, {a_val}, {b_val})")
print(f"Expected: {b_val} (cond < 0.5, so select b)")
print()

# Units 0-1: step(cond >= 0.5) * a
# Unit 0: step(2*cond >= 1) * a
up0 = 2 * S * cond_val + 0  # = 2*100*0.2 = 40
gate0 = a_val
hidden0 = F.silu(torch.tensor(up0)).item() * gate0
contrib0 = hidden0 / S
print(f"Unit 0: step(2*{cond_val} >= 1) * {a_val}")
print(f"  up = {up0:.1f}, silu = {F.silu(torch.tensor(up0)).item():.2f}")
print(f"  hidden = {hidden0:.2f}, contrib = {contrib0:.4f}")

# Unit 1: -step(2*cond >= 2) * a
up1 = 2 * S * cond_val - S  # = 40 - 100 = -60
gate1 = -a_val
hidden1 = F.silu(torch.tensor(up1)).item() * gate1
contrib1 = hidden1 / S
print(f"Unit 1: -step(2*{cond_val} >= 2) * {a_val}")
print(f"  up = {up1:.1f}, silu = {F.silu(torch.tensor(up1)).item():.2f}")
print(f"  hidden = {hidden1:.2f}, contrib = {contrib1:.4f}")

a_contribution = contrib0 + contrib1
print(f"Total from a: {a_contribution:.4f} (should be 0)")
print()

# Units 2-3: step(cond < 0.5) * b = step(1 - 2*cond >= 0) * b
# Unit 2: step(-2*cond + 1 >= 0) * b
up2 = -2 * S * cond_val + S  # = -40 + 100 = 60
gate2 = b_val
hidden2 = F.silu(torch.tensor(up2)).item() * gate2
contrib2 = hidden2 / S
print(f"Unit 2: step(1 - 2*{cond_val} >= 0) * {b_val}")
print(f"  up = {up2:.1f}, silu = {F.silu(torch.tensor(up2)).item():.2f}")
print(f"  hidden = {hidden2:.2f}, contrib = {contrib2:.4f}")

# Unit 3: -step(1 - 2*cond >= 1) * b
up3 = -2 * S * cond_val + 0  # = -40
gate3 = -b_val
hidden3 = F.silu(torch.tensor(up3)).item() * gate3
contrib3 = hidden3 / S
print(f"Unit 3: -step(1 - 2*{cond_val} >= 1) * {b_val}")
print(f"  up = {up3:.1f}, silu = {F.silu(torch.tensor(up3)).item():.2f}")
print(f"  hidden = {hidden3:.2f}, contrib = {contrib3:.4f}")

b_contribution = contrib2 + contrib3
print(f"Total from b: {b_contribution:.4f} (should be {b_val})")
print()

total = a_contribution + b_contribution
print(f"Final result: {total:.4f}")
print(f"Expected: {b_val}")
print(f"Error: {abs(total - b_val):.4f}")
