#!/usr/bin/env python3
"""
C4 Transformer - Fixed version with correct SUB and DIV
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def silu(x):
    return x * torch.sigmoid(x)

def sharp_gate(x, scale=20.0):
    return (silu(x * scale + 0.5 * scale) - silu(x * scale - 0.5 * scale)) / scale

def eq_gate(a, b, scale=20.0):
    diff = a - b
    return sharp_gate(diff + 0.5, scale) * sharp_gate(-diff + 0.5, scale)

# =============================================================================
# FIXED SUB: Use selection instead of linear combination
# =============================================================================
print("FIXED ARITHMETIC")
print("=" * 60)

print("""
Problem with SUB: silu(negative) ≈ 0

Solution: Use eq_gate to SELECT values, not combine them.
For SUB, we compute arg1 - ax directly by:
1. Scaling inputs to linear region of SiLU
2. Or using gated selection

Actually, the cleanest fix: compute ADD of (arg1) and (-ax)
We can negate ax by having W1 extract -ax in a controlled way.
""")

# For SUB: ax = arg1 - old_ax
# We want ffn to output: arg1 - 2*old_ax (then residual adds old_ax back)
# 
# Trick: Use the POSITIVE silu region
# silu(x + large_bias) ≈ x + large_bias for large bias
# So: silu(arg1 + B) - silu(2*ax + B) ≈ arg1 - 2*ax

class SubFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = 100.0  # Large positive bias to stay in linear region
        
    def forward(self, state):
        ax = state[0]
        arg1 = state[1]
        
        # Compute in positive region of SiLU
        h1 = F.silu(arg1 + self.bias) - self.bias  # ≈ arg1
        h2 = F.silu(2 * ax + self.bias) - self.bias  # ≈ 2*ax
        
        delta = h1 - h2  # ≈ arg1 - 2*ax
        
        result = torch.zeros_like(state)
        result[0] = delta  # Will be added via residual
        return result

print("Testing fixed SUB:")
AX, ARG1 = 0, 1
sub_ffn = SubFFN()
state = torch.zeros(4)
state[AX] = 3.0
state[ARG1] = 10.0
result = state + sub_ffn(state)
print(f"  ax=3, arg1=10 → ax = {result[AX].item():.1f} (expected 7)")

state[AX] = 15.0
state[ARG1] = 20.0
result = state + sub_ffn(state)
print(f"  ax=15, arg1=20 → ax = {result[AX].item():.1f} (expected 5)")

# =============================================================================
# FIXED DIVISION: Better log approximation
# =============================================================================
print("\n" + "=" * 60)
print("FIXED DIVISION")
print("=" * 60)

print("""
Problem: Log lookup with sparse table (powers of 2) is inaccurate.

Solution: Dense interpolation table OR direct reciprocal lookup.

Actually, simplest: lookup 1/b directly, then multiply!
  - Keys: [1, 2, 3, 4, ..., N]
  - Values: [1, 0.5, 0.333, 0.25, ..., 1/N]
  - Query b → attention returns 1/b
  - Then compute a * (1/b)
""")

MAX_VAL = 128
# Reciprocal table
recip_keys = torch.arange(1, MAX_VAL + 1).float()
recip_values = 1.0 / recip_keys

def reciprocal_lookup(b):
    """Look up 1/b using attention."""
    b = torch.clamp(b, min=1.0, max=MAX_VAL)
    # Sharp matching via eq_gate
    scores = torch.zeros(MAX_VAL)
    for i, k in enumerate(recip_keys):
        scores[i] = eq_gate(b, k, scale=30.0)
    weights = scores / (scores.sum() + 1e-10)
    return (weights * recip_values).sum()

def divide_fixed(a, b):
    """a / b via reciprocal lookup."""
    inv_b = reciprocal_lookup(b)
    return torch.floor(a * inv_b)

print("Testing fixed division:")
for a, b in [(20, 4), (100, 10), (42, 7), (15, 3), (35, 2)]:
    result = divide_fixed(torch.tensor(float(a)), torch.tensor(float(b)))
    expected = a // b
    status = "✓" if abs(result.item() - expected) < 0.5 else "✗"
    print(f"  {a} / {b} = {result.item():.0f} (expected {expected}) {status}")

# =============================================================================
# ALTERNATIVE: Exact integer division via repeated subtraction
# =============================================================================
print("\n" + "=" * 60)
print("ALTERNATIVE: Exact Division via Comparison Gates")
print("=" * 60)

print("""
For exact integer division, use the fact that:
  a / b = max{q : q*b <= a}

We can compute this with a table lookup:
  For each possible quotient q, check if q*b <= a
  Use sharp gates to select the largest valid q
""")

def divide_exact(a, b, max_quotient=64):
    """Exact integer division using comparison gates."""
    if b == 0:
        return torch.tensor(0.0)
    
    # Check each possible quotient
    valid_gates = torch.zeros(max_quotient + 1)
    for q in range(max_quotient + 1):
        # q is valid if q*b <= a, i.e., a - q*b >= 0
        diff = a - q * b
        # Gate: 1 if diff >= 0
        valid_gates[q] = torch.sigmoid(diff * 10.0)  # Smooth step at 0
    
    # Find largest valid q
    # Weight by q value, normalized by validity
    q_values = torch.arange(max_quotient + 1).float()
    
    # Trick: use the LAST valid q by weighting higher q more
    weights = valid_gates * torch.exp(q_values * 0.1)
    weights = weights / (weights.sum() + 1e-10)
    
    result = (weights * q_values).sum()
    return torch.floor(result)

print("Testing exact division:")
for a, b in [(20, 4), (100, 10), (42, 7), (15, 3), (35, 2), (99, 9)]:
    result = divide_exact(torch.tensor(float(a)), torch.tensor(float(b)))
    expected = a // b
    status = "✓" if abs(result.item() - expected) < 0.5 else "✗"
    print(f"  {a} / {b} = {result.item():.0f} (expected {expected}) {status}")

# =============================================================================
# FULL EXECUTION TEST
# =============================================================================
print("\n" + "=" * 60)
print("FULL EXECUTION: (3 + 4) * 5 / 2")
print("=" * 60)

# Quarter-squares multiplication (from before)
MAX_SQ = 128
squares_table = torch.arange(MAX_SQ + 1).float() ** 2

def square_lookup(x):
    x = torch.clamp(torch.abs(x), max=MAX_SQ)
    scores = torch.zeros(MAX_SQ + 1)
    for i in range(MAX_SQ + 1):
        scores[i] = eq_gate(x, torch.tensor(float(i)), scale=30.0)
    weights = scores / (scores.sum() + 1e-10)
    return (weights * squares_table).sum()

def multiply(a, b):
    sum_sq = square_lookup(a + b)
    diff_sq = square_lookup(torch.abs(a - b))
    return (sum_sq - diff_sq) / 4

ax = torch.tensor(3.0)
print(f"  IMM 3      → ax = {ax.item():.0f}")

ax = ax + 4.0  # ADD
print(f"  ADD 4      → ax = {ax.item():.0f}")

ax = multiply(torch.tensor(5.0), ax)
print(f"  MUL 5      → ax = {ax.item():.0f}")

ax = divide_fixed(ax, torch.tensor(2.0))
print(f"  DIV 2      → ax = {ax.item():.0f}")

print(f"\n  Final: {ax.item():.0f} (expected 17)")
