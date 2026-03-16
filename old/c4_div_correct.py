#!/usr/bin/env python3
"""
Correct integer division: floor(a/b), handling precision
"""
import torch
import torch.nn.functional as F
import math

def silu(x): return x * torch.sigmoid(x)
def sharp_gate(x, s=20.0): return (silu(x*s + 0.5*s) - silu(x*s - 0.5*s)) / s
def eq_gate(a, b, s=20.0): 
    d = a - b
    return sharp_gate(d + 0.5, s) * sharp_gate(-d + 0.5, s)

MAX_DIVISOR = 64
log_keys = torch.arange(1, MAX_DIVISOR + 1).float()
log_values = torch.log2(log_keys)

def log2_via_attention(b):
    b = torch.clamp(b, min=1.0, max=float(MAX_DIVISOR))
    scores = torch.stack([eq_gate(b, k, s=30.0) for k in log_keys])
    weights = F.softmax(scores * 100, dim=0)
    return (weights * log_values).sum()

def exp_via_softmax(x):
    logits = torch.stack([x, torch.tensor(0.0)])
    probs = F.softmax(logits, dim=0)
    return probs[0] / probs[1]

def exp2_via_softmax(x):
    return exp_via_softmax(x * math.log(2))

def integer_floor(x):
    """
    Floor with precision handling.
    
    Problem: 5.9999999 should floor to 6, not 5
    Solution: round(x - 0.4999) = floor(x) for x > 0.5
    
    Or simpler: if x is within epsilon of an integer, snap to it first
    """
    # Snap to nearest integer if within epsilon
    rounded = torch.round(x)
    if torch.abs(x - rounded) < 0.001:
        x = rounded
    return torch.floor(x)

def div_via_log_exp(a, b):
    """Integer division a // b using log-exp method."""
    if b <= 0:
        return torch.tensor(0.0)
    
    log_b = log2_via_attention(b)
    inv_b = exp2_via_softmax(-log_b)
    result = a * inv_b
    
    return integer_floor(result)

print("CORRECT INTEGER DIVISION (floor, not round)")
print("=" * 60)

test_cases = [
    (20, 4), (100, 10), (42, 7), (15, 3), (35, 2),
    (8, 2), (64, 8), (27, 9), (50, 5), (63, 7),
    (99, 9), (48, 6), (56, 8), (81, 9), (72, 8),
    (17, 5), (23, 7), (100, 3), (7, 2), (9, 4)
]

print("\nLog-Exp division with proper floor:")
all_pass = True
for a, b in test_cases:
    result = div_via_log_exp(torch.tensor(float(a)), torch.tensor(float(b)))
    expected = a // b
    ok = abs(result.item() - expected) < 0.5
    all_pass = all_pass and ok
    status = "✓" if ok else "✗"
    print(f"  {a:3d} / {b:2d} = {result.item():.0f} (expected {expected}) {status}")

print(f"\n{'ALL PASS!' if all_pass else 'Some failures'}")

print("""
┌──────────────────────────────────────────────────────────────────────┐
│ FINAL ARCHITECTURE                                                   │
└──────────────────────────────────────────────────────────────────────┘

  MUL via FFN:
  ────────────
    a × b = [(a+b)² - (a-b)²] / 4
    
    FFN structure:
      W1: extracts a, b, computes a+b and a-b
      SwiGLU: eq_gate lookups compute squares
      W2: combines (sum² - diff²) / 4

  DIV via Attention + Softmax:
  ────────────────────────────
    a / b = floor(a × 2^(-log₂(b)))
    
    1. ATTENTION: log₂(b)
       Keys:   [1, 2, 3, ..., 64]
       Values: [0, 1, 1.585, 2, ...]
       
    2. SOFTMAX: 2^(-log₂(b)) = 1/b
       exp(x) = softmax([x,0])[0] / softmax([x,0])[1]
       
    3. FFN: a × (1/b), then floor
""")
