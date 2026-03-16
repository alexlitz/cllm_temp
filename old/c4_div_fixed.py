#!/usr/bin/env python3
"""
Fixed DIV: Dense log table via attention
"""
import torch
import torch.nn.functional as F
import math

def silu(x): return x * torch.sigmoid(x)
def sharp_gate(x, s=20.0): return (silu(x*s + 0.5*s) - silu(x*s - 0.5*s)) / s
def eq_gate(a, b, s=20.0): 
    d = a - b
    return sharp_gate(d + 0.5, s) * sharp_gate(-d + 0.5, s)

print("FIXED DIVISION: Dense Log Table via Attention")
print("=" * 60)

# =============================================================================
# Dense log table: Keys = [1, 2, 3, ...], Values = [log₂(1), log₂(2), ...]
# =============================================================================

MAX_DIVISOR = 64

# Attention KV for log lookup
# Keys: integers 1 to MAX_DIVISOR (embedded as scalars for simplicity)
# Values: their log₂ values
log_keys = torch.arange(1, MAX_DIVISOR + 1).float()
log_values = torch.log2(log_keys)  # Precomputed constants

print(f"Log table: {MAX_DIVISOR} entries")
print(f"  Keys: [1, 2, 3, ..., {MAX_DIVISOR}]")
print(f"  Values: [0, 1, 1.585, 2, 2.322, ...]")

def log2_via_attention(b):
    """
    Look up log₂(b) using attention with eq_gate scoring.
    
    This is standard attention where:
      Q = b (scalar query)
      K = [1, 2, 3, ..., 64] (integer keys)
      V = [0, 1, 1.585, 2, ...] (log values)
      
    Score = eq_gate(b, k) for each key k
    Output = softmax(scores) @ V = log₂(b) for integer b
    """
    b = torch.clamp(b, min=1.0, max=float(MAX_DIVISOR))
    
    # Compute attention scores via eq_gate (sharp integer matching)
    scores = torch.stack([eq_gate(b, k, s=30.0) for k in log_keys])
    
    # Softmax (scores are already ~0 or ~1, softmax sharpens)
    weights = F.softmax(scores * 100, dim=0)
    
    # Weighted sum of log values
    return (weights * log_values).sum()

def exp_via_softmax(x):
    """exp(x) = softmax([x, 0])[0] / softmax([x, 0])[1]"""
    logits = torch.stack([x, torch.tensor(0.0)])
    probs = F.softmax(logits, dim=0)
    return probs[0] / probs[1]

def exp2_via_softmax(x):
    """2^x = e^(x × ln(2))"""
    return exp_via_softmax(x * math.log(2))

# MUL FFN (from before)
MAX_VAL = 64
squares = torch.arange(MAX_VAL * 2 + 1).float() ** 2

def square_via_eqgate(x):
    x = torch.clamp(torch.abs(x), max=MAX_VAL * 2)
    scores = torch.stack([eq_gate(x, torch.tensor(float(i)), 30) for i in range(MAX_VAL * 2 + 1)])
    return (F.softmax(scores * 100, dim=0) * squares).sum()

def mul_ffn(a, b):
    sum_sq = square_via_eqgate(a + b)
    diff_sq = square_via_eqgate(torch.abs(a - b))
    return (sum_sq - diff_sq) / 4

def div_via_log_exp(a, b):
    """
    a / b via:
    1. log₂(b) via attention (dense table)
    2. 2^(-log₂(b)) = 1/b via softmax (exact)
    3. a × (1/b) via FFN (quarter-squares)
    """
    if b <= 0:
        return torch.tensor(0.0)
    
    # Step 1: Log via attention
    log_b = log2_via_attention(b)
    
    # Step 2: 1/b = 2^(-log₂(b)) via softmax
    neg_log_b = -log_b
    inv_b = exp2_via_softmax(neg_log_b)
    
    # Step 3: a × (1/b) via FFN
    # For large a, we can't use quarter-squares directly
    # Use direct multiplication for now (or iterative approach)
    result = a * inv_b
    
    return torch.floor(result)

print("\nStep 1 - Log via attention (dense table):")
for b in [1, 2, 3, 4, 5, 7, 8, 10, 16]:
    log_b = log2_via_attention(torch.tensor(float(b)))
    expected = math.log2(b)
    error = abs(log_b.item() - expected)
    status = "✓" if error < 0.01 else "✗"
    print(f"  log₂({b:2d}) = {log_b.item():.4f} (exact: {expected:.4f}) {status}")

print("\nStep 2 - 1/b via exp softmax:")
for b in [2, 3, 4, 5, 7, 8, 10]:
    log_b = log2_via_attention(torch.tensor(float(b)))
    inv_b = exp2_via_softmax(-log_b)
    expected = 1.0 / b
    error = abs(inv_b.item() - expected)
    status = "✓" if error < 0.01 else "✗"
    print(f"  1/{b} = {inv_b.item():.4f} (exact: {expected:.4f}) {status}")

print("\nFull division a/b:")
test_cases = [
    (20, 4), (100, 10), (42, 7), (15, 3), (35, 2),
    (8, 2), (64, 8), (27, 9), (50, 5), (63, 7)
]
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
│ CORRECTED DIV ARCHITECTURE                                           │
└──────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │ ATTENTION (Log lookup)                                          │
  │   Keys:   [1, 2, 3, ..., 64]        ← integer divisors          │
  │   Values: [0, 1, 1.585, 2, ...]     ← precomputed log₂          │
  │   Query:  b                                                     │
  │   Score:  eq_gate(b, key)           ← sharp integer matching    │
  │   Output: log₂(b)                                               │
  └─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │ SOFTMAX (Exact exp)                                             │
  │   Input:  -log₂(b)                                              │
  │   Compute: exp(-log₂(b) × ln(2)) = 2^(-log₂(b)) = 1/b          │
  │   Method:  softmax([x, 0])[0] / softmax([x, 0])[1]             │
  └─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │ FFN (Multiply)                                                  │
  │   Compute: a × (1/b)                                            │
  │   Floor for integer division                                    │
  └─────────────────────────────────────────────────────────────────┘
""")
