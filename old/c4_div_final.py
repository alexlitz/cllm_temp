#!/usr/bin/env python3
"""
Final DIV: handles floating point precision properly
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

def div_via_log_exp(a, b):
    """
    a / b using log-exp method.
    
    Note: For integer division, we use round() not floor()
    because the log-exp method gives very precise results
    that may be epsilon below the true quotient.
    """
    if b <= 0:
        return torch.tensor(0.0)
    
    # Step 1: Log via attention
    log_b = log2_via_attention(b)
    
    # Step 2: 1/b = 2^(-log₂(b)) via softmax (exact for integer b!)
    inv_b = exp2_via_softmax(-log_b)
    
    # Step 3: a × (1/b)
    result = a * inv_b
    
    # For integer division: round to nearest integer
    # This handles floating point precision issues
    return torch.round(result)

print("FINAL DIVISION TEST")
print("=" * 60)

print("\nDiagnostics - checking precision:")
for a, b in [(42, 7), (100, 10), (15, 3)]:
    log_b = log2_via_attention(torch.tensor(float(b)))
    inv_b = exp2_via_softmax(-log_b)
    raw = a * inv_b.item()
    print(f"  {a}/{b}: log₂({b})={log_b.item():.6f}, 1/{b}={inv_b.item():.6f}, raw={raw:.6f}")

print("\nFull division with round():")
test_cases = [
    (20, 4), (100, 10), (42, 7), (15, 3), (35, 2),
    (8, 2), (64, 8), (27, 9), (50, 5), (63, 7),
    (99, 9), (48, 6), (56, 8), (81, 9), (72, 8)
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

# =============================================================================
# Alternative: Direct reciprocal table (more accurate for integer div)
# =============================================================================
print("\n" + "=" * 60)
print("ALTERNATIVE: If log-exp has precision issues, use direct 1/b table")
print("=" * 60)

# Direct reciprocal table via attention
recip_keys = torch.arange(1, MAX_DIVISOR + 1).float()
recip_values = 1.0 / recip_keys

def reciprocal_via_attention(b):
    """Direct 1/b lookup - no log/exp needed."""
    b = torch.clamp(b, min=1.0, max=float(MAX_DIVISOR))
    scores = torch.stack([eq_gate(b, k, s=30.0) for k in recip_keys])
    weights = F.softmax(scores * 100, dim=0)
    return (weights * recip_values).sum()

def div_direct(a, b):
    if b <= 0:
        return torch.tensor(0.0)
    inv_b = reciprocal_via_attention(b)
    return torch.round(a * inv_b)

print("\nDirect reciprocal method:")
all_pass = True
for a, b in test_cases:
    result = div_direct(torch.tensor(float(a)), torch.tensor(float(b)))
    expected = a // b
    ok = abs(result.item() - expected) < 0.5
    all_pass = all_pass and ok
    status = "✓" if ok else "✗"
    print(f"  {a:3d} / {b:2d} = {result.item():.0f} (expected {expected}) {status}")

print(f"\n{'ALL PASS!' if all_pass else 'Some failures'}")

print("""
┌──────────────────────────────────────────────────────────────────────┐
│ SUMMARY: Two Division Methods                                        │
└──────────────────────────────────────────────────────────────────────┘

  METHOD 1: Log-Exp (your preferred approach)
  ───────────────────────────────────────────
    1. Attention: log₂(b) from table [1..64] → [0, 1, 1.585, ...]
    2. Softmax:   2^(-log₂(b)) = 1/b  (exact via ratio trick)
    3. Multiply:  a × (1/b), round to integer
    
    Pros: Uses the elegant exp-via-softmax trick
    Cons: Slight precision loss in log lookup for non-powers-of-2

  METHOD 2: Direct Reciprocal
  ───────────────────────────
    1. Attention: 1/b from table [1..64] → [1, 0.5, 0.333, ...]
    2. Multiply:  a × (1/b), round to integer
    
    Pros: More accurate for integer division
    Cons: Doesn't demonstrate the exp-via-softmax trick

  Both use attention for table lookup + FFN for multiply.
""")
