#!/usr/bin/env python3
"""
Corrected MUL and DIV operations:
- MUL: FFN-based using quarter-squares with eq_gate lookups
- DIV: log via attention, exp via softmax, then multiply
"""
import torch
import torch.nn.functional as F
import math

def silu(x): return x * torch.sigmoid(x)
def sharp_gate(x, s=20.0): return (silu(x*s + 0.5*s) - silu(x*s - 0.5*s)) / s
def eq_gate(a, b, s=20.0): 
    d = a - b
    return sharp_gate(d + 0.5, s) * sharp_gate(-d + 0.5, s)

print("=" * 70)
print("CORRECTED OPERATIONS")
print("=" * 70)

# =============================================================================
# MULTIPLICATION - FFN with eq_gate square lookups
# =============================================================================
print("""
┌──────────────────────────────────────────────────────────────────────┐
│ MULTIPLICATION via FFN                                               │
└──────────────────────────────────────────────────────────────────────┘

  Identity: a × b = [(a+b)² - (a-b)²] / 4

  Implementation in FFN:
    1. Compute sum = a + b, diff = |a - b|  (linear in W1)
    2. Look up squares via eq_gate         (SwiGLU activation)
    3. Combine: (sum² - diff²) / 4         (linear in W2)

  The eq_gate IS the SwiGLU pattern:
    eq_gate(x, k) uses silu internally
    
  So squares lookup is:
    x² = Σᵢ i² × eq_gate(x, i)
    
  This is computed by FFN weights:
    W1 encodes the eq_gate comparisons
    W2 encodes the square values and combination
""")

MAX_VAL = 32

class MulFFN:
    """Multiplication via FFN with embedded square table."""
    
    def __init__(self, max_val=MAX_VAL):
        self.max_val = max_val
        # Square table embedded in "weights"
        self.squares = torch.arange(max_val * 2 + 1).float() ** 2
    
    def square_via_eqgate(self, x):
        """Compute x² using eq_gate (this IS the FFN activation pattern)."""
        x = torch.clamp(torch.abs(x), max=self.max_val * 2)
        # This sum of eq_gates IS what the FFN computes with SwiGLU
        result = torch.tensor(0.0)
        for i in range(self.max_val * 2 + 1):
            gate = eq_gate(x, torch.tensor(float(i)), s=30.0)
            result = result + gate * self.squares[i]
        return result
    
    def forward(self, a, b):
        """a × b via quarter-squares, all in FFN."""
        # Step 1: Linear combinations (W1 layer)
        sum_val = a + b
        diff_val = torch.abs(a - b)
        
        # Step 2: Square lookups via eq_gate (SwiGLU activation)
        sum_sq = self.square_via_eqgate(sum_val)
        diff_sq = self.square_via_eqgate(diff_val)
        
        # Step 3: Combine (W2 layer)
        return (sum_sq - diff_sq) / 4

mul_ffn = MulFFN()

print("  Testing MUL via FFN:")
for a, b in [(3, 4), (7, 8), (5, 5), (10, 3), (0, 5)]:
    result = mul_ffn.forward(torch.tensor(float(a)), torch.tensor(float(b)))
    expected = a * b
    status = "✓" if abs(result.item() - expected) < 0.5 else "✗"
    print(f"    {a} × {b} = {result.item():.0f} (expected {expected}) {status}")

# =============================================================================
# DIVISION - Log via attention, exp via softmax
# =============================================================================
print("""
┌──────────────────────────────────────────────────────────────────────┐
│ DIVISION via Log Attention + Exp Softmax                            │
└──────────────────────────────────────────────────────────────────────┘

  Formula: a / b = a × (1/b) = a × 2^(-log₂(b))

  Step 1: LOG via ATTENTION
    Keys:   [1, 2, 4, 8, 16, 32, ...]  (powers of 2)
    Values: [0, 1, 2, 3,  4,  5, ...]  (log₂ values)
    Query b matches nearest power → returns log₂(b)

  Step 2: EXP via SOFTMAX (exact!)
    2^(-log₂(b)) = exp(-log₂(b) × ln(2))
    
    exp(x) = softmax([x, 0])[0] / softmax([x, 0])[1]
    
  Step 3: MULTIPLY (FFN)
    a × (1/b) using quarter-squares FFN

  This is cleaner because:
    - Log table is small (only powers of 2)
    - Exp is EXACT via softmax
    - Reuses MUL FFN for final step
""")

# Log table: powers of 2 up to 2^10
LOG_BITS = 10
log_keys = torch.tensor([2.0 ** i for i in range(LOG_BITS + 1)])  # [1, 2, 4, 8, ...]
log_values = torch.arange(LOG_BITS + 1).float()  # [0, 1, 2, 3, ...]

def log2_via_attention(b):
    """
    Compute log₂(b) via attention over powers of 2.
    Uses soft matching for interpolation.
    """
    b = torch.clamp(b, min=1.0)
    
    # Query matches keys via distance-based scoring
    # For exact powers of 2, this is exact
    # For others, it interpolates (which is correct for log!)
    
    # Score: negative distance in log space
    # log₂(b) - log₂(key) = log₂(b/key)
    log_b = torch.log2(b)  # We'll replace this with attention-based version
    scores = -torch.abs(log_values - log_b) * 5.0  # Sharp matching
    
    weights = F.softmax(scores, dim=0)
    return (weights * log_values).sum()

def log2_via_attention_pure(b):
    """
    Pure attention version without using torch.log2.
    Uses ratio-based scoring.
    """
    b = torch.clamp(b, min=1.0, max=log_keys[-1])
    
    # For each power of 2, compute how close b is
    # Using the fact that b/2^i should be close to 1 for the right i
    scores = torch.zeros(LOG_BITS + 1)
    for i, key in enumerate(log_keys):
        # Ratio b/key: should be ~1 for matching power
        ratio = b / key
        # Score peaks when ratio = 1
        scores[i] = -torch.abs(ratio - 1.0) * 10.0
    
    weights = F.softmax(scores, dim=0)
    return (weights * log_values).sum()

def exp_via_softmax(x):
    """
    Compute e^x using softmax ratio trick.
    exp(x) = softmax([x, 0])[0] / softmax([x, 0])[1]
    """
    logits = torch.stack([x, torch.tensor(0.0)])
    probs = F.softmax(logits, dim=0)
    return probs[0] / probs[1]

def exp2_via_softmax(x):
    """Compute 2^x = e^(x × ln(2))"""
    return exp_via_softmax(x * math.log(2))

def div_via_log_exp(a, b):
    """
    a / b via:
    1. log₂(b) via attention
    2. 2^(-log₂(b)) via softmax
    3. a × result via FFN
    """
    if b <= 0:
        return torch.tensor(0.0)
    
    # Step 1: Log via attention
    log_b = log2_via_attention_pure(b)
    
    # Step 2: Exp via softmax (gives 1/b)
    inv_b = exp2_via_softmax(-log_b)
    
    # Step 3: Multiply via FFN
    result = mul_ffn.forward(a, inv_b)
    
    return torch.floor(result)

print("  Step 1 - Log via attention:")
for b in [1, 2, 4, 8, 16, 3, 5, 10]:
    log_b = log2_via_attention_pure(torch.tensor(float(b)))
    expected = math.log2(b)
    print(f"    log₂({b:2d}) = {log_b.item():.3f} (exact: {expected:.3f})")

print("\n  Step 2 - Exp via softmax (2^x):")
for x in [0, 1, 2, 3, -1, -2]:
    result = exp2_via_softmax(torch.tensor(float(x)))
    expected = 2.0 ** x
    print(f"    2^{x:2d} = {result.item():.4f} (exact: {expected:.4f})")

print("\n  Full division a/b:")
for a, b in [(20, 4), (100, 10), (42, 7), (15, 3), (35, 2), (8, 2), (64, 8)]:
    result = div_via_log_exp(torch.tensor(float(a)), torch.tensor(float(b)))
    expected = a // b
    status = "✓" if abs(result.item() - expected) < 1 else "✗"
    print(f"    {a:3d} / {b:2d} = {result.item():.0f} (expected {expected}) {status}")

# =============================================================================
# SUMMARY
# =============================================================================
print("""
┌──────────────────────────────────────────────────────────────────────┐
│ CORRECTED SUMMARY                                                    │
└──────────────────────────────────────────────────────────────────────┘

  MUL (FFN-based):
    ┌─────────────────────────────────────────────────────────────────┐
    │  W1 layer:  sum = a + b,  diff = |a - b|                        │
    │  SwiGLU:    eq_gate lookups compute sum², diff²                 │
    │  W2 layer:  result = (sum² - diff²) / 4                         │
    └─────────────────────────────────────────────────────────────────┘
    
  DIV (Attention + Softmax):
    ┌─────────────────────────────────────────────────────────────────┐
    │  Attention: Query b against keys [1,2,4,8,...] → log₂(b)        │
    │  Softmax:   exp(-log₂(b) × ln2) = 1/b  (EXACT via ratio trick)  │
    │  FFN:       a × (1/b) via quarter-squares                       │
    └─────────────────────────────────────────────────────────────────┘

  Both use ONLY standard primitives:
    - Attention (Q @ K.T, softmax, @ V)
    - FFN (W2 @ silu(W1 @ x))
    - Constructed weights (no learning)
""")
