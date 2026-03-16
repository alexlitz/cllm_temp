"""
Test: fp64 magic number floor trick in SwiGLU architecture.

Validates the 4-layer DIV pipeline:
  Layer 1: clear + gather + reciprocal (merged)
  Layer 2: multiply Q = dividend * reciprocal
  Layer 3: floor(Q/16^j) via MAGIC, store to TEMP[j]
  Layer 4: nibble_j = floor_j - 16*floor_{j+1}, write to RESULT[j]

The MAGIC trick: floor(x) = (x - 0.5 + eps + MAGIC) - MAGIC
  MAGIC = 3 * 2^51 (at this scale, ULP = 1, so addition rounds to integer)
  eps = 2^(-20-4j) breaks round-to-even ties for exact-integer inputs

Precision guarantee:
  - Layer 3: h_j = silu(Q/16^j - 0.5 + eps + MAGIC) ≈ MAGIC + floor(Q/16^j)
    W_down subtracts MAGIC (exact cancellation in single hidden unit → output pair)
  - Layer 4: nibble_j = floor_j - 16*floor_{j+1} (small values, full precision)
"""

import numpy as np
import random

# ============================================================
# Constants
# ============================================================

MAGIC = np.float64(3.0 * 2**51)  # 6755399441055744.0

def eps_for_j(j):
    """Per-j epsilon = ULP at max Q/16^j scale. Breaks round-to-even ties."""
    return np.float64(2**(-20 - 4*j))

def silu(x):
    x = np.float64(x)
    return x / (1.0 + np.exp(-x))

def softmax1_reciprocal(divisor):
    """1/divisor via softmax1 nibble construction."""
    n = int(divisor)
    nibbles = []
    for j in range(8):
        nibbles.append(n & 0xF)
        n >>= 4
    S = 60.0
    scores = []
    for j, d in enumerate(nibbles):
        if d > 0:
            scores.append(np.log(np.float64(16**j * d)))
        else:
            scores.append(-S)
    scores = np.array(scores, dtype=np.float64)
    mx = np.max(scores)
    ex = np.exp(scores - mx)
    return np.float64(np.exp(-mx) / np.sum(ex))


# ============================================================
# Simulated SwiGLU layers
# ============================================================

def simulate_layer3_floor(Q_float):
    """Layer 3: compute floor(Q/16^j) via MAGIC trick, store to TEMP[j]."""
    Q = np.float64(Q_float)
    opcode = np.float64(1.0)

    floors = []
    for j in range(8):
        scale = np.float64(1.0 / 16**j)
        eps = eps_for_j(j)
        offset = np.float64(-(0.5 - eps))

        # SwiGLU hidden unit: up = scale*Q + offset*opcode + MAGIC
        up = scale * Q + offset * opcode + MAGIC
        h = silu(up) * opcode  # silu ≈ identity for huge positive

        # W_down: TEMP[j] = 1.0 * h + b_down(-MAGIC)
        floor_val = np.float64(1.0) * h + np.float64(-MAGIC)
        floors.append(floor_val)

    return floors


def simulate_layer4_nibbles(floors):
    """Layer 4: nibble_j = floor_j - 16*floor_{j+1}."""
    S = np.float64(100.0)
    nibbles = []
    for j in range(8):
        fj = np.float64(floors[j])
        h = silu(S * fj) / S  # SwiGLU identity for positive val

        if j < 7:
            fjp1 = np.float64(floors[j + 1])
            hsub = silu(S * fjp1) / S
            nib = h - np.float64(16.0) * hsub
        else:
            nib = h

        nibbles.append(int(round(nib)))
    return nibbles


def full_div_pipeline(dividend, divisor):
    """Full 4-layer DIV pipeline simulation."""
    recip = softmax1_reciprocal(divisor)
    Q_float = np.float64(dividend) * recip
    floors = simulate_layer3_floor(Q_float)
    nibbles = simulate_layer4_nibbles(floors)
    return sum(nibbles[j] * 16**j for j in range(8))


# ============================================================
# Tests
# ============================================================

random.seed(42)

# Test 1: Targeted cases (including previously failing ones)
print("=" * 72)
print("  Test 1: Targeted cases")
print("=" * 72)

targeted = [
    (42, 7, 6), (100, 10, 10), (1000, 33, 30),
    (255, 16, 15), (65535, 255, 257), (999999, 7, 142857),
    (0, 5, 0), (5, 1, 5), (15, 15, 1),
    (1000000, 1000000, 1),
    (2**32-1, 1, 2**32-1),    # max odd quotient
    (2**32-1, 2, 2**31-1),    # large odd quotient
    (2**31-1, 3, (2**31-1)//3),
    (2**32-1, 17, (2**32-1)//17),
    (2946529189, 1473538672, 1),  # prev failure: near-integer
    (3066343536, 1022218630, 2),  # prev failure: near-integer
    (1436260363, 1436925100, 0),  # prev failure: near-integer
]
t_fail = 0
for a, b, exp in targeted:
    r = full_div_pipeline(a, b)
    ok = "OK" if r == exp else "FAIL"
    if r != exp: t_fail += 1
    print(f"  {ok}: {a}/{b} = {r}" + (f" (expected {exp})" if r != exp else ""))
print(f"\n  Targeted: {t_fail}/{len(targeted)} failures")


# Test 2: Random stress test (200k)
print(f"\n{'=' * 72}")
print("  Test 2: Random stress test (200k)")
print("=" * 72)

random.seed(42)
r_fail = 0
fail_ex = []
for _ in range(200000):
    a = random.randint(0, 2**32-1)
    b = random.randint(1, 2**32-1)
    exp = a // b
    r = full_div_pipeline(a, b)
    if r != exp:
        r_fail += 1
        if len(fail_ex) < 5:
            fail_ex.append((a, b, r, exp))
for a, b, r, e in fail_ex:
    print(f"  FAIL: {a}/{b} = {r} (expected {e})")
print(f"  Result: {r_fail}/200000 failures")


# Test 3: Small divisor sweep (common in C programs)
print(f"\n{'=' * 72}")
print("  Test 3: Small divisor sweep")
print("=" * 72)

s_fail = 0
for b in range(1, 101):
    for a in list(range(0, 201)) + [b*k for k in range(1, 1000)] + [2**32-1, 2**31-1, 2**20]:
        if a > 2**32-1: continue
        exp = a // b
        r = full_div_pipeline(a, b)
        if r != exp:
            s_fail += 1
            if s_fail <= 5:
                print(f"  FAIL: {a}/{b} = {r} (expected {exp})")
print(f"  Small divisor sweep: {s_fail} failures")


# Test 4: W_down MAGIC cancellation precision
print(f"\n{'=' * 72}")
print("  Test 4: W_down MAGIC cancellation (layer 3)")
print("=" * 72)

w_fail = 0
for Q in [0, 1, 5, 15, 16, 255, 256, 65535, 2**20, 2**31-1, 2**32-1]:
    for j in range(8):
        val = np.float64(Q) / np.float64(16**j)
        eps = eps_for_j(j)
        up = val - np.float64(0.5) + eps + MAGIC
        h = silu(up)
        result = np.float64(1.0) * h + np.float64(-MAGIC)

        # Also compute via MAGIC floor for comparison
        tmp = val - np.float64(0.5) + eps + MAGIC
        expected = tmp - MAGIC

        if result != expected:
            w_fail += 1
            if w_fail <= 3:
                print(f"  FAIL Q={Q} j={j}: {result} != {expected}")
print(f"  W_down cancellation: {w_fail} failures")


# Test 5: Per-j eps specification
print(f"\n{'=' * 72}")
print("  Test 5: SwiGLU weight specification")
print("=" * 72)

print(f"  MAGIC = {MAGIC} (3*2^51, exact fp64)")
for j in range(8):
    eps = eps_for_j(j)
    w_Q = np.float64(1.0/16**j)
    w_op = np.float64(-(0.5 - eps))
    print(f"  j={j}: W_up[Q]={w_Q:.6e}  W_up[op]={w_op:.15f}  eps={eps:.3e}")


# Summary
print(f"\n{'=' * 72}")
print("  SUMMARY")
print(f"{'=' * 72}")
print(f"""
  4-LAYER DIV PIPELINE (was 11 layers, 2.75x reduction):

  Layer 1: Clear + Gather + Softmax1 Reciprocal [MERGED]
  Layer 2: Multiply Q_float = dividend * reciprocal
  Layer 3: Floor extraction: floor(Q/16^j) -> TEMP[j] via MAGIC trick
           eps_j = 2^(-20-4j) for round-to-even tie-breaking
           8 hidden units (floor) + 16 (clear) = 24 total
  Layer 4: Nibble extraction: nibble_j = TEMP[j] - 16*TEMP[j+1] -> RESULT[j]
           Small values, full fp64 precision
           7*4 + 3 = 31 hidden units

  MOD: 15 layers (was 22, 4 for quotient + 7 multiply + 3 subtract + 1 correction)

  Results: {t_fail} targeted + {r_fail} random + {s_fail} small-div = {t_fail+r_fail+s_fail} total failures
""")
