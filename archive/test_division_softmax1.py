"""
Division via softmax1: computing 1/n with binary and nibble constructions.

Both use: leftover = 1/(1 + sum_j exp(s_j)).
Set up scores so sum_j exp(s_j) = n-1, then leftover = 1/n.

BINARY (32 tokens):
  Keys = I_32 (one-hot). Query encodes n-1 in binary.
  q[k] = b_k·(S+k)·ln2 - S·ln2
    active (b=1):  score = k·ln2  → exp = 2^k   (place value)
    inactive (b=0): score = -S·ln2 → exp ≈ 0     (suppressed)

NIBBLE (8 tokens):
  n-1 = Σ_j 16^j · d_j,  d_j ∈ {0..15}
  Score for token j: log(16^j · d_j) if d_j>0, else -S (suppressed)
  Implemented as dot product: 128-dim query (one-hot per 16-block),
  key[j] stores the 16-entry lookup table L_j[d] = log(16^j·d).
"""

import numpy as np

ln = np.log
S = 60  # suppression scale


def softmax1_leftover(scores):
    """1 / (1 + Σ exp(s_j)), numerically stable."""
    if len(scores) == 0:
        return 1.0
    mx = np.max(scores)
    ex = np.exp(scores - mx)
    return np.exp(-mx) / (np.exp(-mx) + np.sum(ex))


# ================================================================
# BINARY: 32 tokens, one-hot keys
# ================================================================

K_bin = np.eye(32)

def recip_binary(n):
    nm1 = n - 1
    q = np.zeros(32)
    for k in range(32):
        b = (nm1 >> k) & 1
        q[k] = b * (S + k) * ln(2) - S * ln(2)
    return softmax1_leftover(K_bin @ q)


# ================================================================
# NIBBLE: 8 tokens, base-16 lookup tables
# ================================================================

K_nib = np.zeros((8, 128))
for j in range(8):
    K_nib[j, j * 16] = -S  # d=0 → suppressed
    for d in range(1, 16):
        K_nib[j, j * 16 + d] = ln(float(16**j * d))

def recip_nibble(n):
    nm1 = n - 1
    q = np.zeros(128)
    for j in range(8):
        d = (nm1 >> (4 * j)) & 0xF
        q[j * 16 + d] = 1.0
    return softmax1_leftover(K_nib @ q)


# ================================================================
# TEST HARNESS
# ================================================================

def test(name, fn, vals):
    print(f"\n{'='*72}")
    print(f"  {name}")
    print(f"{'='*72}")
    print(f"{'n':>12} | {'computed':>20} | {'exact':>20} | {'error':>12}")
    print("-" * 72)
    mx = 0
    for n in vals:
        r = fn(n)
        e = 1.0 / n
        err = abs(r - e)
        mx = max(mx, err)
        ok = " ✓" if err < 1e-10 else ""
        print(f"{n:12d} | {r:20.16f} | {e:20.16f} | {err:12.2e}{ok}")
    print(f"  Max error: {mx:.2e}")
    return mx


vals = sorted(set([
    1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16, 17, 31, 32, 33,
    100, 255, 256, 257, 1000, 1023, 1024,
    65535, 65536, 100000, 1000000,
    2**24, 2**24 + 1, 2**30, 2**31 - 1, 2**32 - 1
]))

test("BINARY (32 tokens, one-hot keys, SCALE suppression)", recip_binary, vals)
test("NIBBLE (8 tokens, base-16 lookup)", recip_nibble, vals)


# ================================================================
# Verify the sums are exactly n-1
# ================================================================
print(f"\n{'='*72}")
print("  Verify: sum_exp = n - 1")
print(f"{'='*72}")
print(f"{'n':>12} | {'n-1':>12} | {'binary sum':>18} | {'nibble sum':>18}")
print("-" * 72)
for n in [1, 2, 7, 256, 65537, 2**31 - 1, 2**32 - 1]:
    nm1 = n - 1
    # binary
    q = np.zeros(32)
    for k in range(32):
        b = (nm1 >> k) & 1
        q[k] = b * (S + k) * ln(2) - S * ln(2)
    sb = np.sum(np.exp(K_bin @ q))
    # nibble
    q2 = np.zeros(128)
    for j in range(8):
        d = (nm1 >> (4 * j)) & 0xF
        q2[j * 16 + d] = 1.0
    sn = np.sum(np.exp(K_nib @ q2))
    print(f"{n:12d} | {nm1:12d} | {sb:18.10f} | {sn:18.10f}")


# ================================================================
# Show nibble lookup tables
# ================================================================
print(f"\n{'='*72}")
print("  Nibble lookup tables: L_j[d] = log(16^j · d)")
print(f"{'='*72}")
for j in range(8):
    entries = []
    for d in range(16):
        if d == 0:
            entries.append("-∞")
        else:
            v = ln(float(16**j * d))
            entries.append(f"{v:.2f}")
    if j < 3:
        print(f"  j={j}: [{', '.join(entries[:8])}]")
        print(f"        [{', '.join(entries[8:])}]")
    else:
        print(f"  j={j}: [d=0: -∞,  d=1: {ln(float(16**j)):.2f}, "
              f" d=15: {ln(float(16**j * 15)):.2f}]")


# ================================================================
# Division a/b via two-step: leftover gives 1/b, scale by a
# ================================================================
print(f"\n{'='*72}")
print("  Division a/b = a · (1/b) via nibble 1/b")
print(f"{'='*72}")
print(f"{'a':>6} / {'b':>6} | {'computed':>18} | {'exact':>18} | {'error':>12}")
print("-" * 72)
for a, b in [(7, 3), (100, 7), (1000, 13), (65536, 255), (2**20, 17),
             (1, 2**31 - 1), (2**31 - 1, 1)]:
    r = a * recip_nibble(b)
    e = a / b
    err = abs(r - e)
    print(f"{a:6d} / {b:6d} | {r:18.12f} | {e:18.12f} | {err:12.2e}")


# ================================================================
# Causal constraint: first-8-token problem
# ================================================================
print(f"\n{'='*72}")
print("  CAUSAL CONSTRAINT (nibble version)")
print(f"{'='*72}")
print("""
  In a causal transformer, position i attends to positions 0..i-1.
  If nibble lookup tokens occupy positions 0-7:

    Position 0:  sees 0 nibble tokens → no division
    Position 1:  sees 1 nibble token  → only nibble 0 (mod 16)
    ...
    Position 7:  sees 7 nibble tokens → missing nibble 7
    Position 8+: sees all 8 nibble tokens → FULL DIVISION ✓

  First 8 positions cannot compute exact 1/n.
  Partial workaround: positions 1-7 can divide by small n
  (where high nibbles are 0 and don't matter).
""")

# Demonstrate partial division in early positions
print("  Partial division with limited nibble tokens:")
print(f"  {'pos':>4} | {'n':>6} | {'nibbles used':>14} | {'result':>18} | {'exact':>18} | {'ok':>4}")
print("  " + "-" * 72)
for pos in range(1, 9):
    # position `pos` sees nibble tokens 0..pos-1
    for n in [2, 16, 256, 65536]:
        nm1 = n - 1
        q = np.zeros(128)
        for j in range(min(pos, 8)):  # only see first `pos` tokens
            d = (nm1 >> (4 * j)) & 0xF
            q[j * 16 + d] = 1.0
        # only use first `pos` nibble tokens
        scores = K_nib[:min(pos, 8)] @ q
        r = softmax1_leftover(scores)
        e = 1.0 / n
        ok = "✓" if abs(r - e) < 1e-10 else "✗"
        high_nibbles_zero = all((nm1 >> (4*j)) & 0xF == 0 for j in range(pos, 8))
        if high_nibbles_zero:
            ok += " (all high nibbles 0)"
        print(f"  {pos:4d} | {n:6d} | {min(pos,8):>14d} | {r:18.14f} | {e:18.14f} | {ok}")
