"""
Test: softmax1 attention computing 1/(2+N) with binary-encoded queries.

softmax1 denominator = 1 + sum_j exp(s_j)
                          ^   ^^^^^^^^^^^
                          |   token contributions
                          phantom "1"

With token 0 (value=1, score=0):
  output = attention_0 = 1 / (1 + exp(0) + sum_bit) = 1/(2 + sum_bit)
  Need sum_bit = N  →  inactive tokens must contribute ≈ 0

With NO token 0, using leftover:
  leftover = 1 / (1 + sum_bit)
  Need sum_bit = 1+N  →  inactive tokens can contribute 1/n each (exact!)
"""

import numpy as np

ln = np.log


def softmax1_output(query, keys, values, biases=None):
    """Standard softmax1: output = sum(attention_i * value_i)."""
    scores = keys @ query
    if biases is not None:
        scores = scores + biases
    mx = np.max(scores)
    ex = np.exp(scores - mx)
    denom = np.exp(-mx) + np.sum(ex)
    attn = ex / denom
    return np.sum(attn * values), denom * np.exp(mx)  # output, raw_denom


def softmax1_leftover(query, keys, biases=None):
    """Leftover = 1 - sum(attention) = 1/(1 + sum exp(s_j))."""
    scores = keys @ query
    if biases is not None:
        scores = scores + biases
    mx = np.max(scores)
    ex = np.exp(scores - mx)
    denom = np.exp(-mx) + np.sum(ex)
    return np.exp(-mx) / denom  # = 1/(1 + sum_exp)


def run_test(name, n_bits, compute_fn, max_N=None):
    if max_N is None:
        max_N = 2**n_bits - 1
    print(f"\n{'N':>3} | {'output':>15} | {'1/(2+N)':>15} | {'ratio':>10} | {'error':>12}")
    print("-" * 70)
    max_err = 0
    for N in range(max_N + 1):
        out = compute_fn(N)
        exp = 1.0 / (2 + N)
        err = abs(out - exp)
        max_err = max(max_err, err)
        r = out / exp if exp > 0 else 0
        mark = " ✓" if err < 1e-6 else ""
        print(f"{N:3d} | {out:15.10f} | {exp:15.10f} | {r:10.8f} | {err:12.2e}{mark}")
    print(f"\nMax error: {max_err:.2e}")
    return max_err


# ============================================================
# APPROACH 1: Token 0 (value=1) + SCALE suppression
# output = attention_0 = 1/(2 + sum_bit)
# Need sum_bit ≈ N, so inactive tokens → exp ≈ 0
# ============================================================
print("=" * 70)
print("APPROACH 1: Value token + SCALE suppression (approximate)")
print()
print("  Token 0: score=0, value=1")
print("  Bit tokens: key=(SCALE+i)*log2, bias=-SCALE*log2")
print("  b=0: score=-SCALE*log2 → exp ≈ 0  (suppressed)")
print("  b=1: score=i*log2      → exp = 2^i (place value)")
print("  sum_bit ≈ N, output = 1/(2 + sum_bit) ≈ 1/(2+N)")
print("=" * 70)

SCALE = 30
n = 5
n_tok = 1 + n
K1 = np.zeros((n_tok, n))
B1 = np.zeros(n_tok)
V1 = np.zeros(n_tok)
V1[0] = 1.0
for i in range(n):
    K1[i+1, i] = (SCALE + i) * ln(2)
    B1[i+1] = -SCALE * ln(2)


def approach1(N):
    q = np.zeros(n)
    for i in range(n):
        q[i] = float((N >> i) & 1)
    out, _ = softmax1_output(q, K1, V1, B1)
    return out


run_test("SCALE suppression", n, approach1)
print(f"  Suppression: inactive exp = 2^(-{SCALE}) = {2**(-SCALE):.2e}")


# ============================================================
# APPROACH 2: Leftover (no value token) + exact coefficients
# leftover = 1/(1 + sum_bit)
# Need sum_bit = 1+N, so inactive tokens contribute 1/n each
# ============================================================
print("\n" + "=" * 70)
print("APPROACH 2: Leftover + exact coefficients (no approximation)")
print()
print("  NO token 0. Only n bit tokens.")
print("  key_i = log(1 + n*2^i),  bias = -log(n)")
print("  b=0: exp = 1/n              (n of these sum to 1)")
print("  b=1: exp = 1/n + 2^i        (adds exact place value)")
print("  sum = 1 + N,  leftover = 1/(2+N)  EXACT")
print("=" * 70)

K2 = np.zeros((n, n))
B2 = np.zeros(n)
for i in range(n):
    K2[i, i] = ln(1 + n * 2**i)
    B2[i] = -ln(n)


def approach2(N):
    q = np.zeros(n)
    for i in range(n):
        q[i] = float((N >> i) & 1)
    return softmax1_leftover(q, K2, B2)


run_test("Leftover exact", n, approach2)

# Show the math
print("\nCoefficients (n=5 bits):")
for i in range(n):
    k = ln(1 + n * 2**i)
    b = -ln(n)
    e0 = np.exp(b)
    e1 = np.exp(k + b)
    print(f"  bit {i} (place value {2**i:2d}): key={k:.4f}, bias={b:.4f}"
          f"  →  exp(b=0)={e0:.4f}=1/{n}, exp(b=1)={e1:.4f}=1/{n}+{2**i}")

print(f"\n  Verify sum: {n}×(1/{n}) = 1  (inactive baseline)")
print(f"  Active bits add: 2^0=1, 2^1=2, 2^2=4, 2^3=8, 2^4=16")
print(f"  Total when N=31: 1 + 31 = 32, leftover = 1/33 = 1/(2+31) ✓")


# ============================================================
# APPROACH 2b: Same but using dot product (no additive bias)
# Encode bias in key[0], use query = [1, bits]
# ============================================================
print("\n" + "=" * 70)
print("APPROACH 2b: Leftover + dot product form (query[0]=1)")
print("  key = [-log(n), 0, ..., log(1+n*2^i), ..., 0]")
print("  query = [1, b_0, b_1, ..., b_{n-1}]")
print("=" * 70)

d2b = 1 + n
K2b = np.zeros((n, d2b))
for i in range(n):
    K2b[i, 0] = -ln(n)
    K2b[i, i+1] = ln(1 + n * 2**i)


def approach2b(N):
    q = np.zeros(d2b)
    q[0] = 1.0
    for i in range(n):
        q[i+1] = float((N >> i) & 1)
    return softmax1_leftover(q, K2b)


run_test("Leftover dot-product", n, approach2b)


# ============================================================
# APPROACH 2c: Leftover + zero pad + additive bias
# query = [0, b_0, ..., b_{n-1}]  (zero pad, no "1")
# -SCALE stored as additive bias
# ============================================================
print("\n" + "=" * 70)
print("APPROACH 2c: Leftover + zero pad + additive bias")
print("  query = [0, b_0, b_1, ..., b_{n-1}]")
print("  key_i has 1 at position i+1 (unit vector)")
print("  bias_i = -log(n)  (additive, constant)")
print("  key coefficient at pos i+1 = log(1 + n*2^i)")
print("=" * 70)

d2c = 1 + n
K2c = np.zeros((n, d2c))
B2c = np.zeros(n)
for i in range(n):
    K2c[i, i+1] = ln(1 + n * 2**i)  # the "1" at encoding position, scaled
    B2c[i] = -ln(n)                  # -SCALE (here SCALE = log(n))


def approach2c(N):
    q = np.zeros(d2c)
    q[0] = 0.0  # zero pad
    for i in range(n):
        q[i+1] = float((N >> i) & 1)
    return softmax1_leftover(q, K2c, B2c)


run_test("Leftover zero-pad", n, approach2c)


# ============================================================
# Scale test: works for any n
# ============================================================
print("\n" + "=" * 70)
print("SCALING TEST: exact leftover works for any n")
print("=" * 70)

for n_bits in [5, 8, 10, 16, 20, 32]:
    max_N = 2**n_bits - 1
    K_n = np.zeros((n_bits, n_bits))
    B_n = np.zeros(n_bits)
    for i in range(n_bits):
        K_n[i, i] = ln(1 + n_bits * 2**i)
        B_n[i] = -ln(n_bits)

    test_vals = sorted(set([0, 1, 2, max_N//4, max_N//2, 3*max_N//4, max_N-1, max_N]))
    max_err = 0
    for N in test_vals:
        q = np.zeros(n_bits)
        for i in range(n_bits):
            q[i] = float((N >> i) & 1)
        out = softmax1_leftover(q, K_n, B_n)
        err = abs(out - 1.0/(2+N))
        max_err = max(max_err, err)

    print(f"  n={n_bits:2d} bits, N up to {max_N:>12d}: max_error = {max_err:.2e}")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
Two working constructions for 1/(2+N) via softmax1:

APPROACH 1 (approximate): Token with value=1 + SCALE suppression
  - Token 0: score=0, value=1  →  output = attention_0
  - Bit tokens: key=(SCALE+i)*log2, bias=-SCALE*log2
  - Inactive exp ≈ 2^(-SCALE) ≈ 0, so sum_bit ≈ N
  - output = 1/(2 + sum_bit) ≈ 1/(2+N)
  - Error: ~n * 2^(-SCALE)

APPROACH 2 (exact): Leftover with balanced coefficients
  - NO value token. n bit tokens only.
  - key_i = log(1 + n*2^i),  bias = -log(n)
  - Inactive: exp = 1/n  (n of them sum to exactly 1)
  - Active:   exp = 1/n + 2^i  (adds exact place value)
  - sum = 1 + N (always, independent of popcount!)
  - leftover = 1/(1 + sum) = 1/(2+N)  EXACT
  - "Invert it" = use leftover (1 - sum_attention) as output
  - Zero-pad query works (no "1" needed)
  - bias = -log(n) is constant (like -SCALE with SCALE=log(n))
""")
