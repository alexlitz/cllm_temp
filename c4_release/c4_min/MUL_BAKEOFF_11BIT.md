# 11-bit-chunk 32-bit multiply — MSB-first subtractive digit peel

Bake-off candidate in `c4_min/mul_bakeoff_11bit.py` (stand-alone; composes the
proven `nibble_alu32.py` primitives — `_mul_gate`, `_empty_spec`, `RELU_S`, `S`,
`_clear`/`_ident`/`_truncate` — and edits NOTHING; does not touch `mul_bakeoff.py`).

Run: `python -m c4_min.mul_bakeoff_11bit`

## The design (as specced)

- Split each 32-bit operand into **3 chunks of 11 bits** (`a = a2·2²² + a1·2¹¹ + a0`,
  each `< 2048`).
- **6 products** `a_i·b_j`, `i+j≤2` (11×11 = 22-bit each), via `_mul_gate`.
- **Accumulate into 3 columns** at weights `2⁰, 2¹¹, 2²²`. Column 2 sums 3 products
  `≤ 3·2047² = 12,570,627 < 2²⁴` — the column VALUE has fp32 headroom (confirmed).
- **Assemble**: peel each column into nibbles MSB-first, then
  `result = (col0 + col1·2¹¹ + col2·2²²) & 0xFFFFFFFF`.

### MSB-first subtractive digit extraction (the trick)

```
n_top = floor(x / 2^(4·top))       # divide by the LARGEST remaining power → 0..15
x     = x - n_top · 2^(4·top)      # subtract it out; x now < 2^(4·top)
... repeat down to n_0
```

`floor(x/2^p)` (0..15) = `Σ_{k=1..15}[x ≥ k·2^p]` — **15 tripwires per digit,
regardless of how big x is** (we divide by the large power). Each tripwire is an
EXACT INTEGER UNIT STEP with no `1/w` ramp amplification:

```
[x ≥ t] = relu(x − t + 1) − relu(x − t)        # exact 0/1 for integer x
```

Confirmed numerically (`_selftest_msb_peel_math`): decompose→recompose is **EXACT**
on 100k random 24-bit values, and the widest relu argument is **16,777,013 < 2²⁴** —
so the peel ARITHMETIC uses **no MAGIC-floor and no fp64**. That part of the
prompt's thesis holds.

## The substrate precision ceiling (the honest, load-bearing finding)

The MSB-peel math is fp32-clean, but the `relu` is realised in this project's
SwiGLU substrate as `silu(RELU_S·z)/RELU_S` with `RELU_S = 200`. That identity is
fp32-EXACT only while the SiLU argument `RELU_S·|z| < 2²⁴`, i.e.
`|z| < 2²⁴/200 ≈ 83,886 ≈ 2^16.4`. Measured (`_selftest_peel_ceiling`):

| silu-relu unit step, argument | max abs error vs exact 0/1 |
|-------------------------------|-----------------------------|
| arg ≤ 2¹⁵ (32,768)            | **0.0** (exact)             |
| arg ≤ 2¹⁷ (131,072)          | **0.0** (exact)             |
| arg ≤ 2²⁰ (1,048,576)        | ~0.0625                     |
| arg ≤ 2²² (4,194,304)        | ~0.25  ← 11-bit product/col |
| arg ≤ 2²⁴ (16,777,216)       | ~1.0   ← summed column      |

The `+1` that makes the unit step crisp is swamped by the ~256 granularity of a
`200·2²² ≈ 8.4e8` intermediate. Lowering RELU_S so `RELU_S·2²² < 2²⁴`
(`RELU_S < 4`) is the other horn: the step stops being sharp at the integer
boundary (`silu(3)/3` at z=1 is ~0.29, not ~1). **No single RELU_S is both sharp
at the boundary AND exact at 2²².**

Because 11-bit chunking makes the products (≤ 2²²) and columns (≤ 2²⁴) the things we
peel, their tripwire arguments live ABOVE the ~2¹⁷ substrate ceiling. The peel of a
large product/column therefore misfires. Isolation test confirms the wiring is
correct — the loss is purely the substrate ceiling:

```
3·5=15 OK    255·255=65025 OK    (args < 2¹⁷ → exact)
1000·1000 = 950110  (want 1,000,000)  FAIL   (product ≈ 2²⁰ → above ceiling)
```

`_mul_gate` itself is also inexact for 11-bit operands: its hidden holds
`60·a_i·b_j` up to `60·2²² ≈ 2.5e8 > 2²⁴`, giving a ≤ 0.25 residue (the peel is
meant to snap it, but the peel is itself over the ceiling).

## Results table

| candidate | depth (blocks) | weights (nz) | fp32/fp64 | byte-exact |
|-----------|---------------:|-------------:|-----------|-----------:|
| **11-bit-chunk (MSB-peel)** | **64** | **16,757** | **fp32 math (no fp64/magic); substrate silu-relu inexact > 2¹⁷** | **11 / 269** |
| nibble schoolbook (baseline) | 10 | 14,370 | fp32 (all args < 256) | 250 / 250 (100%) |

(Depth 64 = 1 products + 6 products × 6 nibbles peel + 1 accum + 3 columns × 6
nibbles peel + 8 result nibbles. The MSB peel is ~1 block/nibble, sequential.)

## Honest verdict vs the nibble-schoolbook baseline

- **Does 6-products + MSB-peel beat 36-products + split + 7-carry on weights?**
  **No.** The 6-product front-end is a genuine win (6 `_mul_gate`s vs 36), but the
  MSB peel of nine 22-bit products/columns needs 6 nibbles × 30 units each, so the
  total lands at **16,757 nz — slightly MORE than the baseline's 14,370.**
- **On depth?** **No.** 64 blocks vs 10. The per-nibble subtractive peel is
  inherently sequential (each block subtracts the higher nibbles written by prior
  blocks), and we peel 9 wide values × 6 nibbles + 8 result nibbles.
- **Does it stay fully fp32-vanilla?** The **math** does (no fp64, no magic-floor,
  widest relu arg < 2²⁴). But the silu-relu **substrate** does not: 11-bit chunk
  arguments (2²²–2²⁴) sit above its ~2¹⁷ exactness ceiling, so end-to-end it is
  **only 11/269 byte-exact** — small products pass, anything ≥ ~2²⁰ fails.

**Bottom line.** MSB-first peel is the *correct, fp32-clean* digit-extraction
algorithm (it beats magic-floor/staircase, which need fp64) — the prompt's core
claim about the peel is right. But it does not rescue the 11-bit chunking, because
the win of MSB-first (results 0..15) does not shrink the tripwire *argument*, which
is still the whole 22–24-bit value, and that argument exceeds this substrate's
silu-relu precision ceiling. The baseline nibble schoolbook wins on every axis here
precisely because it keeps EVERY argument `< 256` (`RELU_S·255 = 51k << 2²⁴`), so it
is deeply fp32-exact. **For a genuinely fp32-exact wide multiply in the RELU_S=200
silu substrate, keep the peel arguments small (≤ ~2¹⁷) — i.e. use nibble-sized
chunks, not 11-bit chunks.**
