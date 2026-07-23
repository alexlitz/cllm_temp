# Byte multiplier, carry-save accumulation, tunable RELU_S — head-to-head

`c4_min/mul_byte_carrysave.py` builds and MEASURES a 32-bit
`(a*b) & 0xFFFFFFFF` multiplier as persistent SwiGLU FFN blocks, composed from
the proven `nibble_alu32` primitives (`_mul_gate`, `_empty_spec`, `_ident`,
`_clear`, `_truncate`, `S`). It does not edit those, `mul_bakeoff.py`, or
`mul_bakeoff_11bit.py`. It re-derives exactly one thing — a **tunable-RELU_S**
tripwire / peel — so the RELU_S knob is an explicit per-gadget parameter with an
fp32-headroom assertion.

Run: `python -m c4_min.mul_byte_carrysave`.

## The substrate fact (why RELU_S is a knob)

Under a transformer, `relu(z)` is realised as `silu(RELU_S*z)/RELU_S`, and a
tripwire `[x >= t]` is a difference of two such units. Two constraints pin
RELU_S from opposite sides:

- **Sharpness (lower bound).** The unit step needs `silu(RELU_S*1) ≈ RELU_S`, so
  the 0→1 transition is crisp on integers. Measured: RELU_S ∈ {20, 32, 64, 200}
  all give a bit-exact step (`step(x=t)=1.000000`, `step(x=t-1)=0`); RELU_S=8
  leaks ~3.3e-4, RELU_S=4 leaks ~1.8e-2. So RELU_S ≥ ~20 is required.
- **fp32 exactness (upper bound).** The SiLU argument AND the baked bias
  `RELU_S*t` are fp32 numbers; fp32 holds integers exactly only below 2²⁴. So a
  gadget is exact iff **`RELU_S · max(|x|, |t|) < 2²⁴`**. Max exact argument is
  `2²⁴/RELU_S` (RELU_S=200 → 83 886; RELU_S=64 → 262 144; RELU_S=32 → 524 288).

Every gadget is built with an explicit `relu_s=` and asserts
`relu_s · max_arg < 2²⁴` (`_assert_headroom`); the harness reports the single
tightest ratio across the whole gadget. The assertion is enforced, not just
claimed: `build_byte_carrysave(..., relu_s=128)` is rejected with
`need relu_s < 64.5`.

## The design — byte multiplier, carry-save

1. **4 byte-chunks** per 32-bit operand → **10 byte-products** `a_i·b_j` (i+j<4),
   each an 8×8 = 16-bit product ≤ 65 025 via `_mul_gate`. The gated multiply's
   internal `silu(S·a)·b ≈ S·a·b ≤ 60·255·255 = 3 901 500 < 2²⁴` for byte
   operands, so the products are fp32-exact.
2. **Accumulate into 4 byte-position columns** (column c = weight 256^c), **no
   per-product split**. Column c gathers the products with `i+j == c`, so a
   column holds at most 4 products → **≤ 4·65 025 = 260 100**. This is the
   carry-SAVE step: raw partial products summed into their weighted column, no
   intermediate normalisation.
3. **Decompose each 260 100-column into nibbles ONCE**, MSB-first, at a low
   RELU_S (≤ 64 so `RELU_S·260 100 < 2²⁴`). Then combine the 4 columns at their
   byte offsets (column c → result nibbles 2c..2c+4) with a base-16 carry ripple.

MSB-first peel: for the largest remaining nibble-power `2^p`,
`n = floor(x/2^p) ∈ 0..kmax`, then `x −= n·2^p`, repeat down. `kmax = 15` for
every power except the **top**, where it is **tight** (`floor(colmax/2^p)`).

### The subtle trap (why the tightest arg is the column, not the threshold)

A naïve peel with `kmax=15` at the top nibble bakes a bias `RELU_S·15·65536 =
RELU_S·983 040` — a threshold **larger than the column value**, which is
fp32-lossy *even though it never fires*. With a **tight per-power kmax** the
largest baked threshold at 2¹⁶ is `3·65536 = 196 608 < 260 100`, so the binding
argument is the column value itself (260 100). This is exactly what
"report the tightest one" is meant to surface.

### The three fp-discipline details that make it byte-exact

The wide byte products carry ~0.02 fp residue; peeling and combining a 260 100
column amplifies it, and it compounds. Three fixes (all in the code):

- **Sharp half-integer ramps** (threshold `k·m − 0.5`, width 0.25) in both the
  MSB-first digit tripwire and the base-16 carry floor — NOT the integer unit
  step. A carry chain does not deliver clean integers (a lane arrives at ~15.96);
  the integer unit step floors 15.96 to 15 and starves the carry, but the ramp
  flips at 15.5, safely between near-integer values. (Same reason the library
  `_floor_div_pow` uses `thr − 0.5`.)
- **Snap each peeled digit to a clean integer** (`_snap_nibble`) in its own block
  *before* the `digit · 2^p` residue subtract. A raw ramp digit carries ~3e-4;
  multiplied by 2^p (up to 65 536) that is ~23 of spurious residue that flips a
  low nibble. Snapping first makes the subtract lossless.
- **The snap uses a HIGH RELU_S (200), independent of the peel's low RELU_S.**
  The snap acts only on nibble values ≤ 15.5 (thresholds ≤ 14.5 → `RELU_S·14.5`
  is tiny → oceans of headroom), so a bit-*perfect* snap is free. At the peel's
  RELU_S=32 a snap-in-place would leave ~3e-4 that `×65536` re-amplifies; the
  high-RELU_S snap is what lets the low-RELU_S RS=32 peel stay correct.

## Results (byte-exact on ~425 structured + random `(a,b)`)

All four variants are **byte-exact 425/425** (incl. 0, 0xFFFFFFFF·0xFFFFFFFF,
0xDEADBEEF, powers of two, single-byte and full-width; plus 605 extended random
+ adversarial pairs at both RELU_S → 0 fails).

| VARIANT              | DEPTH | WEIGHTS (nz) | tightest RELU_S·arg | fp32 headroom | byte-exact |
|---------------------|------:|-------------:|--------------------:|--------------:|-----------:|
| nibble baseline     |  **10** |  **14 370** |    46 400 (0.28%)   |   **99.72%**  |  425/425   |
| nibble Dadda CSA    |    10 |     14 370  |    48 000 (0.29%)   |     99.71%    |  425/425   |
| byte carry-save RS=32 |   73 |     16 493  | 8 323 200 (49.61%)  |     50.39%    |  425/425   |
| byte carry-save RS=64 |   73 |     16 493  | 16 646 400 (99.22%) |    **0.78%**  |  425/425   |

(depth = #FFN blocks; weights = nonzero W_up/W_gate/W_down; tightest = the single
largest `RELU_S·arg` over all gadgets, as a fraction of 2²⁴.)

## The honest verdict

**Byte carry-save LOSES to the nibble baseline on both depth and weights.**

- **Depth: 73 vs 10 blocks — 7.3× deeper.** The "decompose the columns once"
  step is the killer: the MSB-first peel of a 260 100 column is inherently
  sequential (each nibble = digit → snap → subtract = 3 serial blocks, ×5 nibbles
  ×4 columns = **60 of the 73 blocks, 82%**). Larger partial products don't
  save work — they just move the decomposition cost to the back end, and peeling
  a wide column is more serial depth than peeling many small (≤225) nibble
  products in parallel.
- **Weights: 16 493 vs 14 370 nz — ~15% more.** The 8 byte-offset carry rounds
  dominate weights (12 672 of 16 493 nz); the base-16 ramp staircases are not
  cheaper than the baseline's, and there are more of them.
- **The baseline wins because its intermediate values are tiny.** Every nibble
  product ≤ 225 and every column < 256, so its staircase args are ≤ 232 →
  99.7% fp32 headroom AND no wide peel is ever needed. The byte design's whole
  premise (fewer, bigger products) forces the expensive wide-column peel that
  more than eats the saving.

**On the RELU_S margin: RS=64 is too fragile; RS=32 is comfortable.**

- RS=64 uses **99.22%** of the 2²⁴ ceiling — a 0.78% margin. That is real (it
  passes 425/425) but far too thin to trust: any upstream change that lets a
  column exceed 260 100, or any extra fp accumulation before the peel, tips it
  over the fp32 cliff. Do not ship RS=64.
- RS=32 uses **49.61%** of the ceiling — ~2× headroom (the column could nearly
  double before the peel became lossy) — and the snap/ramp discipline keeps it a
  bit-exact step. **RS=32 is the right operating point** if this design were
  used; the fp32 hard ceiling is `RELU_S < 64.5` (`_assert_headroom` reports it).

### Note on nibble Dadda

nibble Dadda ties the baseline exactly (depth 10, 14 370 nz) here. Dadda's
log-depth schedule (targets `[13, 9, 6, 4, 3, 2]`, 6 reduction stages + 1 CPA =
7 rounds) equals the baseline's 7 ripple rounds for this 8-column / max-height-8
case — Dadda's win only materialises at larger column heights (deeper adder
trees), which a 32-bit nibble-schoolbook does not reach. So on this problem it is
neither better nor worse.

**Bottom line:** the byte carry-save multiplier is *correct and fully fp32* at
RELU_S=32 with ~2× headroom, and the tunable-RELU_S tripwire does exactly what
the substrate fact predicts — but on this 32-bit problem it does **not** beat the
nibble schoolbook on either depth or weights. The nibble baseline's tiny
intermediate values (deep fp32 headroom, no wide peel) win. Bigger partial
products are a false economy here: they trade many cheap parallel splits for one
expensive serial wide-column decomposition.
