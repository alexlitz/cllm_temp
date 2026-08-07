# Minimal 10-digit-addition transformer — measured results

A hand-constructed (NOT trained) transformer that adds two 10-digit integers to
an 11-digit result, **byte/digit-exact**, following the construction in Alex
Litzenberger's blogpost *"Building a Minimal Transformer for 10-digit
Addition"*. Runs in **float64**. Standalone artifact — unrelated to the c4 VM.

Files:
- `minimal_10digit_adder.py` — the model + parameter census (`MinimalAdder`).
- `verify_minimal_adder.py` — exactness + precision-boundary harness.

Run: `python verify_minimal_adder.py` (CPU only).

## The construction (from the blogpost)

- **Embedding is NOT one-hot.** `d_model = 4`: dim 0 carries the digit *face
  value* (0..9); dims 1/2/3 are flag bits for BOS / `+` / `=`.
- **ALiBi place-value head, slope = ln(10).** One relative-position step scales
  the attention logit by `ln(10)`, so `e^{logit}` is a *power of ten*: the
  most-significant digit gets weight `1e9` down to `1e0` for the least — a
  descending place-value ladder. BOS/`+`/`=` get a large negative bias so they
  drop out (the "reset" at `+`/`=`).
- **softmax1** = `e^{x_i} / (1 + Σ_j e^{x_j})` (the off-by-one softmax). The
  weighted pool is a *mean*; multiplying back by the softmax1 denominator (the
  effective count N) turns the mean into a **sum**, so attention *adds* the two
  operands by place value: `S = a + b` exactly.
- **Digit selection by negative absolute difference.** Output digits are read
  MSB-first: for candidate `d`, `logit_d = -|value - (d + 0.5)|`. The `+0.5`
  recentres each integer bin so argmax picks `floor(value)`. A running remainder
  is reduced by `d · 10^p` after each place. (Exact-integer values sit on a bin
  boundary — a tie whose correct resolution is the floor; a vanishing monotone
  tie-break selects the larger `d` without ever flipping a genuine decision.)

Format: `BOS a[10, MSB-first] + b[10] = c[11, MSB-first]`, zero-padded, e.g.
`7650676663 + 0149460439 = 07800137102`.

Hyperparameters: **d_model = 4, n_layers = 1, n_heads = 1, vocab = 13**
(digits 0–9, `+`, `=`, BOS).

### Deviation from the post
The post is prose and underspecifies the exact remainder-update mechanics and
the exact model dims. This implementation reproduces the same *ideas*
(place-value ALiBi pooling + softmax1 mean→sum + `-|value-(d+0.5)|` floor
selection) as a concrete, verifiable forward pass, and is deliberately **leaner**
(d_model=4, 1 layer, 1 head) than the author's ~95-param variant — hence the
measured counts below come out under the blogpost's headline figures.

## Exactness (float64)

**100,016 / 100,016 exact — 0 errors.** (100,000 random 10-digit pairs + 16
hard cases: `9999999999+9999999999`, full carry cascades, zeros, `a+0`,
`max+max`, every-column-carries, etc.) Every one of the 11 output digits matches
by argmax.

## Parameter counts (measured) vs the blogpost

| scheme | what it counts | measured | blogpost |
|--------|----------------|---------:|---------:|
| (a) | ALL non-zero dense entries | **42** | ~95 |
| (b) | (a) minus identity matrices | **26** | ~36 |
| (c) | (b) reusing the embedding value axis as decode candidates | **16** | ~28 |
| (d) | scalars only (no embedding table, no identities) | **4** | ~12 |

Per-scheme breakdown (measured):

- **(a) = 42**: 12 embedding non-zeros (9 digit values `1..9` + 3 flags) + 16
  attention Q/K/V/O identity diagonals (4 × d_model) + 10 candidate digits
  (`0..9`) + `alibi_slope=ln(10)` + `softmax1_const=+1` + `half_shift=+0.5` +
  `place_base=10.0`.
- **(b) = 26**: drop the 16 identity-projection diagonals (structural routing).
- **(c) = 16**: also drop the 10 stored candidate digits — reuse the embedding's
  value axis (`0..9` already live there) as the decode candidates.
- **(d) = 4**: the true learned scalars only — **`ln(10)` (ALiBi slope), `+1`
  (softmax1 off-by-one), `+0.5` (floor shift), `10.0` (place base)** — dropping
  the embedding table and all identities.

The four scalars in (d) are the irreducible core: the place-value slope, the
mean→sum off-by-one, the floor-realizing half-shift, and the base.

**Honest delta:** every measured scheme lands *below* the blogpost's headline
(42 vs 95, 26 vs 36, 16 vs 28, 4 vs 12). This model is a strictly leaner
realization (d_model=4 / 1 layer / 1 head vs the author's wider ~2-layer,
d≈5 variant), so it carries fewer identity-matrix and embedding entries. The
*mechanism* is identical; the *count* is smaller because the network is smaller.

## Precision boundary (fp32 vs fp64)

Isolating the place-value **sum** `S` (the arithmetic core, before decode):

| operand width (digits) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| fp32 sum errors / 20k | 0 | 0 | 0 | 0 | 0 | 10789 | 19758 | 19919 | 19962 | 20000 |
| fp64 sum errors / 20k | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

- **fp32 first produces a wrong sum at operand width 6** and is essentially
  always wrong by width 7 — exactly the `2^24 = 16,777,216` mantissa limit
  (~7–8 decimal digits): the sum of two 7-digit numbers reaches ~2·10^7 > 2^24,
  past which fp32 cannot represent consecutive integers.
- **fp64 is exact for the full sum at every width** (`2^52 ≈ 4.5·10^15`,
  ~15–16 decimal digits).

The full end-to-end fp32 model is worse still (unusable even at small widths):
the MSB-first floor decode needs sub-`1e-9` resolution on values up to ~`1e10`
(~19 significant digits), far beyond fp32's ~7. **fp64 is required** — and it
holds the entire 11-digit sum in a single float.

## Contrast with the c4 VM

This adder keeps the **whole sum in one fp64 value** and reads digits out of it,
which is why it needs double precision (10 digits > fp32's ~7). The c4 neural VM
instead works in **fp32 by decomposing every value into 4-bit nibbles**, so each
lane stays tiny and exact without ever needing a >7-digit float — the opposite
precision strategy for the same "exact integer arithmetic in a transformer" goal.
