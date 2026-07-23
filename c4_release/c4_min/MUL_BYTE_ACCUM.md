# MUL_BYTE_ACCUM — nibble products, BYTE-WIDE column accumulation

Head-to-head test of the **"carry on a larger representation"** idea done
*without* the wide-product penalty: keep the CHEAP fp32-safe nibble products the
production baseline already uses, but **accumulate them into 4 BYTE-position
columns** (weight `256^c`) instead of 8 nibble columns — then peel each byte
column back to nibbles MSB-first, in PARALLEL across the 4 columns.

Built as persistent SwiGLU FFN blocks in
[`mul_byte_accum.py`](mul_byte_accum.py), composed from the proven
`nibble_alu32` primitives (`_mul_gate`, `_empty_spec`, `_floor_div_pow`,
`_ident`, `_clear`, `_truncate`, `S`, `RELU_S`). It edits none of them and does
not touch `mul_bakeoff*.py` / `mul_byte_carrysave.py` (owned by other agents).

Run it:

```
python -m c4_min.mul_byte_accum
```

## The design (nibble products, byte-column carry-save)

1. **36 nibble products** `a_i·b_j` (`i+j<8`), each `≤ 15·15 = 225` — the exact
   same cheap products the nibble schoolbook baseline uses. The `_mul_gate`
   internal `silu(S·a)·b ≈ S·a·b ≤ 200·225 = 45,000 ≪ 2²⁴` → deeply fp32-exact.

2. **Accumulate into 4 BYTE-position columns** (weight `256^c`, c=0..3), **NO**
   per-product split. A nibble product at nibble-position `s = i+j` lands in byte
   column `c = s÷2`, scaled by `16` when `s` is odd (since
   `16^s = 256^(s//2)·16^(s%2)`):

   > **byte-col c = Σ_{i+j=2c} p + 16·Σ_{i+j=2c+1} p**

   **Exact byte-column max** (all nibbles = 15, so every product = 225):

   | byte column c | products at s=2c (w=1) | products at s=2c+1 (w=16) | **colmax** |
   |:---:|:---:|:---:|:---:|
   | 0 | 1 | 2 | 7,425  |
   | 1 | 3 | 4 | 15,075 |
   | 2 | 5 | 6 | 22,725 |
   | 3 | 7 | 8 | **30,375** |

   Overall max byte-column = **30,375**. At the **default RELU_S=200**:
   `200 × 30,375 = 6,075,000` = **36.21 % of 2²⁴** (headroom 63.79 %). So **no**
   low-RELU_S, **no** residue-snap machinery is needed — this is the whole
   advantage over byte-PRODUCT accumulation (`mul_byte_carrysave.py`), whose
   260,100 columns forced RELU_S ≤ 64 and a fragile per-digit snap. The
   *tightest baked staircase threshold* is even smaller — 28,672 (the top-nibble
   `kmax·16³`, tightly capped), 34.18 % of 2²⁴.

3. **Peel each byte column to nibbles, MSB-first, IN PARALLEL** across the 4
   columns (they are independent until the final combine, so the 4 peels share
   block depth — one block does all 4 columns per stage, not 4 serial
   per-column peels). Each column is `≤ 30,375 < 2¹⁵` → 4 nibbles. Every peel
   argument is `≤ 30,375`, so the library RELU_S=200 base-16 `_floor_div_pow`
   (kmax=15) is bit-exact. Because the byte columns are EXACT integer sums of
   exact `_mul_gate` integers (no fp residue), **no per-digit snap is needed**.

4. **Combine the 4 columns at their byte offsets** (column c → result nibbles
   2c..2c+3) into 9 raw nibble lanes, then a base-16 carry ripple settles them
   into the final 8 result nibbles.

## Results (425 operands: 0, 0xFFFFFFFF, 0xDEADBEEF, powers of two, single-byte, full-width, + 180 random)

| VARIANT            | DEPTH | WEIGHTS (nz) | TIGHTEST RELU_S·arg | fp32 HEADROOM | BYTE-EXACT |
|:-------------------|:-----:|:------------:|:-------------------:|:-------------:|:----------:|
| nibble baseline    | **10**  | 14,370       | 46,400 (0.28 %)     | 99.72 %       | 425/425    |
| **byte_accum**     | 17    | **11,520**   | 5,734,400 (34.18 %) | 65.82 %       | 425/425    |

*(nibble baseline = the production `compile_mul_blocks`: 36 nibble products, 8
nibble columns < 256, 7 carry rounds; the `14,370 nz / 10 blocks` reference.)*

### byte_accum block breakdown (depth 17)

```
products (1) + accum (1)              = 2   cheap nibble products, byte-column accumulate
peel: 4 digit + 3 residue            = 7   MSB-first, 4 columns PARALLEL per block
combine (1)                          = 1   gather column nibbles at byte offsets
carry ripple (6)                     = 6   settle the 9-lane combine
result copy (1)                      = 1
                                       ─
                                       17
```

## Honest verdict

**Byte-column accumulation of cheap nibble products WINS on weights and fp32
discipline, but LOSES on depth.**

- **Weights: WIN.** 11,520 vs 14,370 nz (**−20 %**). Accumulating into 4 byte
  columns skips the baseline's per-product low/high nibble *split* (a
  `_floor_div_pow2` staircase on all 36 products) and needs fewer, narrower
  carry rounds — a real weight saving.

- **fp32: COMFORTABLE at RELU_S=200 — the design goal, achieved.** Tightest
  baked arg 28,672 = 34.18 % of 2²⁴ (column value 30,375 = 36.21 %). ~63 %
  headroom, **no fragile margin**, and it runs at the library's default
  RELU_S=200 — a strict improvement over the byte-PRODUCT attempt
  (`mul_byte_carrysave.py`), which used 99.22 % of the ceiling at RELU_S=64.
  This confirms the task's premise: cheap nibble products keep the columns
  small (30,375 vs the byte-product 260,100), so byte-wide accumulation costs
  no fp32 headroom.

- **Depth: LOSS (17 vs 10).** This is the honest breakdown of *why* the "4
  byte-columns vs 8 nibble-columns" argument does **not** buy fewer carry
  rounds overall:

  1. The **peel costs 7 blocks** (4 digit stages + 3 residue subtracts). The
     baseline *never peels* — its columns are already nibble-width, so it goes
     straight from products → split → carry. You pay to peel the wide byte
     columns back down to nibbles.
  2. The **byte-offset combine reintroduces a nibble ripple.** After peeling,
     the 4 columns' nibbles land at overlapping byte offsets, producing 9 raw
     nibble lanes that carry like any nibble column (a carry ripples ONE lane
     per round). The literal max product `0xFFFFFFFF²` (which yields the
     genuine worst-case all-15s columns, so no input rips longer) needs 5
     rounds (max chain `[16,15,15,15,15]`); we keep 6 (+1 headroom). So the
     post-combine ripple is ~as long as the baseline's carry tail.

  Net: the byte-column accumulation shortens the *pre-peel* carry to zero (4
  columns, no rounds — vs the baseline's 8 columns needing 7 rounds), but that
  saving (~7 blocks) is more than eaten by the peel (7 blocks) **plus** the
  post-combine ripple (6 blocks) the byte offsets reintroduce. The peel + combine
  tax is the fundamental cost of carrying on a larger representation and then
  having to decompose it back to nibbles for the byte-exact result.

**Bottom line:** the design is correct (425/425), sits comfortably in fp32 at
RELU_S=200 (the key claim), and is **20 % lighter in weights** — but it is
**1.7× deeper** than the nibble schoolbook. If the metric that matters is
weight count / fp32 safety, byte-column accumulation of cheap nibble products
is a genuine win; if the metric is block depth, the nibble schoolbook baseline
still wins, because the byte columns must be peeled back to nibbles (7 blocks)
and then re-carried after the byte-offset combine (6 blocks).
