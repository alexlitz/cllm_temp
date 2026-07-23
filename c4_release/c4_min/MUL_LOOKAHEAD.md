# Carry-lookahead (prefix) multiply — shrink the general MUL depth

`mul_lookahead.py` replaces the **7 sequential ripple carry rounds** in the general
32-bit multiply (`nibble_alu32.compile_mul_blocks`) with a **base-16 carry-lookahead
(parallel-prefix) resolve** that computes every column's carry in `⌈log₂ 8⌉ = 3`
prefix stages instead of 7 serial ripples. Standalone byte-exact bakeoff, **composed**
from the proven `nibble_alu32` SwiGLU primitives (`_nibble_carry_round`, `_step_ge`,
`_ident`, `_clear`, `_truncate`, `compile_mul_blocks`); it never edits shared files.

```
python -m c4_min.mul_lookahead
OMP_NUM_THREADS=4 PYTHONPATH=$(pwd) python -m pytest c4_min/test_mul_lookahead.py -v
```

## The baseline and the lever

`compile_mul_blocks` = **10 blocks**: `products (1) | split (1) | 7× ripple carry
round | result copy (1)`. The 36 nibble partial products `a_i·b_j` (i+j<8) are split
into low/high nibbles and accumulated into 8 nibble columns `MCOL[c] < 256`. Each
ripple round is `dst[c] = src[c] mod 16 + floor(src[c-1]/16)` — a carry moves **one
column per round**, so 7 (+1 headroom) serial rounds settle an 8-column stack. The
ripple **dominates the depth** (7 of 10 blocks); products + split + result are only 3.

The post-split columns are a base-16 **carry-save** number, so the resolve is a
carry-propagate add over 8 columns — a textbook target for a parallel-prefix
(Kogge-Stone) carry-lookahead.

## The design (Kogge-Stone, byte-exact)

The trick that makes the prefix a **clean binary** carry-lookahead is one round-1:

1. **round-1** (one library carry round) reduces the columns from `< 256` to
   `t_c = (MCOL[c] mod 16) + floor(MCOL[c-1]/16) ∈ [0, 30]` — a digit 0..15 plus a
   single-nibble carry 0..15 from below. Now, given an **incoming binary carry**
   `b_c ∈ {0,1}`, each column emits final digit `(t_c + b_c) mod 16` and a **binary**
   carry-out `[t_c + b_c ≥ 16] ∈ {0,1}` (since `t_c + b_c ≤ 31`). Carries are pure
   binary → textbook CLA.
2. **G/P**: `G_c = [t_c ≥ 16]` (generate), `P_c = [t_c == 15]` (propagate). Both 0/1.
3. **3 Kogge-Stone prefix stages** (distances 1, 2, 4) combine the pairs with the
   associative operator `(g,p)∘(g',p') = (g ∨ (p ∧ g'), p ∧ p')`. Over these 0/1 lanes
   the combine is one staircase each: `G_dst = [2·G_c + P_c + G_{c-d} ≥ 2]` (an exact
   single-level `G ∨ (P ∧ G_{c-d})`) and `P_dst = [P_c + P_{c-d} ≥ 2]`. After the
   stages `Gband[c]` is the carry **into** column `c+1`.
4. **fused apply/result** writes the result nibble directly into `MUL_RES`
   (no separate copy): `digit_c = t_c + b_c − 16·b_{c+1} = t_c + Gband[c−1] − 16·Gband[c]`
   (`b_0 = 0`; the top column's carry-out overflows the kept nibbles and is dropped,
   exactly as the baseline dropped it → `& 0xFFFFFFFF`).

Depth: `products (1) | split (1) | round-1 (1) | G/P (1) | KS×3 | apply/result (1)` =
**8 blocks**, **3 prefix stages** (vs 7 ripple rounds).

## Results (byte-exact = every operand, incl `0xFFFFFFFF²`)

| variant | depth | nz | prefix / rounds | tightest `RELU_S·arg` | `0xFFFFFFFF²` | byte-exact |
|---------|------:|------:|:---:|---:|:---:|:---:|
| baseline_ripple       | **10** | **11 430** | 7 rounds | 46 400 (0.277 % of 2²⁴) | ✓ | 20685/20685 |
| **kogge_stone_lookahead** | **8** | **6 291** | **3 stages** | 48 000 (0.286 % of 2²⁴) | ✓ | 20685/20685 |
| stride2_partial       | 13 | 14 364 | (wash) | 48 000 (0.286 %) | ✓ | 20685/20685 |

Verified byte-exact over the shared edge set `{0, 1, 2, 2³¹, 2³²−1, every power of two
2⁰..2³¹, single-byte lanes, full-width}` × a multiplier fan, the literal worst case
**`0xFFFFFFFF²`** (max carry propagation — the whole point of the carry path), plus
**20 000 random** `(a,b) < 2³²` — **20 685 / 20 685 byte-exact** for both.

- **Depth 10 → 8**, **3 prefix stages replace 7 serial ripples**, **nz nearly halved
  (11 430 → 6 291)** — the ripple carry rounds were the bulk of the multiply's weight
  as well as its depth.
- **fp32 discipline**: every relu argument is trivially exact — post-split columns
  `< 256`, round-1 outputs `t_c ≤ 30`, and the entire prefix runs on {0,1}
  generate/propagate lanes (arguments ≤ 4). The tightest `RELU_S·arg = 48 000` is the
  round-1 floor staircase, **0.286 % of the 2²⁴ ceiling** — oceans of headroom. **0 fp64.**

### Honest verdict on 10 → ~6

The clean Kogge-Stone lands at **8 blocks**, not 6. The front-end (products + split)
is 2 irreducible blocks that must be kept; the resolve is 6 blocks (round-1, G/P, 3
prefix stages, fused apply). The three prefix stages are the theoretical minimum for
8 columns (`⌈log₂ 8⌉`), and round-1/G/P/apply cannot be folded into each other because
every SwiGLU unit reads only the **block input** — a materialised intermediate forces a
block boundary. So **8 is the honest floor for a byte-exact binary-prefix resolve**;
the "≈6" target would require a sub-block-boundary fusion the substrate does not allow.
The win is real and clean: **shallower, 3 stages not 7, and about half the weight**.

The `stride2_partial` variant (resolve two columns per super-round via two ripple
substeps) is included for the brief's "measure whatever lands" honesty: it lands at
**13 blocks — worse than baseline** — because each super-round still needs 2 blocks (the
first ripple must materialise the intermediate before the second can read it). This
proves the **prefix** structure, not stride, is the real lever.

## Div knock-on (quantified, not inflated)

The base-16 long division (`compile_divmod_blocks`) runs 8 iterations; each carry-
normalises `QB = q·b` with `_QB_CARRY_ROUNDS = 6` ripple rounds over `RN = 9` columns.
A Kogge-Stone resolve of `QB` would use `⌈log₂ 9⌉ = 4` prefix stages + round-1 + G/P +
apply = **7 resolve blocks vs the 6 ripple rounds** — a **wash-to-slight-loss for QB
alone**, because QB's ripple is already only 6 rounds over 9 columns (the multiply win
is real precisely because the mul ripple is 7 rounds over 8 columns, where 3 prefix
stages genuinely beat it).

The honest div reuse is therefore the **primitive** — the same `_nibble_carry_round`
and the new G/P-prefix gadget — not a per-iteration block-count win. The larger latent
div lever is the KB-precompute (15 × `_KB_CARRY_ROUNDS = 6` = 90 ripple blocks, one per
`KB[k] = k·b`), where a **shared/batched** prefix resolve across the 15 independent
9-column ripples could shave real depth — out of scope for this standalone multiply
bakeoff, flagged for follow-up.
