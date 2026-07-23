# Attention-select radix-16 (nibble digit-recurrence) 32-bit divide

`c4_min/div_radix16_attn.py` — a **measure-only bakeoff** that asks whether the
runtime radix-16 divide can be made **shallower** by moving its per-iteration
quotient-digit **SELECT** out of the SwiGLU FFN and into a **softmax1 attention
head**, keeping only the arithmetic (subtract / bring-down) as FFN.

Baseline: `div_radix16_hardened.py` — 88 blocks, **10 FFN blocks/iteration**. Of
those 10, three are the compare-SELECT:

| block | does |
|---|---|
| `shift` | `R' = 16·R + dividend_nibble` (bring-down) — FFN |
| `gteq` | per-nibble `GT[k,i]`, `EQ[k,i]` for R' vs each `KB[k]=k·b` — **SELECT** |
| `qdigit` | lexicographic suffix-AND `GE[k]=[R'≥KB[k]]`, `q = Σ GE[k]` — **SELECT** |
| `qbsel` | one-hot(q) select of the subtrahend `KB[q]` — **SELECT** |
| `gp, ks0..3, apply` | Kogge-Stone parallel-prefix borrow `R = R' − KB[q]` — FFN |

Self-contained: it **imports** the hardened divide's `shift`, `gp/ks/apply`
borrow, `KB`-precompute, `init`, `finalize`, `LeanDivBands` layout, and the
`_bake_memory_cam` sink / slow-RoPE-lane pattern + Qwen head geometry from
`qwen_full_vm`; it does **not** edit any shared file.

## The idea — the SELECT is a threshold retrieval

Per iteration we want

```
q = max{ k : KB[k] ≤ R' }     over the 15 precomputed KB[k] = k·b  (k = 1..15).
```

That is a **threshold retrieval**, which fits a monotone-scored softmax1 head.
Score key `k` (a `KB[k]` row) against the query `R'`:

```
score_k = ALPHA·k  −  PEN·[ KB[k] > R' ]          (k = 1..15)
```

with an implicit content-free **sink** row at logit 0 (softmax1 / ZFOD). Valid
keys (`KB[k] ≤ R'`) score `ALPHA·k` — a monotone, ALiBi-style positive position
bias over `k` — while invalid keys are pushed `PEN` below the sink. A sharp
`ALPHA` makes the softmax concentrate on the **largest valid** `k = q`; a
weighted-index readback `Σ_k w_k·k` recovers `q`, and the value rows copy back
`KB[q]`. If no `k` is valid (`q = 0`) the **sink wins** and the head returns 0 —
exactly the `q = 0` digit.

## Two variants — and the honest wall between them

### 1. `attn-mono` (fp64) — attention does the WHOLE select

One head replaces **all three** FFN SELECT blocks (`gteq + qdigit + qbsel`):

```
10 FFN blk/iter  →  7 FFN blk/iter + 1 attention head
total 88 blocks  →  64 FFN blocks + 8 attention heads   (shaved 3 FFN blk/iter = 24 blocks)
```

The head's q/k/v/o cost is tiny (**26 nz** on one Qwen head). **In fp64 it is
byte-exact** (the algorithm is correct — verified on the edge grid + adversarial
classes + ≥6000 random + the digit-boundary grid).

**But it is NOT byte-exact in fp32.** The head's threshold **gate** `[KB[k] > R']`
is a comparison of two **36-bit magnitudes** (`KB[k]` and `R'` are each up to
`16·b < 2³⁶`). A softmax score is a **dot product** — a linear form — and the
only linear way to encode the sign of `KB[k] − R'` is the **positional recompose**
`Σ_i 16^i·(KB[k][i] − R'[i])`, whose weights span `16⁸ = 2³²`. That **overflows
fp32's 2²⁴ exact-integer range** at exactly the digit boundaries `R' ≈ q·b` for
large `b`: fp32 rounds `R'` and `k·b` — differing by 1 near `2³⁶` — to the *same*
fp32 value, so the gate mis-fires. Measured: the raw fp32 value-diff gate is
**wrong on ~25 % of the boundary grid** (199 562 / 800 000).

This is the **same `16^p` amplification the hardened variant was founded to
kill**. A single attention dot **cannot** do the nibble-**lexicographic** compare
that dodges it — lexicographic ordering is not a fixed-weight linear form under
the `2²⁴` cap.

### 2. `attn-cam` (fp32) — attention does only the RETRIEVAL (the honest answer)

Keep the **compare** in fp32-exact FFN (the nibble-lexicographic `gteq + qdigit`,
which is *what produces the fp32-safe `q`*), and move **only** the
`one-hot(q) → KB[q]` **retrieval** (the `qbsel` block) into an **exact-match CAM
head** — the `_bake_memory_cam` per-bit-agreement pattern shrunk to a **4-bit
digit key** `q`. Matching bits ADD `+G²`, mismatching CANCEL `−G²`, a bias lane
sinks non-exact matches below the logit-0 sink; the matched row (`k == q`) copies
its `KB[k]` nibbles. A 4-bit exact match is **not** a wide compare — every score
term is a per-bit agreement in `[−G², +G²]`, fp32-exact — so **this head is
byte-exact in fp32**.

```
10 FFN blk/iter  →  9 FFN blk/iter + 1 attention head
total 88 blocks  →  80 FFN blocks + 8 attention heads   (shaved 1 FFN blk/iter = 8 blocks)
```

Head cost **36 nz** on one Qwen head.

## Measured (`python -m c4_min.div_radix16_attn`)

Grids: the hardened divide's edge grid + adversarial classes (`(2³²−1)//{2,3,7}`,
`2^k±1`, `b=1`, `a<b`, `a=b`, `k·b` boundaries, div-by-zero→(0,0)) + ≥6000
random, plus a dedicated **digit-boundary grid** (`R'≈q·b`, large `b`).

Measured (main grid = edge + adversarial + 6000 random = **8094** cases; boundary
grid = **4500** cases):

| variant | FFN blk/iter | total depth | head nz | fp64 main | fp32 main | fp64 boundary | fp32 boundary |
|---|---:|---|---:|:---:|:---:|:---:|:---:|
| hardened (FFN-only) | 10 | 88 blk | — | 100% | **100%** | 100% | 100% |
| `attn-mono` | 7 + 1 head | 64 FFN + 8 heads | 26 | **8094/8094** | 7976/8094 (**residue floor**) | 4500/4500 | **4248/4500** (wall) |
| `attn-cam` | 9 + 1 head | 80 FFN + 8 heads | 36 | 8094/8094 | **8094/8094** | 4500/4500 | **4500/4500** |

The `attn-mono` fp32 wall is sharpest at `a < b` / `a ≈ q·b` for large operands —
e.g. `(451 622 136, 451 622 137)` expects `q=0` but the value-diff dot returns
`q=1` (fp32 rounds `a` and `b`, differing by 1 near `2²⁹`, to the same value, so
the gate `KB[1]=b ≤ a` mis-fires). `attn-cam` is exact everywhere.

(The fp32 forward is a **batched** SwiGLU, whose fp32 accumulation order differs
from a single-row forward — a *stricter* robustness test, the same reordering
that flushed out the lean divide's residue floor. 0 fp64 params in the FFN
blocks; attention scores are in the sharp regime — 4-bit per-bit-agreement for
`attn-cam`. Run: `python -m c4_min.div_radix16_attn`.)

## Verdict — what attention CAN and CANNOT shave off the runtime divide

- **The SELECT algorithm IS a threshold retrieval** and fits a monotone softmax1
  head (`score_k = ALPHA·k − PEN·[KB[k]>R']`, sink at 0). In **fp64** it is
  byte-exact and replaces **three** FFN blocks (`gteq + qdigit + qbsel`) with
  **one** attention head: `10 → 7` FFN blk/iter (88 → 64 FFN + 8 heads).
- **The catch is fp32.** The head's threshold gate `[KB[k]>R']` is a 36-bit
  magnitude compare, and a softmax score is a linear dot → the only encoding is a
  `16^p` positional recompose spanning `2³²`, which **overflows fp32 (2²⁴)** at
  the digit boundaries. This is exactly the amplification the hardened variant
  eliminated; a single attention dot cannot do the nibble-lexicographic compare
  that avoids it. So **attention can do the whole select only in fp64** (or with
  the gate pre-computed in FFN).
- **The honest fp32 answer** (`attn-cam`): keep the compare (`gteq + qdigit`) in
  fp32-exact FFN and move **only** the `one-hot(q)→KB[q]` **retrieval** into an
  exact-match 4-bit-digit CAM head. Byte-exact in fp32, shaves **1 FFN
  block/iter** (`qbsel`): `10 → 9` (88 → 80 FFN + 8 heads).
- **The subtract and bring-down are irreducibly FFN** — attention does no
  arithmetic; the Kogge-Stone borrow and the `R' = 16R + nibble` shift are
  unchanged.

**Bottom line.** Attention shaves the *retrieval* of the SELECT cheaply and
exactly in fp32 (−8 blocks, 88 → 80); it can shave the *whole* SELECT (−24
blocks, 88 → 64) but only in fp64 — the sharp threshold **compare** at the heart
of the SELECT is exactly the wide-magnitude operation a linear attention score
cannot do in fp32 without the `16^p` overflow the FFN nibble-lexicographic
compare exists to avoid.
