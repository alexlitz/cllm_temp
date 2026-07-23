# Multiplier design bakeoff — depth · weights · precision · byte-exactness

`mul_bakeoff.py` builds seven `(a*b) & 0xFFFFFFFF` gadget designs (a, b are two
8-nibble / 32-bit operands seeded into the `STACK0` and `AX` nibble bands) as
SwiGLU FFN-block lists and measures each on four axes:

| axis | how |
|------|-----|
| **DEPTH** | `len(block list)` — the number of sequential SwiGLU sub-blocks. Cheap, no forward. |
| **WEIGHTS** | sum of non-zero `W_up`/`W_gate`/`W_down` entries across all blocks. Cheap, no forward. |
| **PRECISION** | does any extraction feed a `RELU_S·threshold` bias past the working dtype's integer-exactness ceiling? (fp32 → `2^24`.) |
| **BYTE-EXACT** | a **sparse** arithmetic simulation of `x ← x + W_down·(silu(W_up·x+b)·(W_gate·x+b))` over just the bands each gadget reads/writes (a `dict` band→value), applying `down(silu(gate)·up)` only over the non-zero units, on ~160 `(a,b)` pairs. No dense `DIM` forward — runs in **~10 s total**. |

Everything is **composed** from the proven `nibble_alu32` primitives (`_mul_gate`,
`_floor_div_pow`/`_floor_div_pow2`, `_empty_spec`, `RELU_S`, `S`, `_ident`/`_clear`/
`_relu`/`_step_ge`/`_truncate`); this file never edits them. Run:

```
python -m c4_min.mul_bakeoff
```

## Results (byte-exact = 160/160 for every variant, sorted by depth then weights)

| # | variant | depth | weights (nz) | precision | fp32 by-arg? | byte-exact | fp32 spot |
|---|---------|------:|-------------:|-----------|:---:|:---:|:---:|
| 2 | nibble + carry-share | **10** | **11 430** | fp32 | ✓ | 160/160 | 40/40 |
| 3 | nibble Dadda CSA     | **10** | **11 430** | fp32 | ✓ | 160/160 | 40/40 |
| 4 | bit-level Dadda      | **10** | 12 075 | fp32 | ✓ | 160/160 | 40/40 |
| 1 | nibble baseline      | **10** | 14 370 | fp32 | ✓ | 160/160 | 40/40 |
| 7 | 16-bit chunk         | 39 | 44 535 | **fp64** | ✗ | 160/160 | 18/40 |
| 5 | byte-chunk           | 62 | 57 938 | **fp64** | (values ✓) | 160/160 | 40/40 |
| 6 | 16×nibble            | 107 | 107 841 | **fp64** | ✗ | 160/160 | 24/40 |

(`weights` count excludes biases; `fp32 spot` = pass count on the first 40 operands
run forced-fp32. For a fp64 variant a *low* fp32-spot count is the point — it fails
fp32 on the large operands, proving the wall.)

## The MSB-first extraction primitive and the honest fp32 wall

The brief's hypothesis was that **MSB-first subtractive extraction** — peel a wide
value into nibbles from the *top* power, so each digit is `floor(x / 2^p) ∈ 0..15`
(**kmax bounded, independent of `x`**), then `x -= n·2^p` — would let the wide-chunk
designs (byte / 16×nibble / 16-bit-chunk) stay fp32.

**It does not, with this project's `RELU_S = 200` SwiGLU primitive.** MSB-first
does fix the tripwire *count* (base-256 peel: 255 per byte + 15 per nibble,
independent of magnitude), but it does **not** fix the argument *magnitude*, and
the wall is set by the weight-tensor precision, not the runtime dtype:

* Each threshold `t` bakes a bias `RELU_S · t` into `b_up`. The weight tensors are
  the model's native **fp32**, so `RELU_S · t` must itself be fp32-integer-exact:
  `RELU_S · t < 2^24` ⟹ `t < 2^24 / 200 ≈ 83 886 ≈ 2^16.4`. Above that the *weight*
  is quantised, **regardless of runtime dtype**.
* A second, subtler fp32 loss: the residue-subtract routes `−m·floor(x/m)` through
  `W_down` on hidden values of magnitude `RELU_S · x`. Summing 255 such terms
  telescopes at the `~4·10^6` scale where fp32's spacing is `~0.5`, so a fraction
  leaks and a nibble mis-rounds on ~1/256 values.

The `_verify_peel_roundtrip` decompose→recompose confirms the wall exactly (40 random
values per width, per spec-tensor dtype):

| width | fp32 spec | fp64 spec | why |
|-------|:---:|:---:|-----|
| **W=16** (byte product ≤ 65 025) | 40/40 | 40/40 | top-byte threshold ≤ 65 280 < 83 886 |
| W=20 (16×nibble partial ≤ 983 025) | 35/40 | 40/40 | top-byte threshold ~15·2^16 ⟹ `RELU_S·t = 2·10^8 > 2^24` |
| W=24 | 4/40 | 40/40 | |
| W=32 (16-bit-chunk product) | 0/40 | 40/40 | product itself > `2^24` |

So MSB-first extraction **only** stays fp32 for values whose largest peel threshold
is below ~83 886 — i.e. the nibble-granular designs (staircase arg ≤ 232) — and
needs **fp64 weight tensors** for anything wider. The wide-chunk variants (5–7) are
therefore baked in fp64 and marked so.

### Per-variant notes

1. **nibble baseline** (`compile_mul_blocks`) — 36 nibble partials `a_i·b_j ≤ 225`,
   split into columns kept `< 256` (kmax=15), 7 carry rounds. Every staircase
   argument ≤ 232 ⟹ fully **fp32**. 14 370 nz / 10 blocks (confirms the ~14 370 nz
   expectation).
2. **nibble + carry-share** — same products/split front-end; the carry round peels
   `floor(col/16)` **once** and routes it to both mod (`×−16`) and carry (`×+1`) via
   one `_floor_div_pow2` staircase → **−2 940 nz** vs baseline (11 430), same depth,
   fully **fp32**. Best-weight tie.
3. **nibble Dadda carry-save** — the split leaves column heights `[1,3,5,7,9,11,13,15]`
   (low+high nibble contributions), max 15. Dadda schedule targets `[13,9,6,4,3,2]`
   ⟹ **6 reduction stages + 1 CPA = 7 rounds**. With value-carrying nibble wires each
   3:2 compressor row *is* one base-16 carry round, so the Dadda structuring caps the
   round count; identical 11 430 nz / 10 blocks to carry-share here (both use the
   shared staircase over 8 columns). Fully **fp32**.
4. **bit-level Dadda** — 528 one-bit `AND` partials (`i+j<32`), classic Dadda tree
   census: **240 full adders, 0 half adders, 8 reduction stages, 31-bit CPA
   (271 adders total)**. 1-bit wires ⟹ no staircase at all. The built verifiable
   slice (all 528 ANDs → nibble columns → carry settle) is byte-exact and **fp32**.
5. **byte-chunk** — 4 byte chunks → 10 byte products (`8×8=16-bit`, ≤ 65 025), each
   split MSB-first (base-256) into nibbles, accumulate, carry-settle. **The product
   VALUES fit fp32** (65 025 < 2^24) — but the base-256 peel's `−256·floor` down-sum
   cancels at `~4·10^6`, so ~1/256 products (e.g. `0x3FFF`) mis-round a nibble in
   fp32; the peel needs **fp64**. MSB-first does *not* keep this design fp32.
6. **16×nibble** — `a` in two 16-bit halves; partial = `nibble × 16-bit half` =
   20-bit (≤ 983 025). Top-byte threshold `~15·2^16` ⟹ `RELU_S·t = 2·10^8 > 2^24`
   ⟹ **fp64** mandatory. Heaviest (107 blocks, 107 841 nz): the 13 partials each need
   a full 3-byte base-256 peel.
7. **16-bit chunk** — 3 products `a_lo·b_lo`, `a_hi·b_lo`, `a_lo·b_hi` (`16×16 =
   32-bit`, > 2^24). The **product itself** exceeds fp32, so `_mul_gate` forms it as
   4 `(nibble × 16-bit)` fp64 partials and the 32-bit peel is fp64 too — **fp64**
   throughout (product AND extraction). Only 39 blocks / 44 535 nz — the leanest of
   the fp64 group because it forms only 3 wide products.

## Verdict

* **Best DEPTH** — the four **nibble-granular** designs (baseline, carry-share,
  Dadda-CSA, bit-Dadda) all tie at **10 blocks**. The wide-chunk designs are
  *deeper*, not shallower, once the MSB-first peel is spread across the blocks it
  needs (the peel is inherently sequential — each byte reads the residue the previous
  byte shrank, so it cannot be one block): byte-chunk 62, 16-bit-chunk 39,
  16×nibble 107.
* **Best WEIGHT** — **nibble + carry-share** and **nibble Dadda** tie at **11 430 nz**
  (−20 % vs the baseline's 14 370, from the shared mod/carry staircase). Every
  wide-chunk design is 4–9× *heavier* (44 k–108 k) because the base-256 kmax=255
  staircases dominate.
* **Fully fp32-vanilla** — **only the four nibble-granular designs** (1–4). They
  never let a staircase argument exceed 232, so `RELU_S·arg < 2^24` everywhere and
  the fp32 weight tensors are exact.

### Headline answer

**No.** With MSB-first extraction the wide-chunk designs (byte / 16×nibble /
16-bit-chunk) do **not** beat nibble schoolbook — they are both **deeper** and
**heavier**, and they do **not** stay fp32: the `RELU_S·threshold` weight bias (and
the residue-subtract down-sum) blows past fp32's `2^24` integer ceiling for any peel
threshold above ~83 886, which every wide chunk crosses. Only the nibble-granular
designs — which keep every staircase argument `< 256` — stay fully fp32-vanilla, and
of those the **carry-sharing** and **Dadda** variants are the best on weights (11 430)
at the same minimal depth (10). MSB-first extraction is a genuine *win for tripwire
count* (kmax 15/255 instead of thousands), but with this SwiGLU multiplier primitive
the fp32/fp64 boundary is set by the *value magnitude an extraction sees*, and the
wide chunks unavoidably see it.
