# Hardened radix-16 (nibble digit-recurrence) 32-bit divide

`c4_min/div_radix16_hardened.py` — the **fp32-byte-exact hardening** of
`div_radix16_lean.py`. Same *robust, always-exact* algorithm (base-16 long
division, **NO reciprocal**, **NO per-digit multiply**), re-founded so the
running remainder `R` never leaves nibble-lane form → the fp32 residue floor is
gone.

Self-contained: it **imports** primitives from `nibble_alu32`
(`_ident/_clear/_step_ge/_guard/_empty_spec/_truncate`, `S`, `RELU_S`) and
reuses the proven **Kogge-Stone parallel-prefix** pattern from the general MUL
carry resolve there (adapted carry→borrow), but does **not** edit
`nibble_alu32.py`, `nibble_vm.py`, `qwen_full_vm.py`, `nibble_logsink*.py` or
`const_*`.

## What was wrong with the lean variant

`div_radix16_lean` packs the `R − KB[q]` subtract into **3 limbs of 3 nibbles**,
forming each limb VALUE by a `16^j` positional recompose
(`R_limb = R[0] + 16·R[1] + 256·R[2]`, up to 4095), doing a 3-limb borrow, then
re-splitting the limb value back into nibbles with a `floor(RV/16, kmax=255)` /
floor256 staircase. In genuine fp32:

- the `16^j` recompose **amplifies** the tiny `_step_ge` ramp residue on a
  nibble by up to **256×** (the `16²` weight), and
- the wide `floor(RV/16, kmax=255)` staircase reads that amplified residue — near
  a limb boundary the remainder nibble lands a fraction off (`3.4999` / `4.0001`)
  and mis-rounds.

On the full 32-bit adversarial grid (large divisors, `b` near 2³²) this is **not
a 1/3000 floor** — it fails the vast majority of large-divisor cases (measured
**690 / 9692** at BS=512), and is so marginal it flips under a mere change of
fp32 matmul accumulation ORDER (batched-vs-single forward). The fp64 ALU sim is
byte-exact everywhere (**9692/9692** — the algorithm is correct); only the fp32
substrate residue misfires. Measured **max per-step R residue = 5.0e-1** (a full
half-nibble).

## The fix — nibble-lane Kogge-Stone borrow (NO `16^p` recompose)

The whole `16^p` limb machinery (`borrow0, borrow1, subval, split1, split2`) is
replaced by a **base-16 parallel-prefix BORROW** over the 9 remainder nibbles:

| block        | does                                                                  |
|--------------|-----------------------------------------------------------------------|
| `gp`         | per nibble `diff_i = R[i] − KB[q][i]` in [−15,15]: seed the prefix buffer with `g_i = [diff_i < 0]` (generate) and `p_i = [diff_i == 0]` (propagate) — both 0/1 |
| `ks0..ks3`   | ceil(log₂ 9) = **4** log-depth prefix combine stages (double-buffered) settle the per-nibble borrow-OUT `Bout_i = g_i OR (p_i AND Bout_{i−1})` |
| `apply`      | `R[i] ← diff_i − Bin_i + 16·Bout_i` (`Bin_i = Bout_{i−1}`, `Bin_0 = 0`), a value in [−16,15] SNAPPED to [0,15] by a sharp `_step_ge` — fused with emit-q + IT++ |

**Every quantity in the whole borrow stays in [−16,15]** — there is no scalar
`16^p` recompose anywhere, so there is nothing to amplify. The prefix lanes are
pure 0/1 (sharp AND/OR via `_step_ge` at integer thresholds), residue-free by
construction; the result nibble is re-snapped by a sharp staircase every
iteration. This is the exact residue-free pattern the general MUL carry resolve
uses (`nibble_alu32._mul_gp_block` / `_mul_ks_stage_block`), carry→borrow.

## Blocks per iteration = 10 (was 9)

`shift, gteq, qdigit, qbsel, gp, ks0, ks1, ks2, ks3, apply(+emit q)`

The lean variant's 3-limb borrow (`borrow0, borrow1, subval, split1, split2` = 5
blocks) becomes `gp` + 4 prefix stages + fused `apply` = **6 blocks**: net **+1
block/iter, +8 blocks total** — the fp32-robustness cost.

## Depth

```
1 (kb-raw) + 4 (fused kb-carry) + 1 (bz) + 1 (init) + 8×10 (iters) + 1 (finalize)
= 88 blocks   (unrolled straight depth; recurrent stores 18 unique blocks)
```

## Measured (`python -m c4_min.div_radix16_hardened`)

| metric                | lean            | **hardened**                                  |
|-----------------------|-----------------|-----------------------------------------------|
| depth (unrolled)      | 80              | **88 blocks** (+8, the fp32 cost)             |
| blocks / iteration    | 9               | **10**                                        |
| stored unique blocks  | 17              | 18 (recurrent)                                |
| nz (nonzero weights)  | ≈144k           | **≈137.4k** (the wide 255-staircase is gone)  |
| max relu arg          | 1.47e7          | **50 900** (no `16^p` recompose) — < 2²⁴      |
| fp64 params           | 0               | **0**                                         |
| max per-step R residue| **5.0e-1**      | **≈4.1e-4** (~1200× under the ½-nibble margin)|
| byte-exact (fp64)     | 9692/9692       | **8094/8094** (measure grid) — algorithm OK   |
| byte-exact (fp32 e2e) | ≈690/9692       | **100%**: measure 8094/8094; +8000 random +   |
|                       |  (residue floor)| 6000 random (2nd seed) + all adversarial      |
|                       |                 | classes; the 21 prev-failing cases now pass   |

The fp32 forward is validated through a **batched** CPU SwiGLU forward, whose
fp32 accumulation ORDER differs from a single-row forward — so passing there is a
**stricter** robustness test than the production single-token path (where the
lean variant's marginal residue could flip under reordering).

**fp32 discipline.** Every relu/silu argument stays well < 2²⁴: the lexicographic
nibble compares keep every compared quantity in [−15,15]; the borrow `diff` is in
[−16,15]; the prefix lanes are 0/1; the biggest arg is `RELU_S·(15·65536)`-ish in
the KB-value bound path, still 50 900. 0 fp64 params.

## Verdict — ready as the shallow fp32 default divide

```
base-16 LONG DIVISION (nibble_alu32) : ~262 blocks
fp32 two-digit                       : ~274 blocks
fp64 log-sink                        : ~127 blocks
THIS hardened radix-16 (unrolled)    :   88 blocks   <-- shallowest, and NOW fp32-exact
```

88 blocks is still the shallowest of the four alternatives (88 < 127 / 262 / 274)
AND the only one that is fully-robust, always-exact, reciprocal-free **and**
byte-exact in genuine fp32 (0 fp64). The +8-block cost over the lean 80 buys the
elimination of the entire residue floor. **Ready to be the shallow (≤80s of
blocks; here 88) fp32 default divide.**
