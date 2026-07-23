# Batched Kogge-Stone KB-precompute — cutting the divide's KB prologue depth

`div_kb_lookahead.py` is a **standalone, pure-CPU, byte-exact bakeoff** that shrinks
the c4_min general divide's **KB-precompute depth** by replacing the
`15 × _KB_CARRY_ROUNDS = 90` serial ripple carry rounds with **one batched base-16
Kogge-Stone parallel-prefix resolve** that normalises all 15 `KB[k] = k·b` at once.

It never edits `nibble_alu32.py`; it *imports* the proven SwiGLU primitives
(`_ident`, `_clear`, `_step_ge`, `_floor_div_pow`, `_floor_div_pow2`,
`_nibble_carry_round`, `_truncate`, `_empty_spec`, `S`, `RELU_S`) and reuses the
production `_kb_precompute_blocks` front-end / baseline, then ports the Kogge-Stone
`round1 / G-P / KS-stage / apply` logic from `mul_lookahead.py`.

## The target

`nibble_alu32._kb_precompute_blocks` builds `KB[k] = k·b` for `k = 1..15` from the AX
(=B) nibble bands. Each `KB[k]` is a per-nibble multiply `b_nib[c]·k ≤ 15·15 = 225`
laid into `RN = 9` raw columns (`< 256`), then carry-normalised **in its own tiny
block** with `_KB_CARRY_ROUNDS = 6` ripple rounds:

```
raw (1) | [ KB[k] : 6 ripple rounds ] × 15 = 90 | BZ (1)   = 92 blocks
```

The 15 `KB[k]` are **independent numbers** (nothing carries *between* them), so their
90 carry-resolves are 15 separate 9-column ripples that need not run serially — they
**batch into one parallel-prefix pass**. The ripple dominates the precompute depth (90
of 92 blocks); raw + BZ are only 2.

## The batched-prefix design

Each `KB[k]`'s 9 post-raw columns are a base-16 carry-save number (`< 256`): the
resolve is a carry-**propagate** add over the columns. Kogge-Stone computes every
column's carry in `⌈log₂ 9⌉ = 4` prefix stages instead of 6 serial ripples, using the
same clean-binary-CLA trick as `mul_lookahead`:

- **round-1** (one carry round, run independently per band via the library
  `_nibble_carry_round`) reduces each column from `< 256` to
  `t_c = (col_c mod 16) + floor(col_{c-1}/16)` in `[0, 30]`. Now, given an incoming
  **binary** carry `b_c ∈ {0,1}`, each column emits final digit `(t_c + b_c) mod 16`
  and a **binary** carry-out `[t_c + b_c ≥ 16] ∈ {0,1}` (since `t_c + b_c ≤ 31`). So
  the carries are pure binary and obey the textbook CLA recurrence:
  - `G_c = [t_c ≥ 16]`, `P_c = [t_c == 15] = [t_c ≥ 15] − [t_c ≥ 16]`
  - `b_{c+1} = G_c OR (P_c AND b_c)`, `b_0 = 0`.
- **Kogge-Stone** combines the pairs `(G, P)` with the associative operator
  `(g_hi,p_hi) ∘ (g_lo,p_lo) = (g_hi OR (p_hi AND g_lo), p_hi AND p_lo)` in 4 stages.
  For 0/1 lanes each combine is one single-level staircase:
  `G_dst = [2·G_c + P_c + G_{c-d} ≥ 2]` (exact OR-of-AND), `P_dst = [P_c + P_{c-d} ≥ 2]`.
- **apply** writes `digit_c = t_c + G_final[c-1] − 16·G_final[c]` directly into `a.KB`
  (no separate result-copy block; the top carry-out overflows past the kept nibbles
  and is dropped, i.e. `KB[k] & 0xFFFFFFFF`).

### The batching — the whole point vs `mul_lookahead`

`mul_lookahead` runs the prefix over **one** 8-column value. Here the 15 `KB[k]` are
independent, so their `t / G / P` lanes are laid **side-by-side** (`15 × RN` per lane
band, in private `DKB_*` scratch), and every prefix block loops over all `15 × RN`
positions in **one** block. A carry never crosses a `KB[k]` boundary (the prefix
distance `d` stays inside each band's 9-column window: the combine at column `c` in
band `k` only reaches `c-d ≥ 0` in the **same** band). So one Kogge-Stone stage
resolves all 15 independent numbers simultaneously.

### Depth

```
raw (1) | round-1 (1) | G/P (1) | KS × 4 | fused apply/result (1) | BZ (1)  = 9 blocks
```

vs the baseline's 92 (raw + `15 × 6 = 90` ripple + BZ). **4 prefix stages vs 90 ripple
rounds.**

## Results (`python -m c4_min.div_kb_lookahead`)

| Variant          | Depth | Weights nz | Prefix / rounds | Tightest RELU_S·arg | 0xFFFFFFFF·k | Byte-exact |
|------------------|-------|-----------|-----------------|---------------------|--------------|------------|
| baseline_ripple  | 92    | 100 151   | 90 ripple       | 48 000 (0.286 % 2²⁴) | True         | 15915/15915 |
| **batched_prefix** | **9**  | **32 426** | **4 stages**    | 48 000 (0.286 % 2²⁴) | True         | 15915/15915 |

- **KB-precompute depth: 92 → 9 blocks (−83), 90 ripple rounds → 4 prefix stages.**
- **nz: 100 151 → 32 426 (32.4 % of baseline)** — the batched prefix is *also* ~3×
  cheaper in nonzero weights (one shared round-1 + 4 prefix stages beat 90 full
  ripple rounds each carrying the `_floor_div_pow2` staircase).

### Byte-exact

Verified through the real SwiGLU sparse forward sim (no dense DIM matmul, fp32):
- Full driver (`n_random=1000`): **15915/15915** = all 15 `KB[k] == (k·b) & 0xFFFFFFFF`
  for 1061 divisors (1000 random + 61 structured edges), 0 fails.
- Extended batched-only run (`n_random=1500`): **23415/23415** (1561 divisors), 0 fails.
- The edge set includes `0xFFFFFFFF` (which with `k = 15` gives the **maximal carry
  propagation** across all 9 columns — the whole point of the carry path), every power
  of two `2^0..2^31`, and byte-lane patterns. `0xFFFFFFFF·k` passes for all k.

### fp32 discipline

Every relu argument is fp32-exact: raw columns `≤ 225 < 256`, round-1 outputs
`t_c ≤ 30`, the whole prefix operates on 0/1 generate/propagate lanes (weighted forms
`≤ 4`). The single tightest `RELU_S·arg` is the round-1 floor staircase
`RELU_S·(15·16) = 48 000`, only **0.286 % of 2²⁴**. **0 fp64** — the sim runs pure
`torch.float32`.

## Divide knock-on

The KB-precompute is the divide's **fixed prologue**: `compile_divmod_blocks` /
`compile_divmod_blocks_recurrent` run it **once per divide**, before the 8 base-16
digit iterations (`KB[k] = k·b` is the per-`k` threshold table the quotient digit
`q = Σ_{k=1..15}[r ≥ k·b]` compares against). Batching the 15 independent `KB[k]`
carry-resolves cuts that prologue:

```
92 → 9 blocks  (saves 83), off EVERY divide
```

independent of the 8-iteration inner loop. This is the **real** batched-prefix win:
the per-iteration QB (`q·b`) normalise is a separate 9-column ripple that is already
only 6 rounds (a wash for prefix, per `mul_lookahead.div_knockon`), but the KB
prologue's `15 × 6 = 90` independent ripple rounds are exactly the batchable
structure a shared parallel-prefix pass collapses.
