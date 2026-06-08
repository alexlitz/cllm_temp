# L34 `tail_bit32_result_correction` over-fire table — XOR_BASIC IMM step (2026-06-07)

Per-rule contribution table for the L34 over-fire surfaced in
`COLLAPSED_STEP_REAL_SURFACE_2026_06_07.md`. This is the diagnostic
data Wave 2 Edits B+C need: it identifies exactly which rules in
the 2,059-rule `tail_bit32_result_correction` family fire at the
IMM step where the `f3342968` collapsed-step override compensates.

No model changes. Probe-only.

## Setup

- Program: `XOR_BASIC` = `IMM 0xFF; PSH; IMM 0xD5; XOR; EXIT`.
- Step probed: **step 2** (the second IMM — collapsed-step override
  surface; see `batched_pure_neural.py:2147-2187`).
- Position probed: **token 132** (STACK0 marker of step 2; same
  position as `/tmp/probe_block_chain.py`).
- Block probed: **block 34** (= `tail_bit32_result_correction` PureFFN,
  hidden_dim 2059).
- Method: per-hidden-unit decomposition of the SwiGLU FFN forward —
  `c[d, u] = W_down[d, u] · silu(W_up[u]·x + b_up[u]) · (W_gate[u]·x + b_gate[u])`
  summed across `d ∈ OUTPUT_LO[0..15] ∪ OUTPUT_HI_THIS_STEP[0..15]`.
- Probe script: `/tmp/probe_l34_overfire_table.py`. JSON dump:
  `/tmp/probe_l34_overfire_table.json`.

## Carried-in OUTPUT residual at L34 input (probe_pos=132)

The dominant spikes carried into L34 from L27 + L28 amplifier cascade
(values from x = ffn34 input residual):

| dim | value | notes |
|---|--:|---|
| `OUTPUT_LO+0`  | +5.29e+02 | (smaller secondary spike, lo nibble = 0 channel) |
| `OUTPUT_LO+5`  | +2.23e+00 | (noise) |
| `OUTPUT_LO+10` | +8.95e-01 | (noise) |
| **`OUTPUT_LO+15`** | **+3.91e+03** | **dominant — low nibble of 0xFF = 0xF = 15** |
| `OUTPUT_HI+0`  | +5.29e+02 | (smaller secondary spike) |
| `OUTPUT_HI+1`  | +8.94e-01 | (noise) |
| `OUTPUT_HI+13` | +2.23e+00 | (noise) |
| **`OUTPUT_HI+15`** | **+3.87e+03** | **dominant — high nibble of 0xFF = 0xF = 15** |

This matches the `COLLAPSED_STEP_REAL_SURFACE_2026_06_07.md` cascade
prediction (L27 +40 spike → L28 ×97 amplifier → ~+3,868 at L34 input)
with the value-specific shift: XOR_BASIC's `IMM 0xFF` writes lo=15,
hi=15 (versus SUB's `IMM 50 → IMM 8` which writes lo=2, hi=3 in the
doc's example).

## Aggregate over-fire (XOR_BASIC IMM step)

| metric | value |
|---|--:|
| Total rules in L34 FFN | 2,059 |
| Rules firing (\|abs_sum\| > 1e-3) | **59** |
| Constructive sum (Σ positive signed_sum) | +0.00e+00 |
| **Destructive sum (Σ negative signed_sum)** | **−1.87e+10** |
| Total \|abs\| sum | +2.14e+10 |
| Firing rule family | `tail_stack0_pop_loaded_byte_*` (all 59) |
| "Legit" load-path rules firing (LI/LC/SI) | 0 |
| "Leaked" rules firing | 59 |

All 59 firing rules belong to the `stack0_pop_loaded_output_rules`
family defined at `l10_ops.py:4007-4052`. None of the legitimate load
paths (`stack0_store_loaded_output_rules`, `addr_from_l13_rules`,
`stack0_store_top_e0_output_rules`, LEA/SP-pop families, exact-output
guarantees) fire at this probe — confirming the cascade is exclusively
amplifying the leaked family.

The signed sum is much larger than the −6.29e8 figure quoted in the
surface doc because:
1. The surface doc used SUB shape (`IMM 50; PSH; IMM 8; SUB`) where
   the lo/hi spike is `+3,868` on lanes `lo=2, hi=3`.
2. XOR_BASIC `IMM 0xFF` puts both spikes on `OUTPUT_LO/HI+15` (the
   corner where the entire (lo=15, *) ∪ (*, hi=15) sub-family of 31
   rules fires hard).
3. The signed sum scales with `hidden_act × Σ_d W_down[d, u]` — and
   `byte_writes(value=0xFF)` writes large negatives across all 16 LO +
   16 HI dims when summed.

## Top-25 firing rules (sorted by |signed_sum|)

| unit | rule name | hidden act | signed Σ (OUTPUT writes) | \|abs\| Σ | class |
|---:|---|--:|--:|--:|:---|
| 308 | `tail_stack0_pop_loaded_byte_ff` | +7.66e+04 | **−1.07e+09** | +1.23e+09 | leaked |
| 293 | `tail_stack0_pop_loaded_byte_0f` | +4.32e+04 | −6.05e+08 | +6.92e+08 | leaked |
| 68  | `tail_stack0_pop_loaded_byte_f0` | +4.28e+04 | −6.00e+08 | +6.85e+08 | leaked |
| 306 | `tail_stack0_pop_loaded_byte_df` | +3.80e+04 | −5.31e+08 | +6.07e+08 | leaked |
| 294 | `tail_stack0_pop_loaded_byte_1f` | +3.79e+04 | −5.31e+08 | +6.07e+08 | leaked |
| 307 | `tail_stack0_pop_loaded_byte_ef` | +3.79e+04 | −5.31e+08 | +6.07e+08 | leaked |
| 297 | `tail_stack0_pop_loaded_byte_4f` | +3.79e+04 | −5.31e+08 | +6.07e+08 | leaked |
| 298 | `tail_stack0_pop_loaded_byte_5f` | +3.79e+04 | −5.31e+08 | +6.07e+08 | leaked |
| 301 | `tail_stack0_pop_loaded_byte_8f` | +3.79e+04 | −5.31e+08 | +6.07e+08 | leaked |
| 302 | `tail_stack0_pop_loaded_byte_9f` | +3.79e+04 | −5.31e+08 | +6.07e+08 | leaked |
| 305 | `tail_stack0_pop_loaded_byte_cf` | +3.79e+04 | −5.31e+08 | +6.07e+08 | leaked |
| 295 | `tail_stack0_pop_loaded_byte_2f` | +3.79e+04 | −5.31e+08 | +6.07e+08 | leaked |
| 296 | `tail_stack0_pop_loaded_byte_3f` | +3.79e+04 | −5.31e+08 | +6.07e+08 | leaked |
| 299 | `tail_stack0_pop_loaded_byte_6f` | +3.79e+04 | −5.31e+08 | +6.07e+08 | leaked |
| 300 | `tail_stack0_pop_loaded_byte_7f` | +3.79e+04 | −5.31e+08 | +6.07e+08 | leaked |
| 303 | `tail_stack0_pop_loaded_byte_af` | +3.79e+04 | −5.31e+08 | +6.07e+08 | leaked |
| 304 | `tail_stack0_pop_loaded_byte_bf` | +3.79e+04 | −5.31e+08 | +6.07e+08 | leaked |
| 148 | `tail_stack0_pop_loaded_byte_f5` | +3.76e+04 | −5.26e+08 | +6.01e+08 | leaked |
| 228 | `tail_stack0_pop_loaded_byte_fa` | +3.75e+04 | −5.26e+08 | +6.01e+08 | leaked |
| 84  | `tail_stack0_pop_loaded_byte_f1` | +3.75e+04 | −5.26e+08 | +6.01e+08 | leaked |
| 196 | `tail_stack0_pop_loaded_byte_f8` | +3.75e+04 | −5.25e+08 | +6.01e+08 | leaked |
| 100 | `tail_stack0_pop_loaded_byte_f2` | +3.75e+04 | −5.25e+08 | +6.01e+08 | leaked |
| 132 | `tail_stack0_pop_loaded_byte_f4` | +3.75e+04 | −5.25e+08 | +6.01e+08 | leaked |
| 164 | `tail_stack0_pop_loaded_byte_f6` | +3.75e+04 | −5.25e+08 | +6.01e+08 | leaked |
| 212 | `tail_stack0_pop_loaded_byte_f9` | +3.75e+04 | −5.25e+08 | +6.01e+08 | leaked |

## Three rule sub-bands

The 59 firing rules cleanly partition by which condition lane they key
on (the broad-family conditions are
`(OUTPUT_LO+lo, 0.1) + (OUTPUT_HI_THIS_STEP+hi, 0.1)`):

### Sub-band A — `(lo=15, hi=15)` corner (rank 1)

`tail_stack0_pop_loaded_byte_ff` (unit 308). Both condition dims hit
the +3.9e+03 spike, so `0.1·3909 + 0.1·3870 ≈ 778` over the +2.5 base
sum, well past threshold 10.5. Largest contribution: −1.07e9.

### Sub-band B — `(lo=15, hi=*)` and `(lo=*, hi=15)` edges (ranks 2-31)

30 rules with exactly one of the two condition dims keyed on `15`. Each
contributes ~ −5.3e+08 to −6.0e+08. These dominate the destructive sum
(31 of the 59 firing rules × ~5.3e8 ≈ −1.65e+10 of the −1.87e+10 total).

### Sub-band C — secondary `(lo=0, hi=*)` and `(lo=*, hi=0)` edges (ranks 32-59)

28 rules with one condition keyed on `0`, picking up the smaller
+5.3e+02 spike on `OUTPUT_LO/HI+0`. Each contributes ~ −5.8e+07. These
form the remaining ~−1.6e+09 of the destructive sum.

## What this means for Wave 2 Edit B+C

The doc's predicted scope (joint plan Edit B at `l10_ops.py:4007-4052`)
is precisely the set of rules that fire — confirming the surgery
target. Recommended thresholds (per joint plan, calibrated against
this table):

- **Edit B** (lower condition weight 0.1 → 0.05, raise threshold 10.5
  → 12.0): For sub-band B's ~+3,868 spike, activation drops to
  `2.5 + 0.05·3868 + 0.05·0 = 196`, still well over 12.0. Doesn't
  fix the over-fire on its own.
- **Edit C** (re-key from `OUTPUT_LO/HI` to `ALU_LO/HI`): structurally
  eliminates the cascade since `ALU_LO/HI` carry the actual loaded
  byte (not the corrupted L28-amplified spike). All 59 firing units
  here would stop firing because `ALU_LO+15` is not the carried-in
  spike — it's the legitimate L8/L9 ALU output, which for the IMM
  step holds `0x00` (no load happened).
- **Edit A** (L28 starve) must land first to bring the +3,868 down to
  L27's raw +40. With Edit A alone applied: condition contribution
  becomes `0.1·40 + 0.1·40 = 8`, total activation = 2.5 + 8 = 10.5
  (right at threshold). Marginal — Edit B+C is the durable fix.

## Cross-references

- Surface doc: `c4_release/docs/COLLAPSED_STEP_REAL_SURFACE_2026_06_07.md`
- Joint plan: `c4_release/docs/COLLAPSED_STEP_JOINT_FIX_PLAN_2026_06_07.md`
- L34 rule source: `c4_release/neural_vm/unified_compiler/ops/l10_ops.py:4007-4052`
- Probe script (preserved): `/tmp/probe_l34_overfire_table.py`
- Per-rule JSON: `/tmp/probe_l34_overfire_table.json`
- Block-chain probe (cascade origin): `/tmp/probe_block_chain.py`

## Confidence

- **High** that all 59 firing rules belong to
  `stack0_pop_loaded_output_rules` (direct rule-name verification).
- **High** that the over-fire is keyed on `OUTPUT_LO+15` and
  `OUTPUT_HI+15` for the XOR_BASIC `IMM 0xFF` shape (matches L27
  nibble-copy semantics: low nibble = 15, high nibble = 15).
- **High** that no legitimate-load (LI/LC/SI) rule fires at this
  probe — Edit C's re-keying to `ALU_LO/HI` would not break any
  currently-firing legitimate path in this step.
- **Medium** that the doc's −6.29e8 figure refers to the SUB shape;
  the XOR_BASIC shape produces a larger −1.87e+10 because both lo and
  hi spikes land on lane 15.
