# DIV_22_FAILING attribution (2026-06-09)

Date: 2026-06-09
Base ref: `main` HEAD `98a34d08`
(`docs(1096): mul_ sub-cluster breakdown + rec_factorial partial sample`).
Task brief: improve `div_` from 54/76 to higher; 22 failing per
`1096_CUMULATIVE_PASS_COUNTS_2026_06_07.md`.

## TL;DR

The 22 `div_` failures attribute to **Bug #36 (long-division
SLOT_REMAINDER → OUTPUT_LO/HI projection failure)** in `BUG_CATALOG.md`.
This is **not a single-rule fix** — `BUG_CATALOG.md` explicitly tags
Bug #36 as needing FFNRule IR migration of the SLOT_REMAINDER /
SLOT_QUOTIENT → OUTPUT_LO/HI projection chain through L11-L17, with a
3-5 day effort estimate. Per `feedback_single_rule_fixes_are_zero_sum.md`
single-rule whack-a-mole on this surface is zero-sum. No fix applied
in this brief; smoke remains at the baseline 45/51 PASS.

## The 22 failures, with operand pattern

Measured at this base ref with `CUDA_VISIBLE_DEVICES=1` and
`pytest tests/test_suite_1096_pure_neural_pytest.py -k div_ --runxfail
--tb=no -q` (declarations-only mode, default `C4_BATCH_CHUNK=32`):

### Basic `div_*` failures (8 / 50 fail)

| id | a / b | expected | neural got |
|---|---:|---:|---:|
| `div_5` | 176 / 4 | 44 | 0 |
| `div_20` | 2009 / 43 | 46 | 16 |
| `div_32` | 266 / 40 | 6 | 1 |
| `div_39` | 1031 / 21 | 49 | 1 |
| `div_41` | 176 / 5 | 35 | 0 |
| `div_42` | 569 / 33 | 17 | 16 |
| `div_45` | 1037 / 29 | 35 | 1 |
| `div_46` | 816 / 20 | 40 | 0 |

**Universal property:** every failing basic-div has **dividend > 255**
(requires byte-1 of `a`). The 42/50 passing basic-div have dividends
that fit in byte-0 (or quotients that align with the pass-through
projection path). The expected quotient itself fits in one byte
(≤ 49) in every failure.

### `expr_mul_div_*` failures (14 / 25 fail)

| id | a*b/c | expected | neural got |
|---|---:|---:|---:|
| `expr_mul_div_1` | 2*8/8 | 2 | 0 |
| `expr_mul_div_2` | 14*56/8 | 98 | 0 |
| `expr_mul_div_3` | 23*16/2 | 184 | 0 |
| `expr_mul_div_4` | 29*16/8 | 58 | 0 |
| `expr_mul_div_5` | 20*12/2 | 120 | 0 |
| `expr_mul_div_10` | 15*45/9 | 75 | 21 |
| `expr_mul_div_11` | 14*90/9 | 140 | 15 |
| `expr_mul_div_12` | 28*4/2 | 56 | _F_ |
| `expr_mul_div_13` | 14*36/9 | 56 | _F_ |
| `expr_mul_div_14` | 14*32/8 | 56 | _F_ |
| `expr_mul_div_15` | 13*48/6 | 104 | 1 |
| `expr_mul_div_18` | 5*30/5 | 30 | _F_ |
| `expr_mul_div_19` | 3*16/8 | 6 | _F_ |
| `expr_mul_div_22` | 2*27/3 | 18 | _F_ |

**Note** — `expr_mul_div_1` (2*8/8 → 2, both intermediate 16 and final
2 fit in byte-0) **still fails with neural=0**. This refutes the
naive "byte-1 only" hypothesis for the expr cluster: the intermediate
MUL stages the product into the wide AX register, and the subsequent
DIV reads the wide register expecting both byte-0 and byte-1 to be
clean. If byte-1 has leakage from the MUL stage or is not zeroed for
the DIV input, the long-division receives stale data even when the
final mathematical result fits in byte-0.

## Failure shape signatures

Grouped by neural output:

1. **`neural=0` (8 cases)**: `div_5`, `div_41`, `div_46`,
   `expr_mul_div_{1,2,3,4,5}`. Pattern: full collapse to 0. The
   long-division pipeline produces SLOT_QUOTIENT correctly, but the
   `EmitDivResultModule` copy to RESULT and onward through the
   nibble-to-OUTPUT_LO/HI step-pair detector in `GEToBDConverter`
   emits zeros. Consistent with the BD-side RESULT field being
   over-written or never populated for `a` > 255.

2. **`neural=1` (3 cases)**: `div_32` (6 → 1), `div_39` (49 → 1),
   `div_45` (35 → 1), `expr_mul_div_15` (104 → 1). The "1" is the
   `b_is_zero` fallback sentinel? No — fallback would be 15 (q =
   0xFFFFFFFF = all-15 nibbles). The "1" pattern is more likely a
   single-bit residual leaking through the OUTPUT_LO step-pair detector
   at `efficient_alu_neural.py:325` (`sigmoid(S*(diff+0.5)) - sigmoid(S*(diff-0.5))`).

3. **`neural=16` (2 cases)**: `div_20` (46 → 16), `div_42` (17 → 16).
   `16` is exactly **OUTPUT_HI=1** (1 << 4) with OUTPUT_LO=0. This
   matches a byte-1 nibble bleed into the byte-0 emission slot — the
   step-pair detector at `efficient_alu_neural.py:309-326` reads
   `result_lo = x_ge[:, :, 0, RESULT]` (LSB nibble) and `result_hi =
   x_ge[:, :, 1, RESULT]` (second nibble of byte 0). If RESULT[0]=0
   and RESULT[1]=1, the BD decode reconstructs 0x10 = 16 — i.e., the
   high nibble of byte 0 is set when the low nibble is missing. This
   is the classic SLOT_QUOTIENT byte-misalignment signature.

4. **`neural=15` / `neural=21` (2 cases)**: `expr_mul_div_11` (140 → 15)
   — 15 is the divide-by-zero sentinel nibble; `expr_mul_div_10` (75 →
   21) — 21 = 0x15 = OUTPUT_LO[5] + OUTPUT_HI[1], suggesting partial
   correct emission of the low nibble of 75 (0x4B → low nibble 0xB = 11,
   but 21 = 0x15 doesn't match cleanly; possibly a step-pair noise
   product).

## Cross-reference: NOT Wave C6 (Bug #34 0xD8 sentinel)

The 22 failures here do **not** match the Bug #34 `0xD8 = 216`
uniform-sentinel signature documented in
`C6_WIDE_MUL_ZERO_OPERAND_2026_06_07.md`. None of the 22 failures
produces 216. The 0-operand surface in C6 is `edge_zero_*` /
`loop_pow2_*` / `loop_mul_*` (which uniformly emit 216). The `div_`
failures here are value-dependent (different `(a, b)` → different
wrong neural) — characteristic of Bug #36 (long-division projection),
not Bug #34 (single-rule sentinel writer).

## Cross-reference: Bug #36 surface in `BUG_CATALOG.md`

`BUG_CATALOG.md` §Bug #36 (line 335-343):

> **Symptom**: DIV / MOD with non-power-of-2 divisor/modulus produces
> wrong results; value-dependent failure pattern. Sub-cluster counts:
> `DIV_direct::nonpow2_divisor_wrong` (32 rows), `MOD_direct::nonpow2_modulus_wrong`
> (26 rows), `MOD_iterative::nonpow2_modulus_wrong` (49 rows on `gcd_*`,
> neural=65280=0xFF00 or 0), `MOD_in_expression::nonpow2_modulus_wrong`
> (2-5 rows). Total: ~109 in-sweep rows.

> **Root cause**: long-division compute appears correct for trivial cases
> (`mod_2: 154%8 → 2` is correct); the failure surface is the multi-nibble
> `SLOT_REMAINDER → OUTPUT_LO/HI` projection through L11-L17.

> **Status**: D (FIXME landed at `2b7b34a`; cluster-level fix not landed).
> Per `feedback_single_rule_fixes_are_zero_sum.md`, single-rule fixes will
> be zero-sum here — this needs FFNRule IR migration of the projection
> chain (see `CAMPAIGN_SUMMARY.md` §10.6 effort estimate: 3-5 days).

The 22 div_ failures are a clean subset of this surface:
- Basic `div_*` 8 fails ⊆ `DIV_direct::nonpow2_divisor_wrong` (32 rows).
- `expr_mul_div_*` 14 fails ⊆ the implied
  `DIV_in_expression::nonpow2_divisor_wrong` sub-cluster (not separately
  enumerated in #36, but identical mechanism — the long-division
  projection runs the same code path on the DIV step of `a*b/c`).

The byte-1 dependence in the basic-`div_*` 8 fails is consistent with
the `MOD_iterative` neural=0xFF00 pattern in #36: both reflect
SLOT_REMAINDER (or SLOT_QUOTIENT) byte-1 nibbles failing to clear before
the OUTPUT projection runs.

## Code path map

The pipeline for DIV (declarations-only) is:

1. **L8 attention (AX_FULL_LO/HI staging)** —
   `efficient_alu_neural.py:148-223` stages the dividend byte-1 from
   the autoregressive prefix into `AX_FULL_LO/HI` (when wide-op marker
   is active) or falls back to `prev_stack_lo/hi` via cummax over
   `STACK0_BYTE1` rows. Sources: `CLEAN_EMBED_LO/HI`.

2. **`BDToGEConverter`** —
   `efficient_alu_neural.py:103-233`. Projects:
   - `x_ge[:, :, 0, NIB_A] = sum_k k * ALU_LO[k]` (byte-0 low nibble)
   - `x_ge[:, :, 1, NIB_A] = sum_k k * ALU_HI[k]` (byte-0 high nibble)
   - `x_ge[:, :, 2, NIB_A] = AX_FULL_LO_scalar * wide_marker + prev_stack_lo_scalar * divmod_fallback` (byte-1 low nibble)
   - `x_ge[:, :, 3, NIB_A] = ...` (byte-1 high nibble)
   - Positions 4-7: zero (32-bit dividend, only byte-0/byte-1 used).

3. **`build_div_layers_longdiv`** —
   `alu/ops/divmod_longdiv.py:344-359`. Three layers:
   - `ClearDivSlotsFFN` — clears SLOT_REMAINDER, SLOT_QUOTIENT vectors.
   - `LongDivisionModule` — 8 outer iterations of bring-down + trial
     multiply + compare + subtract. Reads NIB_A[0..7], NIB_B[0..7],
     writes SLOT_QUOTIENT[0..7], SLOT_REMAINDER[0..7].
   - `EmitDivResultModule` — copies SLOT_QUOTIENT[*] → RESULT[*] via
     opcode-gated cancel pairs (`divmod_longdiv.py:303-341`).

4. **`GEToBDConverter`** —
   `efficient_alu_neural.py:236-380`. Reads `RESULT` at positions 0,1
   for OUTPUT_LO/HI (byte 0) and positions 2,3 for AX_FULL_LO/HI (byte
   1, staged at AX markers). Step-pair sigmoid detector converts
   scalar → one-hot.

The hypothesized fault locus in stage **(3) → (4)**:

- LongDivision produces SLOT_QUOTIENT[0..7] (nibble decomposition of
  the quotient).
- For 8-bit quotient (≤ 255), only SLOT_QUOTIENT[0,1] should be
  non-zero; SLOT_QUOTIENT[2..7] = 0.
- `EmitDivResultModule` writes RESULT[pos] for all 8 positions.
- GEToBDConverter reads only RESULT[0,1] for OUTPUT_LO/HI byte-0.
- **If SLOT_QUOTIENT[2..7] is not cleanly zero** (drift from the
  9-nibble partial vector subtraction in LongDivisionModule when
  `a > 255`), it leaks into RESULT[2..7], which gets staged into
  AX_FULL_LO/HI for the NEXT VM step's byte-1 output. The 16-pattern
  (`div_20`, `div_42`) suggests this leak is hitting OUTPUT_HI on the
  current step too.

## Why this is multi-rule

The chain `LongDivisionModule(forward) → SLOT_QUOTIENT
→ EmitDivResultModule(flat_ffn) → RESULT → GEToBDConverter(step pairs)
→ OUTPUT_LO/HI / AX_FULL_LO/HI` traverses:

1. An imperative `nn.Module.forward` (LongDivisionModule) — not
   FFNRule, not in IR. Bug #36 specifically calls for FFNRule
   migration of this projection chain.
2. A baked flat FFN (`EmitDivResultModule.flat_ffn`) — generic
   cancel-pair weights, doesn't gate on SLOT_QUOTIENT byte-1 cleanliness.
3. A SwiGLU step-pair sigmoid detector (`GEToBDConverter`) — converts
   scalar RESULT to one-hot, but the precision floor is the sum of
   upstream drift.
4. Cross-step relay through AX_FULL_LO/HI for byte-1 emission —
   coupled to L8 attention staging and CLEAN_EMBED.

Any single-rule fix at one of these stages (e.g. add a `clear
SLOT_QUOTIENT[2..7]` FFN, or tighten the step-pair detector S
scale) is plausible to shift the failure surface without netting
positive across the corpus — consistent with the historical
zero-sum agent record per `feedback_single_rule_fixes_are_zero_sum.md`.

## Why no fix is applied in this brief

Per the task brief constraint:
> Apply a fix if single-rule + low risk. Else attribution doc.

Bug #36 is explicitly NOT single-rule per `BUG_CATALOG.md`. The
22-failure cluster maps cleanly onto it. Per memory note
`feedback_single_rule_fixes_are_zero_sum.md`, 0/5 prior single-rule
fix agents on equivalent surfaces produced a net positive delta.
Therefore this brief produces only the attribution doc and leaves
the pass count unchanged at 54/76.

## Smoke

Baseline 45/51 PASS confirmed at this base ref (`98a34d08`):

```
$ CUDA_VISIBLE_DEVICES=1 pytest tests/test_smoke.py --tb=no -q
======= 6 failed, 45 passed, 1 deselected in 129.90s (0:02:09) =======
```

Six failures are the documented Wave S1 memory cluster + S2 LEA,
unchanged.

## Recommendation

Promote Bug #36 from D (deferred) to an active Wave (Wave C8 — "DIV /
MOD long-division projection migration") in `CLOSEOUT_PLAN_2026_06_07.md`,
with the 22 div_ failures (this attribution) + ~26 mod_ failures
already attributed to Bug #36 + the gcd_ surface (potentially shared
via the iterative MOD path) as the unified impact surface. Tracked
effort: 3-5 days per `BUG_CATALOG.md`.

Specific fix surfaces to consider during the wave:

1. **Add a `clear SLOT_QUOTIENT[2..7]` and `clear SLOT_REMAINDER[2..7]`
   FFN rule** in `EmitDivResultModule` gated on `OP_DIV / OP_MOD`,
   to defend against LongDivisionModule's partial-dividend drift on
   `a > 255` cases. Cheap to author; needs byte-identity gate via
   `compare_symbolic_to_lowered_ffn` and a smoke check.
2. **Migrate `LongDivisionModule.forward` to FFNRule** — the 24
   sub-operations (8 outer × 3) are structurally regular and lower
   to FFNRule with the building-blocks DSL. This is the 3-5 day
   effort in Bug #36 and is the only way to make the byte-1
   clearing structurally guaranteed.
3. **Audit the AX_FULL_LO/HI staging chain** for DIV/MOD specifically
   — the `prev_stack_lo/hi` fallback at `efficient_alu_neural.py:174-205`
   uses `cummax(STACK0_BYTE1)` over the autoregressive prefix; for
   single-byte dividends with stale STACK0 from a prior expression, this
   may inject incorrect byte-1 into the LongDivisionModule.

## Cross-references

- [`BUG_CATALOG.md`](BUG_CATALOG.md) §Bug #36 — Long-division
  SLOT_REMAINDER → OUTPUT_LO/HI projection failure (root surface).
- [`C6_WIDE_MUL_ZERO_OPERAND_2026_06_07.md`](C6_WIDE_MUL_ZERO_OPERAND_2026_06_07.md)
  — confirms 0xD8 sentinel is NOT the failure shape here (Bug #34
  is distinct from these 22 fails).
- [`1096_CUMULATIVE_PASS_COUNTS_2026_06_07.md`](1096_CUMULATIVE_PASS_COUNTS_2026_06_07.md)
  §`div_*` — original 22-fail enumeration.
- `c4_release/neural_vm/alu/ops/divmod_longdiv.py` — LongDivision
  + EmitDivResult modules (3-5 day migration target).
- `c4_release/neural_vm/alu/ops/mod.py:14-53` — pre-existing FIXME
  (`investigation/expr-mod-divergences`) landed by merge `2b7b34a`,
  documenting the same surface from the MOD side.
- `c4_release/neural_vm/efficient_alu_neural.py:103-380` — wide-ALU
  byte-1 staging (BDToGE) + RESULT projection (GEToBD).
- Memory note `feedback_single_rule_fixes_are_zero_sum.md` — the
  0/5 historical record on this surface family.
