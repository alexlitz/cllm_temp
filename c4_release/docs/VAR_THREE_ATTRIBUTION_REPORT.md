# var_three cluster: step-0 MEM_addr1 0xff under-fire (2026-06-03)

## Status

All 25 `var_three_*` cases in diag-1096 still fail. The attributor and
`run_1096_diag_chunk.sh 300 5` confirm an **identical** first-token
divergence across 8/8 sampled cases (`var_three_0,1,2,3,4,5,10,20`).
This refines (and partially supersedes) `VAR_CLUSTER_NIBBLE_COPY_LEAK.md`,
which described a STACK0-byte0 leak at later steps. The current failure
fires earlier and on a different slot.

## Common signature

```
step0:MEM_addr1 abs=247 gen=27 expected=0xff neural=0x00
residual_diagnosis=expected-token-never-wins
block=34 layer=34 width=2059  (= tail_bit32_result_correction, L17.post_ops[0])
expected=0xff  argmax=0x00   margin=-5.64
OUT_LO[15]=+0.32 arg=0/+0.78   OUT_HI[15]=+0.32 arg=0/+0.78
band_contracts=DRIFT
violations:
  OUTPUT_LO[15]/active_margin_low: obs=-0.464 (limit 0.5)
  OUTPUT_LO[0]/inactive_too_high:   obs=+0.782 (limit 0.2)
  OUTPUT_HI[15]/active_margin_low: obs=-0.464 (limit 0.5)
  OUTPUT_HI[0]/inactive_too_high:   obs=+0.782 (limit 0.2)
projection_diag=OUTPUT_LO[expected_low]:ambiguous_winner winner=0
```

`abs=247` = the MEM_addr1 slot inside step-0's symbolic emit stream.
Bytes 0xFF in addr-byte-1 correspond to the high byte of the
initial-JSR return-PC push at SP=0xFFFFFFF8 — the bootstrap is
storing to `0x..FFF8`, addr_byte0=0xF8 / addr_byte1=0xFF.

## Identified writer (under-firing)

`tail_mem_store_addr1_ff_from_stack_store_exact` at
`c4_release/neural_vm/unified_compiler/ops/l10_ops.py:5369` is the
canonical rule that should emit `OUTPUT_LO+15` / `OUTPUT_HI_THIS_STEP+15`
(value 0xFF) at this slot. It is gated at `threshold=140.0` with:

```
ADDR_B1_LO+15   +50
ADDR_B1_HI+15   +50
ADDR_B0_VALID   +50
IN_STEP_FRESH   +50
IS_BYTE         +5
H1+4            +20
BYTE_INDEX_0    +5
MEM_STORE       +5
MEM_ADDR_SRC    +2
CLEAN_EMBED_LO+8 / HI+15 +5/+5
H1+11           +5
```

Observed OUT_LO[15]=+0.32 indicates the rule fires WEAKLY (partial
activation): the structural-dim quadrant of evidence
(`ADDR_B1_LO/HI+15`, `ADDR_B0_VALID`, `IN_STEP_FRESH` = 4×50 = +200)
is NOT all present at the step-0 abs=247 row, so the activation never
crosses the +140 threshold cleanly. A competing OUTPUT_LO[0]/HI[0]
writer (default-zero output for non-MEM-store rows) wins at +0.78.

## Root-cause hypothesis (L13 producer side)

`ADDR_B0_VALID` is produced by `layer13_mem_addr_gather.head_0`
(slot 34) at `c4_release/neural_vm/unified_compiler/ops/l13_ops.py:485-513`.
At step 0 the L13 attention has no prior step to gather from — there
is no cross-step context yet. The companion `ADDR_B1_LO+15 /
ADDR_B1_HI+15` lanes are filled by the same L13 gather. If those four
witness dims are absent or below saturation on the very first step's
MEM_addr1 row, the +140 threshold is unreachable regardless of how
many positive lower-weight conditions match.

This is consistent with the diag's "band_contracts=DRIFT" tag — the
L17 tail rule sees the +5/+5 lower-strength evidence (IS_BYTE,
MEM_STORE, H1+4, etc.) but is missing the L13-sourced +200 mass.

## Cross-reference with VAR_CLUSTER_NIBBLE_COPY_LEAK.md

Same FFN bank (`tail_bit32_result_correction`, block 34 width 2059),
**different rule** and **different failure shape**:

| Aspect                | nibble_copy_leak.md      | this report             |
|-----------------------|--------------------------|-------------------------|
| Symptom step          | step 15 / step 22 etc.   | step 0 (first divergence) |
| Slot                  | STACK0_byte0             | MEM_addr1               |
| Incorrect argmax      | 0x0a (LEV PC marker)     | 0x00 (band 0 default)   |
| Margin                | -1.5e26 (saturated)      | -5.64 (sub-threshold)   |
| Upstream contaminator | `layer15_nibble_copy`    | L13 ADDR_B*_VALID gap   |

The nibble_copy leak was an **over-firing** of an unrelated PC=10
constant; this is an **under-firing** of the canonical addr1 rule.
Both project through the same tail amplifier, which is why both
show block=34 in the diag.

## Suggested next step

Verify the L13 head-0 slot-34 `ADDR_B0_VALID` lifecycle at step 0
abs=247 (MEM_addr1 slot). Either:

1. The L13 attn at step 0 has no key matches (V row mass = 0), so
   `ADDR_B0_VALID` is silent — fix on the L13 producer side.
2. The structural dims fire but the +50 weights don't sum because the
   attention output is too small in d_model to deliver saturation —
   fix on the head-output scale.

Option 1 is more likely given step 0's bootstrap context (no prior
step for the gather to pull from).

A minimal experiment: drop the threshold from 140 to 70 on
`tail_mem_store_addr1_ff_from_stack_store_exact` and re-run the
var_three slice. If the cluster recovers, the L13 witness gap is
confirmed and a proper fix is to either lift the threshold gating to
soft-evidence mode for step-0 rows, or seed `ADDR_B0_VALID` at
embedding time for the bootstrap MEM rows.

## Reproduction

```
cd c4_release
for tid in var_three_0 var_three_1 var_three_3 var_three_5; do
  timeout 240 python tools/attribute_1096_failure.py --test_id $tid
done
# OR with full residual probe:
bash tests/runners/run_1096_diag_chunk.sh 300 5 /tmp/var_three_diag.log 0
```

Briefs land at `c4_release/.agent-logs/1096_fail_var_three_*.md`.
