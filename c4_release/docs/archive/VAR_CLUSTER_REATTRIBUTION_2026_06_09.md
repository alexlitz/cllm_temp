# var_/if_var_/var_mul_ cluster re-attribution (2026-06-09)

**Supersedes:** `docs/VAR_MUL_ATTRIBUTION_2026_06_09.md`,
`docs/VAR_L3_SP_BYTE2_2026_06_07.md` (block-3 L3 lead is a verifier
false-positive at step 0; the *real* divergent block sits later).

**Driver:** block-aware oracle (`commit c7e45ab2`, see
`neural_vm/unified_compiler/dim_diff.py::first_writer_block_for_dim`).
The new oracle skips pre-writer blocks, so the reported first
divergence is the earliest block whose declared writer fires for the
target dim — not block 0 by default.

## Probe program

`JSR 3; EXIT; EXIT; ENT 8; LEA 0xfffff8; PSH; IMM 42; SI; LEA 0xfffff8; LI; EXIT`

A function-prologue + LEA-local + SI/LI round trip — the canonical
var_simple_0 / if_var_0 / var_mul_0 / func_mul_0 shape.

## Cluster pass/fail (HEAD = c0b02e7d, GPU 1, --runxfail)

`pytest -k "var_simple_0 or if_var_0 or var_three_0 or var_mul_0 or func_mul_0"`

| Test                 | Result | Neural exit | Decl. exit |
|----------------------|--------|-------------|------------|
| `var_simple_0`       | FAIL   | 65512       | 990        |
| `var_mul_0`          | FAIL   | 1           | 1081       |
| `var_three_0`        | FAIL   | 65496       | 55         |
| `if_var_0`           | FAIL   | 0           | 1          |
| `func_mul_0`         | FAIL   | 0           | 1764       |

**5/5 still failing.** No movement since the stale doc.

## Oracle output (atol=1.5; suppresses S=100 scaling noise)

| Dim                     | Block | Suggested op                         | Actual | Expected | Pattern                       |
|-------------------------|-------|--------------------------------------|--------|----------|-------------------------------|
| `OUTPUT_LO`             | **11**| `layer10_psh_stack0_passthrough`     | ~15.1  | 1.0      | ~5x over-fire on every step   |
| `OUTPUT_HI`             | **11**| `layer10_psh_stack0_passthrough`     | 15     | 1.0      | identical 5x over-fire        |
| `AX_CARRY_LO`           | **3** | `layer3_carry_forward_attn`          | 3      | 1.0      | 3x AO gain (intentional)      |
| `STACK0_BYTE_VAL_1_LO`  | **12**| `layer10_psh_ax_broadcast` (h=1)     | 3      | 1.0      | 3x AO gain (intentional)      |
| `STACK0_BYTE_VAL_2_LO`  | **12**| `layer10_psh_ax_broadcast` (h=2)     | 3      | 1.0      | identical                     |
| `AX_FULL_LO`            | —     | unsupported by oracle                | —      | —        | deferred dim family            |
| `MEM_value1`            | —     | unsupported by oracle                | —      | —        | deferred dim family            |

The L3 block-3 finding for `OUTPUT_LO` ("writer did not fire") at the
default `atol=0.5` is the documented step-0 verifier false positive
(residual is correctly zero on step 0). At `atol=1.5` the real
divergence rises to **block 11** (L10 PSH stack0 passthrough) with a
**15x** over-write into OUTPUT band 0.

## Failure topology has NOT shifted materially since 2026-06-02

`tools/attribute_1096_failure.py --test_id var_simple_0` confirms the
same first-token divergence as the stale doc:

- step 0, slot `MEM_value1`, expected `0x00` got `0xff`,
  suspect_dims = `OUTPUT_LO, OUTPUT_HI`.

The new evidence is the *upstream chain*: the OUTPUT band starts
diverging at L10 (block 11) by ~15x. By the time L14 `mem_generation`
reads OUTPUT_LO at the MEM_value1 row (downstream block), the band has
been polluted by overlapping L10 PSH writers.

## First divergent op (canonical): `layer10_psh_stack0_passthrough`

Location: `neural_vm/unified_compiler/ops/l10_ops.py:1559`
(`_layer10_psh_stack0_passthrough_head_spec`).

The head writes `AO(OUTPUT_LO+k, k, 3.0)` and
`AO(OUTPUT_HI+k, 16+k, 3.0)` for `k in range(16)` on three V/O slot
bands (slots 0..31 primary, 32..47 LEA-local diff LO, 48..63 LEA-local
diff HI; lines 1597, 1647, 1651). Combined with the LEA-local
differential routing each PSH AX-byte source contributes
≈ +3 (primary) + 3 (diff_lo) + 3 (diff_hi) into OUTPUT_LO/HI at the
same Q row. Observed 15.1 ≈ 5 contributors × 3.0.

The 3x O-gain is *intentional* per the LEA-local PSH fix (bug #33,
LEA-local AX byte 0 differential routing). The oracle's
`_emit_output_at_marker` projects a one-hot magnitude of 1.0 —
mismatch is in *spec*, not implementation.

## What's NOT the root cause

- L17 `layer14_mem_generation.head_1` (-0.391 OUTPUT_LO contribution,
  `docs/VAR_CLUSTER_L17_ATTRIBUTION_2026_06_05.md`) — that doc
  documents a downstream symptom, not the source.
- L3 `layer3_carry_forward_attn` (the original block-0 attribution,
  `docs/VAR_L3_SP_BYTE2_2026_06_07.md`) — step-0 false positive.
- L15 `nibble_copy` (`docs/VAR_CLUSTER_NIBBLE_COPY_LEAK.md`) —
  scope on byte_h=1, not byte_h=0 where the over-fire concentrates.

## Recommended next fix (not attempted here — DO NOT speculate)

Two viable directions, both touching scaling rather than logic:

1. **Oracle-side**: update `_emit_output_at_marker` in
   `dim_oracle.py:605-615` to emit magnitude 3.0 (or the actual L10
   PSH head O-gain), matching the lowered spec. Pro: makes the diff
   pass; con: hides cumulative-over-fire when multiple heads attend
   the same row.
2. **Implementation-side**: split the L10 PSH passthrough O-gain
   across the three V/O bands so the *summed* contribution at any
   single OUTPUT_LO cell is bounded at 1.0 (or 3.0 if downstream
   consumers genuinely need that gain). Pro: aligns with the
   one-hot semantics the oracle codifies; con: requires checking
   every downstream OUTPUT consumer (L14 `mem_generation`,
   `layer14_clear_output_corruption`, `layer14_temp_clear`, etc.)
   for the implicit 3.0/15.0 dependency.

Either direction needs a verifier sweep before commit —
single-rule attempts here are zero-sum per
`feedback_single_rule_fixes_are_zero_sum.md`.

**Cross-agent guard:** L10 PSH heads are also touched by the L14/L15
memory cluster agents. Coordinate before changing the O-gain in
`_layer10_psh_stack0_passthrough_head_spec` or
`_layer10_psh_ax_broadcast_head_spec`.
