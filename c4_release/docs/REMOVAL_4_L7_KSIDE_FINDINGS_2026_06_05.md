# Removal 4 — L7 K-side gate fix: premise contradicts the data

Date: 2026-06-05
Branch: speedup-cache-and-buckets (off `0b957e03`)
Author: agent (Claude Opus 4.7)
Status: FINDINGS ONLY — no model change applied, override retained

## TL;DR

The Removal 4 brief from
`docs/RUNNER_OVERRIDE_REMOVAL_PLAN_2026_06_05.md` asks to fix the L7
head 0 K-side gate per
`docs/CMP_POLARITY_INVESTIGATION_2026_06_03.md` §"Recommended fix
sequence", then remove the CMP non-collapsed runner synth at
`batched_pure_neural.py:2177-2192` (commit `69f77682`).

Both upstream docs hypothesize the staleness as an L7 head 0 K-side
issue with `STACK0_BYTE0`. After re-reading the polarity doc's own
empirical capture and the dim layout, **the L7 K-side gate cannot be
the root cause**:

1. `STACK0_BYTE0` is a 1-wide positional flag (`dim_registry_dynamic.py:264`)
   set by `_set_layer1_ffn` via `L1H4[BP] AND IS_BYTE AND NOT H1[BP]`
   — purely positional, not value-dependent.
2. The polarity doc's own failure table is strictly value-dependent:
   `EQ(0, 0)`, `EQ(16, 16)`, `EQ(128, 128)` pass; `EQ(17, 17)`,
   `EQ(42, 42)` fail. The PASS pattern is "either hi or lo nibble is
   0"; the FAIL pattern is "both nibbles nonzero".
3. If the L7 head 0 K-side were the bug, EQ would fail for ALL
   value pairs (the head either fires positionally or it doesn't —
   the value of the operand doesn't affect K-side attention). The
   value dependence rules out L7 attention as the divergence layer.
4. The polarity doc's residual capture at the AX position reads
   `STACK0_BYTE0 = 0.00` at the *Q row*, then concludes "WRONG: should
   encode 17". But `STACK0_BYTE0` at the AX position is *expected* to
   be 0 (the Q row is the AX marker, not a STACK0 byte). The
   "STACK0_BYTE0 = 0" observation is the byte position of the AX row,
   not evidence of a clear on the K target.

The real divergence is one or more layers earlier and is
value-dependent. Candidates:

- L0 byte-token embedding for byte 17 / byte 42 / byte 10 / byte 20:
  `model_ops.py:2099-2109` sets `CLEAN_EMBED_LO+lo=1`,
  `CLEAN_EMBED_HI+hi=1`. For 17 (`0x11`) this means
  `CLEAN_EMBED_LO+1=1` AND `CLEAN_EMBED_HI+1=1`. If a downstream layer
  uses one of those nibbles as a flag/gate input (e.g. an FFN unit
  reading `CLEAN_EMBED_LO+1` as a positive-condition residual), the
  value 17 would inadvertently trigger that gate at the PSH byte-0
  position and corrupt the bytes by L7.
- L1-L6 transit: `CLEAN_EMBED_LO/HI` is a 32-wide one-hot per nibble
  pair. The L1-L6 pipeline writes many flags into adjacent dim slots
  (CMP_GROUP at 305, SP_OLD_HI at 305..312 overlapping CLEAN_EMBED_LO
  at 306..321 — see `dim_registry_dynamic.py:302-309` for the noted
  overlap). A value-dependent CLEAN_EMBED+k bit could activate a
  spuriously-mapped flag at the STACK0-byte-0 position.

## Why no model change was committed

Per the constraints in the brief:

- **NO `git stash`** — held.
- "ONE compile + ONE smoke" budget — held (no compile burned because
  no model edit applied).
- "If smoke regresses after override removal, restore override and
  keep just the model-side L7 fix" — but the L7 fix's premise is
  contradicted by the value-dependent failure mode. Applying a
  speculative L7 K-side change has no theoretical basis and would
  burn smoke budget for no expected gain.
- The `feedback_single_rule_fixes_are_zero_sum.md` memory note: 0/5
  fix agents netted positive with similar speculative single-rule
  attempts in the past.

## What I confirmed

1. L7 head 0 K-side spec
   (`l7_ops.py:253`): `k=(AP(0, BD.STACK0_BYTE0, L), AP(33, BD.CONST, L))`.
   Purely positional gate on `STACK0_BYTE0`. No value-dependent term.
2. L7 head 0 V→O spec (lines 254-261): `_band_projection_writes(1,
   BD.CLEAN_EMBED_LO) + _band_projection_writes(17, BD.CLEAN_EMBED_HI)`
   and `_band_output_writes(BD.ALU_LO, 1, 6.0) + _band_output_writes(
   BD.ALU_HI, 17, 6.0)`. Slot 1+k pulls `CLEAN_EMBED_LO+k → ALU_LO+k`
   for k=0..15. Slot 17+k pulls `CLEAN_EMBED_HI+k → ALU_HI+k` for
   k=0..15. Byte-identical to legacy `_set_layer7_operand_gather`
   (verified by reading `setup_helpers.py:553-624`).
3. STACK0_BYTE0 flag setter (`setup_helpers_l1.py:34-42`): positional,
   value-agnostic. Sets `STACK0_BYTE0 += 1` when `L1H4[BP] AND IS_BYTE
   AND NOT H1[BP]` — i.e. byte-index 0 of the byte-block 6 positions
   after the BP marker. No nibble-value branch.
4. L0 byte embedding (`model_ops.py:2099-2109`): for byte b = (hi<<4)|lo,
   writes `IS_BYTE=1`, `EMBED_LO+lo=1`, `EMBED_HI+hi=1`,
   `CLEAN_EMBED_LO+lo=1`, `CLEAN_EMBED_HI+hi=1`. For b=17 (0x11),
   this sets `CLEAN_EMBED_LO+1=1` AND `CLEAN_EMBED_HI+1=1`. The pair
   (lo=1, hi=1) is what discriminates the FAIL bucket from the PASS
   bucket; values with either nibble = 0 (e.g. b=16 = 0x10 → lo=0,
   hi=1) pass.

## Recommended next step (not implemented here)

Run a single hooked-forward capture on `EQ(17, 17)` with the runner
override DISABLED and inspect the residual at the prior step's STACK0
byte-0 position:

```python
# Pseudocode for the next agent
model = build_model()
ctx = encode_program([(IMM, 17), PSH, (IMM, 17), EQ, EXIT])
state = trace_forward(model, ctx, hook_layer=7)
prior_stack0_b0_pos = locate_stack0_byte0_of_first_psh_step(ctx)
print("CLEAN_EMBED_LO at prior STACK0 b0:",
      state[prior_stack0_b0_pos][BD.CLEAN_EMBED_LO + 1])
print("CLEAN_EMBED_HI at prior STACK0 b0:",
      state[prior_stack0_b0_pos][BD.CLEAN_EMBED_HI + 1])
print("STACK0_BYTE0 flag at prior STACK0 b0:",
      state[prior_stack0_b0_pos][BD.STACK0_BYTE0])
```

If `CLEAN_EMBED_LO+1 = 0` at the prior STACK0 b0 position when
value=17, the bug is value-dependent CLEAN_EMBED clobbering somewhere
in L1-L6 — and the fix must land in whichever layer is clobbering
nibble bit 1. If `CLEAN_EMBED_LO+1 = 1` and `STACK0_BYTE0 = 1`, the
L7 head 0 attention should focus correctly, and the bug must be
downstream (L8/L9 ALU).

The minimum-blast-radius next agent should run that capture, NOT
modify L7 weights blindly.

## Override status

The CMP non-collapsed runner synth at `batched_pure_neural.py:2177-2192`
(commit `69f77682`) **remains in place**. CMP cluster (7 tests)
passes via the override. Removing it without a verified upstream fix
would drop smoke by 2-4 CMP tests with no model-side replacement
candidate identified.

## File touchpoints

- `c4_release/docs/CMP_POLARITY_INVESTIGATION_2026_06_03.md` (source
  brief — L7 K-side hypothesis contradicted)
- `c4_release/docs/CMP_DECODE_REWIRE_FINDINGS.md` (companion brief
  — L5/L9/L10 wires verified clean)
- `c4_release/docs/RUNNER_OVERRIDE_REMOVAL_PLAN_2026_06_05.md`
  (Removal 4 brief — premise inherited from polarity doc)
- `c4_release/neural_vm/unified_compiler/ops/l7_ops.py:233-288`
  (L7 head 0 + head 1 specs — verified structurally correct,
  value-independent)
- `c4_release/neural_vm/setup_helpers_l1.py:34-42` (STACK0_BYTE0
  positional flag — verified value-independent)
- `c4_release/neural_vm/unified_compiler/ops/model_ops.py:2099-2109`
  (L0 byte embedding — CLEAN_EMBED setup, lo/hi one-hot)
- `c4_release/neural_vm/batched_pure_neural.py:2177-2192` (runner CMP
  synth — RETAINED)

## Smoke

No code change applied. Smoke remains at current baseline (overrides
intact). The CMP cluster (7 tests under `TestSmokeComparison`) passes
via the override.

## Confidence

- **High** that the L7 head 0 K-side gate is value-independent
  (verified by source reading of the spec + the L1 STACK0_BYTE0 setter
  + the L0 byte embedding).
- **High** that a positional-only L7 K-side change cannot resolve a
  value-dependent failure mode (logical contradiction).
- **Medium** that the divergence is in L1-L6 value-dependent flag
  clobbering or at an L8/L9 ALU value-dependent gate. A single hooked
  forward at L7 input would discriminate the two.
- **Low** that any single-rule fix here will net positive smoke; the
  prior single-rule fix audit (`feedback_single_rule_fixes_are_zero_sum.md`)
  warns against speculative changes.
