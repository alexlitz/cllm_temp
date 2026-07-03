# STACK0 byte-0 register-dump machinery — dead-weight deletion (2026-07)

## Summary

Deleted the STACK0 byte-0 register-**dump** machinery (Root 2, the historical
if/bool/expr "framing-drift" carry-and-re-emit fix) as **provably-dead weight**
in the default 30-token frame. The deletion is **geometry-changing** (it removes
residual bands, so it shifts `d_model` and the width-sensitive L25 tail bank), so
it was gated on **verdict-neutrality + a new golden hash**, NOT byte-identity.

- **Old golden** (`b4d2ab27…`) → **New golden**
  `81557d21422f3eada0a87c677b00dced41cc26c3ee3bfb094c5eeb71c9b4d3cb`
  (short `81557d21`).
- **FFN units:** 43071 → 43040 (−31: the 14-rule dump-repopulate FFN + the
  1/7/7/2-rule flag precursors). Build clean, no dead units.

## Why the dump machinery is dead

The dump only ever fired on a **carried STACK0-marker row** — a step that
re-emits a stack-top byte the LM head decoded on a prior step. In the default
30-token frame STACK0 is **never emitted** (`no_stack0_emit` / `operand_from_memsp`
default ON), so no STACK0-marker row exists, the dump gate never fires, and none
of the dump ops / bands / LM-head columns affect any decoded PC/AX/SP/BP/MEM
token. Post-AX-refactor the LM head no longer reads the base H1/H3 the dump
columns repoint into either.

## What was DELETED

Ops (`all_core_ops.py` registrations + factories):
- `make_stack0_byte0_dump_repopulate_op` (l11) — the L25 re-point dump FFN
- `make_stack0_byte0_carried_flag_op` (l11) — CARRIED precursor
- `make_stack0_byte0_sharp_flag_op` (l11) — SHARP precursor
- `make_stack0_byte0_prev_dom_flag_op` (l11) — PREV_DOM precursor
- `make_stack0_byte0_not_cmp_flag_op` (l11) — NOT_CMP precursor
- `make_stack0_byte0_popped_latch_op` (l11) — POP-discriminator latch head
- `make_stack0_byte0_dump_head_bake_op` (model_ops) — LM-head DUMP columns

Residual bands (7): `STACK0_B0_DUMP_H1`, `STACK0_B0_DUMP_H3`,
`STACK0_B0_CARRIED`, `STACK0_B0_SHARP`, `STACK0_B0_PREV_DOM`,
`STACK0_B0_NOT_CMP`, `STACK0_B0_POPPED` (the last was flag-gated,
`C4_STACK0_B0_POPPED` default-OFF, so absent from the golden build already).

Supporting code: `_stack0_b0_popped_enabled` predicate, the
`_stack0_b0_dump_blocks_on/off` builders, all `*_flag_rules` / `_dump_repopulate_rules`
helpers, the spec's `dump_blocks` / `precursors` fields, the `C4_STACK0_B0_DUMP`
cache-key entries, and the two obsolete `test_migrated_stack0_*` DSL tests.

## What was KEPT — the LIVE consumer

The L9 **cross-step carry HEAD** (`make_stack0_byte0_dump_carry_op`) and the two
`_PREV` bands `STACK0_B0_H1_PREV` / `STACK0_B0_H3_PREV` are **NOT dead** and were
kept. The head copies the prev step's STACK0-marker H1/H3 byte-0 one-hot into the
two PREV bands, whose **SUM is read LIVE** by the campaign MUL multi-byte L19
boost (`efficient_alu_neural._MulCombineStage`, `mul_multibyte_l19_boost_enabled`,
default-ON in the campaign config) as the literal-vs-`var_mul` frame
discriminator (`var_frame_carry`: ~7379 in a `var_mul` ENT frame, ≤667 for a
single-step literal mul). Deleting the head would zero the discriminator and
mis-fire the boost on `var_mul`. The two `('stack0_byte0_dump_carry',
'H1|H3.*.-1')` entries in `CROSS_STEP_DOCUMENTED_SAFE` are retained for the same
reason.

## Gate evidence

- **Verdict-neutral:** `run_1096_canonical.py --ids 0-9,50-59,100-109,250-259,
  550-559,900-909 --criterion full_trace --spec-k 0 --max-steps-cap 600` is
  **field-identical** pre- vs post-deletion (60/60 rows, 47 pass / 13 fail,
  fingerprint `f48bb520…` unchanged). The first (carry-head-also-deleted) attempt
  broke ids 100-109 (`AttributeError STACK0_B0_H1_PREV` from the live MUL reader) —
  which is exactly how the live consumer was found; keeping the head restored
  field-identity.
- **Smoke:** identical 5-fail set to base `703cff47`
  (`{si_li_overwrite, add_16bit, sub_16bit, sub_borrow_cascade, or_16bit}`, all
  pre-existing 16-bit / memory walls); **`test_cmp_and_branch` PASSES** (the
  width-lock canary — the L25 tail bank survived the geometry change).
- Cross-step safety (`test_compile_cross_step_safety.py`) and
  `test_dim_allocator.py` pass. The lone `test_isa_semantics_dsl` failure
  (`test_cam_lookup_reexpresses_l7_operand_gather_head0`) is pre-existing on base.

## Follow-up (not done here)

The L5 #221 consumer-lookahead gate (`STACK0_B0_NEXT_ARITH`,
`STACK0_PRIOR_ARITH`, `STACK0_B0_DUMP_BLOCK`, flag `C4_STACK0_NEXT_ARITH`) now has
its terminal output `STACK0_B0_DUMP_BLOCK` read by **nobody** (its only reader was
the deleted dump). It is thus dead-but-harmless (a written-unread band, like
`H*_DUMP_OUT`) and was left in place to keep this deletion scoped; it is a
candidate for a follow-on cut.
