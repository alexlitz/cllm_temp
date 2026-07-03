# Wave B Phase 2 — SE relay transmits + SE readers consume SE_*

Date: 2026-06-12. HEAD base: `f92783ac`. Smoke (spec_k=0, GPU0): **48
passed / 3 failed {mul_basic, mul_overflow, simple_function} / 1
deselected** — UNCHANGED from the Phase-1 baseline (zero regression).

Supersedes the Phase-1 *blocker* verdict in
`WAVE_B_PHASE1_AUDIT_2026_06_12.md` §2-4. Phase 1 concluded the relay
could not be re-homed without regressing lt/le (the block-11-head-3
"PSH STACK0 passthrough" load-bearing claim). **That claim was wrong**:
direct weight inspection (below) shows blocks[11] heads 3/4 are PURE
relay heads in the current baseline — the L10 passthrough V/O were
already physically displaced by the relay. The only defect was the
ALiBi *slope*, not head ownership. Re-sloping is additive.

## What landed (3 edits, each smoke-gated 48/3/0)

### 1. The relay TRANSMITS — `make_layer9_se_relay_slope_op` (l9_ops.py)

Root cause (decisive, spec_k=0): the L9 `step_end_operand_relay` heads
A/B land physically on `model.blocks[10]` (pre-expansion) = physical
block 11 = logical L10 (post-expansion). `make_layer10_residual_alibi_
slopes_op` runs LATER (phase 999.1) and overwrites
`blocks[10].alibi_slopes[3]=0.5, [4]=1.0` (the legacy PSH/STACK0
passthrough slopes). Over the d=29 MARK_AX→MARK_SE gap, slope 0.5 is a
−14.5 ALiBi penalty that swamps the L=10 QK match → the relay attends
to nothing → `SE_*` collapses to its bias floor (~−1.4). Wall 2
(`project_attention_dsl_alibi_slope_gap`).

Direct weight probe (`tools/probe_relay_landing.py`, `wb_probe_head3`):
blocks[11] heads 3/4 carry ONLY the relay's `Q@MARK_SE / K@MARK_AX /
V(ALU,CMP,OP / AX_CARRY) / O(SE_*)`. No surviving L10 passthrough V/O.
So the 0.5/1.0 slopes were being applied to heads that no longer do
passthrough — re-sloping them to the relay's intended 0.2 removes
nothing functional.

Fix per the slope-ownership rule ("set the slope LAST"): a new
weight-free model op, `phase=999.2`, `requires after
layer10_residual_alibi_slopes`, re-asserts `blocks[10].alibi_slopes[
head_a]=0.2, [head_b]=0.2` (head_a from `_l9_head_idx`). Registered in
`all_core_ops()` right after the relay op.

VERIFIED (`tools/probe_l9_se_relay.py`, LT 5<7): before = flat −1.4
floor, all SE cold. After = `SE_ALU_LO` hot at {0, 7} (= operand A low
nibble), `SE_AX_CARRY_LO` hot at {5} (= operand B), `SE_OP_LT` hot
(4.35) — faithfully mirroring the MARK_AX source one-hots.

### 2. L9 CMP readers — already on SE_* (no edit)

`_layer9_cmp_rules` (commit 62b64449) already read `SE_ALU_LO/HI`,
`SE_AX_CARRY_LO/HI`, gate `SE_CMP_GROUP`. They were STARVED only because
the relay didn't transmit. With (1) landed they now compute the raw CMP
cascade FRESH at the SE row (probe `wb_probe_se_compute`: rawCMP at the
SE row hot at {0,1,2,3} after L9 FFN). No code change needed here.

### 3. L10 cmp_combine — gate on SE_OP_<cmp> (l10_ops.py)

`_layer10_alu_cmp_combine_rules` gated on `MARK_SE_ONLY` but read raw
`OP_<cmp>` (default) and raw `CMP+i` (override), both via
`dim_ref("opcode_flag", …)`. Raw `OP_<cmp>` is COLD at the SE row (the
opcode flag lives at MARK_AX). Switched the dispatch gate to the relayed
`SE_OP_<cmp>+0` mirror. The CMP *cascade* stays raw `CMP+i` — it is
computed fresh at the SE row by the L9 CMP rules (edit 2), NOT mirrored
from MARK_AX, so `SE_CMP` would be cold and was the WRONG dim. Added
`SE_OP_EQ..GE` to `make_layer10_alu_op`'s `reads` set for liveness.

This is additive/inert at the decode: the live decode reads the MARK_AX
row (the `_layer10_alu_ordering_engine_rules` engine + ComparisonCombine
at block 22), which is untouched. eq/ne/lt/gt/le/ge all decode
correctly (7/7 `TestSmokeComparison`). The cmp_combine's SE-row
`OUTPUT_LO` writes are dwarfed by a −240 OUTPUT clear at that non-decode
row — exactly the Wall-4 frontier (below).

### NOT done this phase (deferred to Phase 3, documented blocker)

**L11/L12 MUL readers** (`_layer11_mul_partial_rules`,
`_layer12_mul_combine_rules`): read raw `ALU_LO/HI`, `AX_CARRY_LO/HI`,
gate raw `OP_MUL`. Switching to SE_* requires a NEW `SE_OP_MUL` dim. The
runtime dim layout is liveness-graph-coloured to d_model=872
(`layer_compiler._compute_dim_layout_with_liveness`) — registry pins are
ADVISORY and reflowed (SE_ALU_LO pins at 837 but lands at 794;
SE_CMP_GROUP pins 911 → 868). Adding `SE_OP_MUL` means declaring it +
referencing it in the relay writes + l11/l12 reads and letting the
colourer place it (free slots exist at 860/861/870/871). This risks
byte-identity drift on the 48 passing tests and — per
`project_mul_div_mod_arch_blocked` and Phase-1 §Phase-3 — MUL is ALSO
blocked by the L15/L25 OP_MUL OUTPUT-materialization corruptor, so the
relay alone won't make mul pass. Deferred to Phase 3 where it is
co-designed with the L15/L25 fix. (Registry edit was attempted and
reverted; `dim_registry_dynamic.py` is unchanged.)

## Phase 3 plan (the Wall-4 decode-row move + MARK_AX retirement)

The SE path now COMPUTES correctly at the SE row (operands transmit, L9
CMP cascade fires, cmp_combine gates on SE_OP) but the decode never
reads the SE row. Phase 3 moves the decode to the SE row, then retires
the MARK_AX engines. Ordered, each per-cmp smoke-gated:

1. **Route the SE cmp result into the decode.** Either (a) a
   within-step SE→AX relay head copying the SE-row `OUTPUT_LO` into the
   same step's MARK_AX `OUTPUT_LO` before ComparisonCombine reads it, OR
   (b) re-point L3 head 5 (`_ax_full_relay_head_spec`,
   `l3_ops.py`) / ComparisonCombine (block 22) to read the SE row.
   FIRST neutralise the −240 OUTPUT clear at the SE row (find the L10
   FFN rule writing OUTPUT_LO≈−240 at MARK_SE_ONLY — `wb_probe_combine_
   fire` shows it dominates) so the cmp_combine ±4/S survives.
   Checkpoint: lt/le/gt/ge/eq/ne green per sub-step.
2. **Retire the MARK_AX engines.** Once (1) drives decode from the SE
   row, delete the CMP-flag writes in
   `_layer10_alu_ordering_engine_rules` + `_layer10_alu_eq_engine_rules`
   (keep only the eq decode-margin if still needed). These are the
   load-bearing +43/48-3 engines — retire ONLY after the SE decode is
   proven green. Checkpoint: full smoke.
3. **MUL fallout.** Add `SE_OP_MUL` (liveness-allocated) + switch
   l11/l12 MUL readers to `SE_ALU_LO/HI`, `SE_AX_CARRY_LO/HI`,
   `SE_OP_MUL`. Co-fix the L15/L25 OP_MUL OUTPUT-materialization
   corruptor (`project_mul_div_mod_arch_blocked`) — the remaining MUL
   blocker independent of the relay. Checkpoint: mul_basic /
   mul_overflow + targeted mul/cmp 1096 slice.

## Tools (read-only, spec_k=0, CPU)
- `tools/probe_l9_se_relay.py` (existing) — relay transmission.
- `tools/probe_relay_landing.py`, `probe_l9_relay_weights.py` (existing).
- `tools/probe_l9_relay_slope_check.py` (new) — per-block slope vector.
- `tools/probe_se_cmp_compute.py` (new) — SE-row CMP cascade + OUTPUT
  for lt/gt/eq/ne with a downstream SE row.
