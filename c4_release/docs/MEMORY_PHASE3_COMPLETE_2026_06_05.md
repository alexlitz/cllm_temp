# Memory cluster Phase 3 — anchor split landed (2026-06-05)

Per `docs/MEMORY_CLUSTER_FIX_PLAN_2026_06_05.md` Phase 3 (re-scoped per
`docs/MEMORY_PHASE2_BLOCKER_2026_06_05.md` §"Why Phase 2 needs Phase 3
first"), this commit splits the joint L10 anchor `layer10_carry_relay`
into two purpose-specific topology anchors so the L10 attn-bake family
can migrate physical layers independently of the L10 FFN family
(`layer10_alu` + post_op composite installs).

## What landed

1. **NEW topology anchor `_layer10_attn_anchor`** (in `ops/l10_ops.py`).
   `kind="attn"`, `declarative_authority="topology_anchor"`, shares the
   `requires["after"]: layer9_marker_suppress` placement chain with the
   existing `layer10_carry_relay` anchor, plus an explicit `phase=10.0`
   and `requires["same_layer_as"]: layer10_carry_relay` so both anchors
   co-resolve to the same physical layer slot. Registered in
   `ops/all_core_ops.py` right after `make_layer10_carry_relay_op()`.
2. **`layer10_carry_relay` retargeted**: explicit `phase=10.0` added so
   the layer-assignment slot tracker shares its (layer, kind="attn")
   slot with the sibling anchor (otherwise the second anchor bumps to
   layer+1). No other change — still anchors via
   `requires["after"]: layer9_marker_suppress`.
3. **6 L10 attn-bake ops re-pointed** to `target_op_name=
   "_layer10_attn_anchor"`:
     * `layer10_carry_relay_bake` (head 0 CARRY)
     * `layer10_byte_passthrough_bake` (head 1 AX byte)
     * `layer10_sp_byte_passthrough_bake` (head 2 SP byte)
     * `layer10_psh_stack0_passthrough_bake` (head 3 PSH STACK0)
     * `layer10_bp_byte_passthrough_bake` (head 7 BP byte)
     * `layer10_stack0_byte_relay_bake` (heads 4/5/6 stack0 relay)
4. **8 FFN-side / post_op-side / model-replacement ops UNCHANGED**:
     * `layer10_alu` (1846 FFN units)
     * `l10_post_op_attach`
     * `l10_alu_postop_attach`
     * `l10_alu_divmod_{bdtoge,longdiv,getobd,install}`
     * `efficient_l10_andorxor_wrap`
     * `null_terminator_detection` (L10 FFN unit 1864)

   All keep `target_op_name="layer10_carry_relay"`.
5. **`CROSS_STEP_DOCUMENTED_SAFE` entry added** for the new anchor's
   `CARRY.*.-1` read (mirroring the existing `layer10_carry_relay`
   entry — topology anchors with no weight bake are intentionally
   cross-step-safe).

## Why this is the right Phase 3

The brief that triggered this work (and the
`MEMORY_CLUSTER_FIX_PLAN_2026_06_05.md` Phase 3 §) described the split
as "rename `_layer13_attn_dep_anchor`" + add a new attn anchor.
Empirically this description does not match the actual coupling:

* `_layer13_attn_dep_anchor` (in `l13_ops.py`) has 5 consumers, all of
  them L13-internal (`layer13_mem_addr_gather`, `layer13_shifts`,
  `l13_alu_postop_attach`, `l13_alu_shift_install`, one composite
  stage). It is already L13-specific; no joint-anchor coupling to split.
* `layer10_carry_relay` (in `l10_ops.py`) has 14 consumers spanning
  both the L10 attn family (6 head bakes) AND the L10 FFN family
  (`layer10_alu` + ~8 post_op / install ops). This is the joint anchor
  the Phase 2 blocker doc identified as the actual prerequisite.

`MEMORY_PHASE2_BLOCKER_2026_06_05.md` §"Why Phase 2 needs Phase 3 first"
makes this explicit: "split `layer10_carry_relay` into separate attn
and FFN anchors so the attn family can migrate independently of the
FFN family". Phase 3 implements that split; the brief's `_layer13_*`
naming is an artifact of an earlier draft that conflated the two L10/
L13 cluster issues.

## Verification

### Smoke (CUDA_VISIBLE_DEVICES=1, 51 selected tests)

```
$ PYTHONPATH=. python -m pytest c4_release/tests/test_smoke.py --no-header -q --tb=no
45 passed, 6 failed, 1 deselected in 153.11s
```

Failures (matches baseline exactly — see
`MEMORY_PHASE2_BLOCKER_2026_06_05.md` §"Smoke baseline"):
* `TestSmokeAddress::test_lea_basic`
* `TestSmokeMemory::test_si_li_roundtrip`
* `TestSmokeMemory::test_sc_lc_roundtrip`
* `TestSmokeMemory::test_si_li_multiple_stores`
* `TestSmokeMemory::test_si_li_overwrite`
* `TestSmokeMemory::test_si_li_16bit_value`

Phase 3 is byte-identity-preserving (smoke-neutral, as the plan
required for a metadata-only refactor): no test moved in either
direction.

### Layer assignment (post-split)

Both anchors resolve to the same physical layer L13:

```
Layer 13: [_layer10_attn_anchor, layer10_carry_relay]
Layer 14: [_layer11_ffn_dep_anchor]
```

All 6 attn-bake ops + the 8 FFN-side ops bake into block 13 (the
post-`_expand_wrapper_blocks` block resolution remains unchanged
from pre-split). The split is therefore a no-op at runtime today; its
value is purely architectural — a future Phase 2 retry can drop the
`same_layer_as` constraint on `_layer10_attn_anchor` and retarget its
`requires["after"]` to a different `kind="ffn"` predecessor to move
the L10 attn family to a separate physical layer without dragging the
1846-unit `layer10_alu` FFN.

### Slot-conflict registry

Both anchors are `declarative_authority="topology_anchor"` — they
contribute no slot ids per `slot_registry.derive_slot_ids_for_op`
(topology anchors return early). The compile path's
`_run_slot_conflict_scan` records 0 conflicts. Adding a second attn
anchor at the same layer is admissible by construction.

### Cross-step safety

The new anchor inherits the same `reads={"CARRY.*.-1"}` /
`writes={"CARRY"}` shape as `layer10_carry_relay`. Added to
`CROSS_STEP_DOCUMENTED_SAFE` (in `full_vm_compiler_dynamic.py`) with a
docstring referencing this Phase 3 doc. No new entries in
`CROSS_STEP_BASELINE_ALLOWLIST` (the ratchet stays).

## Files touched

* `c4_release/neural_vm/unified_compiler/ops/l10_ops.py` — added
  `make_layer10_attn_anchor_op()`; added `phase=10.0` on
  `layer10_carry_relay`; retargeted 6 attn-bake ops.
* `c4_release/neural_vm/unified_compiler/ops/all_core_ops.py` —
  registered the new anchor right after `make_layer10_carry_relay_op()`.
* `c4_release/neural_vm/unified_compiler/full_vm_compiler_dynamic.py` —
  added `('_layer10_attn_anchor', 'CARRY.*.-1')` to
  `CROSS_STEP_DOCUMENTED_SAFE`.
* `c4_release/docs/MEMORY_PHASE3_COMPLETE_2026_06_05.md` — this doc.

## Unblocks

Phase 2 retry (move attn family off L10 layer) and Phase 4 (composite
late-placement re-application — see
`docs/MEMORY_PHASE4_DESIGN_2026_06_05.md`). The L10 FFN family
(`layer10_alu` + post_ops) can now stay anchored at L10 via
`layer10_carry_relay` while the attn family migrates to a different
layer via `_layer10_attn_anchor`.
