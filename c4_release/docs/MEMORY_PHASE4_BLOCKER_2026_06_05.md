# Memory Phase 4 — BLOCKER (2026-06-05)

Attempted to land the Phase 4 composite-late-placement edits per
`docs/MEMORY_PHASE4_DESIGN_2026_06_05.md` (commit `8d999b53`) on top of
Phase 3 (`3423aff1` — split L10 carry_relay anchor into attn/ffn
siblings). Smoke regressed: **45/51 -> 44/51** (lost `test_shl_8bit`;
the 5 memory tests still failed). Backed out.

## Baseline + post-fix smoke

```
Baseline (Phase 3 head, no Phase 4 edits):
  45 passed, 6 failed
  FAILED: test_lea_basic
  FAILED: test_si_li_roundtrip
  FAILED: test_sc_lc_roundtrip
  FAILED: test_si_li_multiple_stores
  FAILED: test_si_li_overwrite
  FAILED: test_si_li_16bit_value

After Phase 4 diff (per docs/MEMORY_PHASE4_DESIGN_2026_06_05.md):
  44 passed, 7 failed
  FAILED: test_lea_basic
  FAILED: test_si_li_roundtrip          (unchanged)
  FAILED: test_sc_lc_roundtrip          (unchanged)
  FAILED: test_si_li_multiple_stores    (unchanged)
  FAILED: test_si_li_overwrite          (unchanged)
  FAILED: test_si_li_16bit_value        (unchanged)
  FAILED: test_shl_8bit                 NEW REGRESSION
```

## Root cause: the design doc's "Phase 3 prerequisite" assumption was wrong

Phase 4's verified data-flow argument hinges on `layer13_mem_addr_gather`
landing at L13. From the design doc:

> With Phase 3 pinning the renamed `_layer13_mem_addr_anchor` to
> `layer_idx=13` (per Phase 3's spec), `mem_addr_gather` lands at L13,
> producing ADDR_B* in-step for L14 to read.

But Phase 3 as actually landed (commit `3423aff1`) split
**`layer10_carry_relay`**, not **`_layer13_attn_dep_anchor`**. Phase 3's
own commit message documents this divergence:

> Note: the brief described renaming _layer13_attn_dep_anchor, but
> empirically that anchor is L13-internal (5 consumers, all L13 ops)
> and not the joint-anchor coupling Phase 4 needs unblocked.

So Phase 3 left `_layer13_attn_dep_anchor` (the dependency `mem_addr_gather`
binds to) unchanged. Verified post-Phase-3, post-Phase-4-diff: that
anchor still lands at **layer 16**, not layer 13.

Verified placements (`alu_mode='efficient'`, Phase 4 diff applied):

```
layer=13  layer10_carry_relay     (attn) — Phase 3 anchor
layer=13  _layer10_attn_anchor    (attn) — Phase 3 sibling
layer=14  _layer11_ffn_dep_anchor (ffn)
layer=15  _layer12_ffn_dep_anchor (ffn)
layer=16  _layer13_attn_dep_anchor (attn)   <-- mem_addr_gather binds here
layer=16  layer13_mem_addr_gather (block, target=_layer13_attn_dep_anchor)
layer=17  layer14_mem_generation  (attn)
layer=17  _layer14_attn_dep_anchor (attn)
layer=18  layer15_memory_lookup   (attn)
layer=19  layer16_lev_routing     (ffn)     <-- shift composite anchored here
layer=20  l13_alu_shift_bdtoge    (ffn)
layer=21  l13_alu_shift_precompute (ffn)
layer=22  l13_alu_shift_select    (ffn)
layer=23  l13_alu_shift_getobd    (ffn)
```

mem_addr_gather at L16 still runs BEFORE mem_generation (L17) and
memory_lookup (L18), so the data-flow chain isn't visibly broken at the
DSL level. But the 5 memory tests don't recover. Hypotheses for why:

1. The actual mem-addr dataflow expects mem_addr_gather at the legacy
   block 13 (where `_set_layer13_mem_addr_gather` ran historically), and
   running it at L16 puts it after several L13-L15 ops that consume
   ADDR_B*/MEM_VAL_* cross-step. Need to verify which ops actually read
   ADDR_B0_LO same-step at L14 and check whether they exist at L13-L16.
2. The `same_layer_as`/`after` chain from `layer10_carry_relay` (L13) to
   `_layer13_attn_dep_anchor` is broken. `layer10_carry_relay` is a
   topology anchor pinned at L13 only through Phase 3's `phase=10.0`
   shared with `_layer10_attn_anchor`; `_layer13_attn_dep_anchor` only
   has `requires={"after": "_layer12_ffn_dep_anchor"}` and lands at the
   earliest layer past L15. Compiler placed it at L16 — the doc's
   "L13" assumption was never satisfied.

## Why the shift composite at B29 broke `test_shl_8bit`

Anchoring the 4 composite stages and the install to `layer16_lev_routing`
moved the composite from B25 (under `_layer13_attn_dep_anchor`@L16,
which was the V3 known-good placement) to B29 (under
`layer16_lev_routing`@L19). The composite still BAKES (block[29].ffn is
`ALUShiftComposite`), and the install runs `block.post_ops.append`. But
running SHL/SHR at B29 vs B25 changes which dims are live in the
residual stream at the time `ALUShiftComposite.forward` reads
`ALU_LO`/`AX_CARRY_LO`/`AX_CARRY_HI`. `test_shl_8bit`'s expected
`0xFF << 1 == 0x1FE & 0xFF == 0xFE = 254` came out 0 — the AX_CARRY
chain was already cleared or mutated by intervening L17-L18 ops.

## What needs to change before Phase 4 can land

**Phase 3 needs to actually split the L13 anchor**, not the L10 anchor.
Per the original Memory Phase 2 blocker doc, the cluster needed two
independent splits:

- L10: split `layer10_carry_relay` into attn-side + ffn-side anchors
  (Phase 3 as landed). This unblocks moving the L10 attn family out of
  block[13] eventually.
- L13: split `_layer13_attn_dep_anchor` into:
  - `_layer13_mem_addr_anchor` (kind="attn", `layer_idx=13`) — for
    `layer13_mem_addr_gather` to bind to, pinned at L13.
  - `_layer13_ffn_dep_anchor` (kind="ffn", existing constraints) — for
    the other 4 L13-internal consumers.

Without the L13 split, Phase 4's design premise (mem_addr_gather at L13,
freed because block[13].ffn no longer holds the composite) never
materializes — block[13].ffn was never the constraint; the constraint
was the anchor placement.

Additionally, the shift composite needs a NEW anchor that pins it at a
layer where the AX_CARRY/ALU_LO residuals are still live. Options:

A. Re-target install + 4 stages to `_layer13_ffn_dep_anchor` (the new
   FFN-side split) and let it pin at L14/L15 — but verify the AX_CARRY
   chain survives intervening ops.
B. Introduce a dedicated `_l13_alu_shift_anchor` topology anchor with
   `requires={"after": "_layer12_ffn_dep_anchor", "before":
   "_layer14_attn_dep_anchor"}` to pin it between L15 and L17.

## Files

* `c4_release/docs/MEMORY_PHASE4_DESIGN_2026_06_05.md` — original
  Phase 4 design (premise stale).
* `c4_release/docs/MEMORY_CLUSTER_FIX_PLAN_2026_06_05.md` — master plan.
* `c4_release/docs/SMOKE_MEMORY_FIX_ATTEMPT_V4_20260605.md` — V4 attempt
  (head_0 contest failure mode; doc claimed Phase 3 resolved it).
* `c4_release/neural_vm/unified_compiler/ops/l13_ops.py:394-459` —
  `_layer13_attn_dep_anchor` (5 L13-internal consumers, needs the actual
  split per Phase 3's deferred work).
* `c4_release/neural_vm/unified_compiler/ops/alu_ops.py:22-229` — the
  5 ops Phase 4 was supposed to mutate.

## Recommendation

Land an **actual L13 anchor split** (call it Phase 3b) before retrying
Phase 4. The L10 anchor split was useful but does not unblock the
memory cluster's L13 placement issue. Once the L13 anchor splits into
mem_addr + ffn-internal siblings, the Phase 4 diff in
`docs/MEMORY_PHASE4_DESIGN_2026_06_05.md` should re-apply cleanly, with
two amendments:

1. `layer13_mem_addr_gather`'s `target_op_name` retargets to the new
   `_layer13_mem_addr_anchor` (pinned at L13 via `layer_idx=13`).
2. The shift composite's `requires["after"]` may need a different
   target than `layer16_lev_routing` — pick something that keeps the
   composite at L13-L16 range so AX_CARRY residuals are still live
   (B25-B27 region per V3 verified topology).
