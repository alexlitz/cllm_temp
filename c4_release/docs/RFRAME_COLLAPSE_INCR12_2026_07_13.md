# R-FRAME collapse INCR-1 + INCR-2 — LAND note (byte-identical)

**Date:** 2026-07-13  **Branch:** `rframe-collapse-incr12` (off `main` `cdb6c23f`).
**Golden gate (the authoritative byte-identity check for these weight-emitting
generators):**

```
CUDA_VISIBLE_DEVICES="" python tools/_isa_golden_hash.py
state_dict_sha256 == e50521f32b0ed952d5730f79b63adb8c4c78f4d4f0466d3bcbaa354bb3c90e86  (e50521f3)
```

Verified UNCHANGED **before** the work (baseline) and **after** INCR-1 (commit
`1af1f0f0`). INCR-2 was already landed on `main` (commit `7084aeb0`, see §2) and
its byte-identity is likewise proven by the same golden hash being the current
baseline. Design source: `docs/RFRAME_EMITTER_SCOPE_2026_07_13.md` §3.b / §4.c
(the INCR-1/INCR-2 de-risked plan).

---

## 1. INCR-1 — passthrough glue collapse (LANDED, byte-identical)

**What replicated.** The four L10 register byte-passthrough heads (AX/SP/BP/PC)
each re-instantiated three near-clone GLUE functions around the ALREADY-shared
`_byte_passthrough_chain_spec` template (whose `for idx in range(16)` V/O nibble
copy `CLEAN_EMBED_LO/HI[idx] -> OUTPUT_LO/HI[idx]` is identical for every
register):

| glue family (`neural_vm/unified_compiler/ops/l10_ops.py`) | what varied per register |
|-----------------------------------------------------------|--------------------------|
| `make_layer10_{byte,sp,bp,pc}_byte_passthrough_bake_op` (4× ~65 LOC `Operation` builders) | `name`, `spec_fn`, alibi/head slot, `reads` set, `smoke_tests`, PC's `requires` |
| `_layer10_{byte,sp,bp,pc}_byte_passthrough_ir` (4× `compiler_ir_factory`) | `spec_fn`, IR head name |
| `_bake_layer10_{byte,sp,bp}_byte_passthrough_head` (3× head bakes) | `spec_fn`, alibi index |

Everything else in those functions was byte-for-byte identical: the per-bake
`_allocate_layer10_attention_heads()` allocator pin, the
`Primitives.generate_attention_head(attn, spec_fn(proxy, S), HD)` lowering, the
single `alibi_slopes[slot] = 1.0` tie-break write, the
`(10, "attn_W_v", "{slot}_{k}", "CLEAN_EMBED_LO/HI+{k}")` V-claims, and the
`target_op_name/migrated/declarative_authority/writes={OUTPUT_LO,
OUTPUT_HI_THIS_STEP}/spec_section` Operation metadata.

**The collapse.** Introduced three shared, table-driven lowerings — each public
`make_*` / `_ir` / `_bake_*` function is now a one-line call over per-register
DATA:

- `_make_r_frame_passthrough_bake_op(op_name, head_slot, spec_fn, ir_factory,
  reads, smoke_tests, requires=None)` — the single `Operation` builder. The head
  slot drives BOTH the `alibi_slopes` index and the V-claim tuple key, so the two
  are guaranteed consistent. `requires` is passed only for PC (the
  `l10_attention_resize` ordering pin); omitting it yields the default empty dict
  the non-PC ops carried, so their Operations are unchanged.
- `_r_frame_passthrough_ir(dim_positions, spec_fn, ir_name)` — the single
  `compiler_ir_factory`.
- `_bake_r_frame_passthrough_head(attn, BD, S, HD, spec_fn, alibi_idx)` — the
  single head bake (kept because `tests/test_declarative_l10_passthrough_specs.py`
  imports the three public `_bake_layer10_*` names directly).

**What stayed inline (correctly).** The four `_layer10_*_byte_passthrough_head_spec`
functions are the genuine per-register SPEC logic — the AX LI-reload query blocks
(slots 39-48 + the campaign slot-82/83 SI/LI CAM), BP's `top_store_query` STACK0
route, SP's marker-carry blockers, PC's non-PC-marker suppress. Per
`RFRAME_EMITTER_SCOPE` §2/§4.b these are the per-register `carry_suppress_ops` /
discriminator DATA that "must be transcribed verbatim, not re-derived"; they are
NOT boilerplate and are left untouched. The `_byte_passthrough_chain_spec`
invocation kwargs (`is_byte_strength`, `q0_threshold`, `gate_extras`,
`suppress_op_dims`, …) already ARE the compact per-register table those specs
read — the byte-identical collapse target was the surrounding glue.

**Proof.**
- golden `e50521f3` UNCHANGED after commit `1af1f0f0`.
- `tests/test_declarative_l10_passthrough_specs.py`: per-test outcomes IDENTICAL
  to clean `main` (verified in a sibling `main` worktree) — the same 5
  pre-existing failures (the LEGACY imperative `_set_layer10_*` helpers in
  `vm_step.py` have drifted from the declarative specs; unrelated to this change)
  and the same 3 passes. No new failure introduced.
- `tests/test_l10_bp_passthrough_isolated.py`, `test_compile_cross_step_safety.py`,
  `test_l10_tail_correction.py`: green.

**Net LOC:** `l10_ops.py` **239 removed − 156 added = −83 LOC**.

*Scope note vs the doc's ~900 estimate.* The `RFRAME_EMITTER_SCOPE` §3.b ~900-LOC
figure for INCR-1 counted the per-register QUERY BLOCKS (the `_*_head_spec`
bodies, e.g. AX's 265-LOC LI-reload block) as collapsible. In practice those are
genuine per-register spec logic (§4.b point 2 of that same doc says the
fresh-vs-carry gates / discriminators cannot be inferred and must stay verbatim),
so the BYTE-IDENTICAL win is the glue: −83 LOC of pure clone boilerplate removed
while unifying the 4 registers behind one table-driven builder. The remaining
per-register head-spec LOC is not byte-identically removable — it belongs to the
INCR-3 (flag-gated, verdict-equivalent) tail-guarantee collapse where the value
bands become correct-by-construction.

---

## 2. INCR-2 — carry migration (ALREADY LANDED on main, byte-identical)

INCR-2 (migrate the AX-byte1 + STACK0-byte0 cross-step carries onto the
`cross_step_carry(CrossStepCarrySpec)` generator, using `BP_SAVE_PREV` as the
byte-identity proof) was **already delivered on `main`** by:

```
7084aeb0  feat(isa-dsl): migrate AX byte-1 + STACK0 byte-0 carries onto cross_step_carry
          (2026-06-16; ancestor of main cdb6c23f)
```

Current state of `neural_vm/unified_compiler/ops/l11_ops.py` (verified):

| carry | spec | generator call |
|-------|------|----------------|
| ENT saved-BP | `_BP_SAVE_CARRY_SPEC` (l11:163) | `_BP_SAVE_CARRY_BUNDLE = cross_step_carry(_BP_SAVE_CARRY_SPEC)` (l11:223) |
| AX byte-1 | `_AX_BYTE1_CARRY_SPEC` (l11:1358) | `_AX_BYTE1_CARRY_BUNDLE = cross_step_carry(_AX_BYTE1_CARRY_SPEC)` (l11:1426) |
| STACK0 byte-0 | `_STACK0_B0_CARRY_SPEC` (l11:2603) | `_STACK0_B0_CARRY_BUNDLE = cross_step_carry(_STACK0_B0_CARRY_SPEC)` (l11:2658) |

All three carry HEADS and DUMP cores are now generator-produced — the consuming
ops call `_*_CARRY_BUNDLE.carry_head_spec_builder(...)` / `.dump_rules_builder(...)`
/ `.precursor_ops_builder(...)`. The generator was generalized over BP's
layout-only form with the fields already present in `CrossStepCarrySpec`
(`position_source="mixed"` + `registry_dims`, explicit `head_q/k/v/o` HeadWrite
directives with `slot_stride`, explicit `dump_blocks`/`dump_blocks_off`,
`precursors`, `register_band=False`). This is the SAME `RFRAME_EMITTER_SCOPE`
§1.b migration and its ~568-LOC INCR-2 win; it is reflected in the current golden
baseline.

**Proof.** golden `e50521f3` is the current baseline (i.e. the migrated build IS
the golden), and `tests/test_compile_cross_step_safety.py` is green. No further
code change is required for INCR-2. The dump-REPOPULATE FFN wrappers
(`make_ax_byte1_dump_repopulate_op`, `make_b1_to_output_op`,
`_bp_save_dump_repopulate_rules`) are the "value -> OUTPUT re-decode" glue that
`RFRAME_EMITTER_SCOPE` §3.b assigns to INCR-3, not INCR-2, so they are correctly
out of scope here.

---

## 3. Bottom line

| increment | status | golden | net LOC |
|-----------|--------|--------|---------|
| INCR-1 passthrough glue collapse | LANDED this branch (`1af1f0f0`) | `e50521f3` UNCHANGED | −83 (l10_ops.py) |
| INCR-2 carry migration | ALREADY on main (`7084aeb0`) | `e50521f3` (is the golden) | ~−568 (previously landed) |

Both increments are BYTE-IDENTICAL to golden `e50521f3` (these ARE
weight-emitting generators, so the golden hash is the valid gate). INCR-1 is
landable to `main`. INCR-2 needs no new work — it was completed by a prior
commit that is already an ancestor of `main`; the `RFRAME_EMITTER_SCOPE` doc
(also dated 2026-07-13) described it as pending against an earlier tree state.

The larger `RFRAME_EMITTER_SCOPE` −1,470-LOC combined target assumed INCR-2 was
still unmigrated. With INCR-2 already banked, the incremental win THIS branch
adds is INCR-1's −83 LOC; the full R-FRAME −3.9k-to-−4.6k envelope remains gated
behind INCR-3 (the flag-gated, verdict-equivalent L25 tail-guarantee collapse,
`C4_R_FRAME_TAIL`), which moves the golden hash by design and is out of scope for
the byte-identical increments.
