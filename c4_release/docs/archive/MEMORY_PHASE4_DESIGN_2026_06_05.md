# Memory cluster Phase 4 — composite scaffolding design (2026-06-05)

Per `docs/MEMORY_CLUSTER_FIX_PLAN_2026_06_05.md`, Phase 4 re-applies V4's
composite-late-placement edits **after** Phases 1-3 have decoupled the
L10/L13 attn anchors. This doc spells out the exact diff so it can land
in one commit immediately after Phase 3 ships the split anchor.

Phase 3 is **not yet landed** — DO NOT apply this diff until the new
`_layer13_mem_addr_anchor` (or equivalent split-anchor naming) exists.

## Why Phase 4 needs Phase 3 first

V4 (`docs/SMOKE_MEMORY_FIX_ATTEMPT_V4_20260605.md`) landed *exactly the
composite scaffolding below*, but **kept** the joint
`_layer13_attn_dep_anchor`. Smoke went from 26/51 → 13/51 (38 fails).
Cause: `layer10_carry_relay_bake` (target=`layer10_carry_relay`) and
`layer13_mem_addr_gather` (target=`_layer13_attn_dep_anchor`, pinned at
L13) both contested `block[13].attn.head_0`. Carry write got overwritten
by the ADDR-gather; ADD/SUB byte-carry broke; every AX-write test failed.

Phase 3 splits the anchor so carry_relay and mem_addr_gather no longer
collide on head_0 (Phase 2's head-allocation work, then Phase 3 wires
the new anchor names).

## Files + lines

* `c4_release/neural_vm/unified_compiler/ops/alu_ops.py:22-229` —
  `make_alu_shift_composite_ops` (5 ops: 4 ffn stages + 1 install).
* `c4_release/neural_vm/unified_compiler/full_vm_compiler_dynamic.py:1015-1500`
  — `CROSS_STEP_DOCUMENTED_SAFE` dict.
* `c4_release/neural_vm/unified_compiler/ops/l13_ops.py:394-459` —
  `_layer13_attn_dep_anchor` (read-only; Phase 3 owns the rename).

## The diff (apply only AFTER Phase 3)

### 1. `alu_ops.py` — 4 stages: SSA-rename ALU_LO + add `requires`

For each of `l13_alu_shift_bdtoge`, `l13_alu_shift_precompute`,
`l13_alu_shift_select`, `l13_alu_shift_getobd` (lines 38-63, 72-96,
105-129, 138-170), inside the `Operation(...)` constructor:

**OLD** (per stage, lines 40-41 / 74-75 / 107-108 / 140-141):
```python
reads={"MARK_AX", "ALU_LO", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI",
       "OP_SHL", "OP_SHR"},
```

**NEW**:
```python
reads={"MARK_AX", "ALU_LO.*.-1", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI",
       "OP_SHL", "OP_SHR"},
```

Add (after the existing `kind="ffn",` line, around line 43/77/110/143):
```python
requires={"after": "layer16_lev_routing"},
```

### 2. `alu_ops.py` — install op: post_ops.append + new target

**OLD** (line 176):
```python
def bake(block, dim_positions, S):
    if builder.composite is None:
        return
    block.ffn = builder.composite
```

**NEW**:
```python
def bake(block, dim_positions, S):
    if builder.composite is None:
        return
    block.post_ops.append(builder.composite)
```

**OLD** (line 198):
```python
target_op_name="_layer13_attn_dep_anchor",
```

**NEW**:
```python
target_op_name="layer16_lev_routing",
```

**OLD** (line 204):
```python
requires={"after": "l13_alu_shift_getobd"},
```

**NEW** (keep intra-target ordering; lev_routing is also the host):
```python
requires={"after": "l13_alu_shift_getobd"},
```
(No change — kept for clarity; the post_ops install runs after the
composite is fully assembled by the 4 stage bakes.)

### 3. `produces` sentinel update on all 5 ops

**OLD** (lines 55, 88, 121, 162, 213):
```python
produces={'__module_replacement': 'L13.ffn[ALUShiftComposite]'},
```

**NEW**:
```python
produces={'__module_replacement': 'L13.post_ops[ALUShiftComposite]'},
```

### 4. `full_vm_compiler_dynamic.py` — 4 CROSS_STEP_DOCUMENTED_SAFE entries

Insert into `CROSS_STEP_DOCUMENTED_SAFE` (alphabetical clustering near
line 1148 `_layer11_ffn_dep_anchor` ALU_LO precedent):

```python
('l13_alu_shift_bdtoge', 'ALU_LO.*.-1'):
    "ALU_LO is cross-step relative to the LayerCompiler scope: the "
    "stage is anchored to layer16_lev_routing via requires['after'], "
    "but ALU_LO writers (L8/L9 ALU) still run before the host. The "
    "composite's runtime forward reads ALU_LO same-step at the host "
    "block; the alias is declarative-only. Mirrors _layer11_ffn_dep_"
    "anchor pattern at line 1148. See ops/alu_ops.py:22-72.",
('l13_alu_shift_precompute', 'ALU_LO.*.-1'):
    "Same design as l13_alu_shift_bdtoge above; precompute stage of "
    "the same composite. See ops/alu_ops.py:65-96.",
('l13_alu_shift_select', 'ALU_LO.*.-1'):
    "Same design; select stage. See ops/alu_ops.py:98-129.",
('l13_alu_shift_getobd', 'ALU_LO.*.-1'):
    "Same design; final GE->BD stage. See ops/alu_ops.py:131-170.",
```

## Expected placement after Phase 4 lands (with Phase 3 in place)

Per V4's verified topology trace (pre-expansion):

```
B13: layer10_carry_relay (attn anchor) + _layer13_mem_addr_anchor  ← Phase 3 split
B14: _layer11_ffn_dep_anchor + _layer14_attn_dep_anchor + layer14_mem_generation
B15: _layer12_ffn_dep_anchor + layer15_memory_lookup
B16: layer16_lev_routing
B17: l13_alu_shift_bdtoge       (post-lev_routing — late, via after=)
B18: l13_alu_shift_precompute
B19: l13_alu_shift_select
B20: l13_alu_shift_getobd
B21: l10_post_ops_combined
B22: post_l9_bz_bnz_pc_override
```

Post-`_expand_wrapper_blocks`: composite lands at **B26** (verified in V4).
`layer13_mem_addr_gather` lands at **L13** (memory-cluster win).

## Expected failure modes if Phase 3 isn't done yet

**DO NOT LAND THIS DIFF BEFORE PHASE 3.** If applied prematurely:

1. **Head_0 contest** (V4's failure mode, 38/51 smoke fails). With the
   joint `_layer13_attn_dep_anchor` still owning both `layer10_carry_relay`
   and `layer13_mem_addr_gather`, `layer13_mem_addr_gather` writes heads
   0-2 ADDR-gather over carry_relay's head 0 carry-relay write. Result:
   L8 ADD/SUB byte carry breaks, every AX-write test returns 0.
2. **Slot conflict** is *not* expected (A2 `post_ops.append` resolves it
   — V3 verified). If you see "two ops want block[13].ffn" the diff was
   applied incorrectly (e.g. the install bake still does `block.ffn = `).
3. **B15 vs B25/B26 placement**: with the `after=layer16_lev_routing` on
   all 4 stages + install, V4 verified composite lands at B26 (1 block
   after V3's known-good B25). Without those `after=` constraints, the
   composite lands at B15 (V3 mode) and breaks AX-write residual chain.

## Symbolic verification (DSLInterpreter)

Performed against the main-branch source at HEAD (not the worktree —
the worktree branch is on a divergent line and lacks
`make_layer13_attn_dep_anchor_op`).

Verified the LI/SI cycle data-flow chain:

```
layer13_mem_addr_gather  writes={ADDR_B0_HI, ADDR_B0_LO, ADDR_B0_VALID,
                                  ADDR_B1_HI, ADDR_B1_LO,
                                  ADDR_B2_HI, ADDR_B2_LO}
layer14_mem_generation   reads ={ADDR_B0_HI.*.-1, ADDR_B0_LO.*.-1, ...,
                                  MEM_VAL_B0..3, MEM_STORE, ...}
layer15_memory_lookup    reads ={MEM_VAL_B1..3, ADDR_KEY, MEM_ADDR_SRC,
                                  OP_LI, OP_LC, OP_SI, OP_SC, ...}
```

The chain is `mem_addr_gather (L13) -> mem_generation (L14) ->
memory_lookup (L15)`. With baseline anchor drift, `mem_addr_gather`
lands at L16, AFTER both `mem_generation` and `memory_lookup`, so the
ADDR_B* dims read by L14 are stale (prev-step) and SI/LI/SC/LC tests
fail.

With Phase 3 pinning the renamed `_layer13_mem_addr_anchor` to
`layer_idx=13` (per Phase 3's spec), `mem_addr_gather` lands at L13,
producing ADDR_B* in-step for L14 to read. The Phase 4 shift composite
moves OUT of L13 (`block[13].ffn` slot freed) and into L16 (via
`after=layer16_lev_routing`), avoiding the V2 slot conflict.

Verified that the 4 new SSA renames (`ALU_LO` → `ALU_LO.*.-1`) follow
the same pattern as `_layer11_ffn_dep_anchor`'s existing entry at
`full_vm_compiler_dynamic.py:1148` — the runtime forward in
`efficient_alu_neural.ALUShiftComposite.forward` reads ALU_LO same-step
at the host block; the alias is purely declarative to satisfy the
LayerCompiler dep-graph when the consumer is anchored late but the
producer (L8/L9 ALU) is anchored early.

## Verification gate (after Phase 4 lands)

1. Compile succeeds (34 runtime blocks post-expansion).
2. `compile_full_vm_dynamic(alu_mode='efficient')`:
   - `layer13_mem_addr_gather` resolves to layer 13.
   - `l13_alu_shift_install` resolves to layer 16.
   - `ALUShiftComposite` lands at B26 post-expansion.
3. Full smoke: 51/52 pass (5 memory tests recover; lea_basic still
   blocked on its own cluster).
4. No new `CROSS_STEP_BASELINE_ALLOWLIST` entries (the 4 new ones go to
   `CROSS_STEP_DOCUMENTED_SAFE`, not the migration backlog).

## Risk register

| Risk | Mitigation |
|---|---|
| Phase 3's anchor naming differs from this doc's assumption | The diff is decoupled from the anchor name — only the install op's `target_op_name` changes (to `layer16_lev_routing`, not the new anchor). Phase 3's rename does not touch this diff. |
| Phase 1's slot registry trips on the new post_ops.append slot claim | The append goes to `("post_ops", current_len)`; no existing op claims that slot. Phase 1 should not flag it. |
| Composite still non-idempotent at B26 (V3 doc's B3 hypothesis) | If smoke regresses identically to V4 (13/51) AFTER Phase 3 landed cleanly, fall through to Phase 5 (idempotence audit on `ALUShiftComposite.forward` per V3 doc B3). |
| `requires["after"]: layer16_lev_routing` drags transitive deps | V4 saw 21 cross-step warnings vs ~12 baseline; the 4 new `CROSS_STEP_DOCUMENTED_SAFE` entries cover the expected ones. Anything else is a real new edge that Phase 1's registry should surface. |

## Files referenced

* `c4_release/docs/MEMORY_CLUSTER_FIX_PLAN_2026_06_05.md` — master plan.
* `c4_release/docs/SMOKE_MEMORY_FIX_ATTEMPT_V4_20260605.md` — V4 attempt
  (this diff, applied prematurely; 13/51 regression).
* `c4_release/docs/SMOKE_MEMORY_FIX_ATTEMPT_V3_20260605.md` — V3 attempt
  (B15 placement regression; documents why `requires["after"]=` is
  necessary).
* `c4_release/docs/SMOKE_MEMORY_FIX_ATTEMPT_V2_20260605.md` — V2 attempt
  (slot conflict diagnosis; motivates A2 `post_ops.append`).
* `c4_release/neural_vm/unified_compiler/ops/alu_ops.py:22-229` — the
  5 ops Phase 4 mutates.
* `c4_release/neural_vm/unified_compiler/full_vm_compiler_dynamic.py:1015-1500`
  — `CROSS_STEP_DOCUMENTED_SAFE` dict to extend.
* `c4_release/neural_vm/unified_compiler/ops/l13_ops.py:394-459` —
  current joint anchor (Phase 3 splits; this diff does NOT touch).
* `c4_release/neural_vm/efficient_alu_neural.py:1486-1554` —
  `ALUShiftComposite.forward` (the same-step ALU_LO reader rationale
  for the cross-step alias).
