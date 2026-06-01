# Path to fully dynamic — concrete plan

End state goal: **adding a new corrective op is `Op(reads=..., writes=..., produces=..., consumes_fresh=..., requires=..., claims=...)` — compiler picks layer, slot, unit, head index automatically; trained weights preserved via `pin=` on existing ops; no `phase=N.M` field.**

## Where we are (post 2026-06-01 session)

| Surface | Coverage |
|---|---|
| FFN unit allocation | 12 of 17 layers migrated; 5 in flight (~100% after wave) |
| Dim slot allocation | dim_registry → Allocator migrated; existing dims pinned, new dims can auto-fit |
| Layer ordering schema | B10 op-name `requires["after"]`/`requires["same_layer_as"]` landed |
| Hybrid compile path | `compile_full_vm_dynamic` byte-identical to static (B11) |
| B14 strict mode flag | Scaffold landed (`strict=False` default) |
| B15 `phase=N.M` deprecation | Triple-gated off |
| B13 CI gate | Scaffold landed (skipped by default) |
| Attention head allocation | Scaffold landed; 0/62 sites migrated |
| Test fixtures | Hardcode `OUTPUT_LO+N` / `OUTPUT_HI+N` — break under B9 rename |
| `phase=N.M` field on Operation | Still present and actively used |
| OUTPUT_HI back-edges | 28 cross-step closed by B9; ~101 SAME_STEP remain |

## Phase 1 — close immediate gaps (~1 day, parallel)

| Unit | What | Effort | Conflicts |
|---|---|---|---|
| **1A B9 SAME_STEP follow-up** | Annotate ~101 same-step OUTPUT_HI back-edges with per-reader `requires["after"]`. Affects layer6_routing_ffn, layer7_operand_gather, l10_post_ops_combined, layer14_clear_output_corruption, layer15_store_stack0_sp_byte0_addr, tail_bit32_result_correction | 2h | L6/L10/L15 unit_counter currently editing same files |
| **1B B12 wave 2** | Backfill 4 ops gated on B9: `layer10_sp_byte_passthrough`, `layer15_si_mem_addr0_from_stack0`, `layer14_clear_addr_key_pollution`, `layer4_sp_to_addr_key` | 3h | L10/L15 unit_counter |
| **1C Test fixture B9 rename pass** | Mechanical update of test files: `OUTPUT_LO+N`/`OUTPUT_HI+N` → use B9's renamed dim names | 3h | None — test files only |
| **1D `dim_registry_dynamic.py` update** | Mirror the 17 _PIN slots from registry-completeness pass (commit 57c6a04) into the dynamic mirror | 30m | None |
| **1E B11 strict_mode parameter audit** | Confirm B14's strict_mode survived the allocator-integration merge (may have been dropped); re-add if missing | 30m | None — full_vm_compiler_dynamic.py |

## Phase 2 — flip B14 strict mode (~1 day)

| Unit | What | Prereq |
|---|---|---|
| **2A** Re-run `analyze_scheduler.py`, confirm `dep_graph_cycle_member == 0` and `phase_required_but_undeclared == 0` | Phase 1A+1B complete |
| **2B** Set `compile_full_vm_dynamic(strict=True)` default | 2A passes |
| **2C** 1096 sweep validation under strict mode | 2B lands |
| **2D** Flip B13 CI gate ON (`B13_GATE_ENABLED=1` in CI) | 2C green |

## Phase 3 — remove `phase=N.M` field (~1 day)

| Unit | What |
|---|---|
| **3A** Migrate every op to drop `phase=N.M` (mostly mechanical — just remove the field, deps already pin order) |
| **3B** Remove `phase` field from `Operation` dataclass + tiebreaker logic in `layer_compiler.py` |
| **3C** Remove B15 deprecation warning + test (no longer needed) |
| **3D** Update docs / migration plan to mark phase retired |

## Phase 4 — attention head_idx migration (~1-2 weeks, parallel)

Same pattern as FFN unit migration. ~14 layers with attention heads, one agent per layer-family:

| Layer | Approximate `head_idx=N` sites |
|---|---|
| L0 / L1 / L2 / L3 | ~4 each |
| L4 / L7 | ~8 each |
| L5 / L6 | ~6 each |
| L8 / L9 / L10 / L13 | ~5 each |
| L14 / L15 / L16 | ~4 each |
| **Total** | ~62 |

Each migration: build `_LN_HEAD_LAYOUT` table, instantiate `AttentionHeadAllocator(layer_max_heads=8)` locally per bake_fn, stash on `block.attn._lN_head_allocator`, pin every existing head_idx. Byte-identical via head-by-head weight comparison.

Parallelization: 14 layers, parallel, ~1-2 days each → ~3-4 days elapsed if dispatched aggressively.

## Phase 5 — final cleanup (~half day)

| Unit | What |
|---|---|
| **5A** Layer count = DAG depth (remove the 17 hardcode; derive in dynamic compile) |
| **5B** Verify allocator `pin=None` works for new dims/units/heads end-to-end (likely already works) |
| **5C** Documentation: write "how to add a new corrective op" guide |
| **5D** Mark DYNAMIC_SCHEDULER_MIGRATION_PLAN.md as complete |

## Total budget

- Phase 1: ~1 day (parallel)
- Phase 2: ~1 day (sequential within phase)
- Phase 3: ~1 day
- Phase 4: ~3-4 days (parallel)
- Phase 5: ~half day

**~6-7 days elapsed** to fully dynamic. ~3 weeks of agent-time total but heavily parallelizable.

## Acceptance test

```python
# After all phases:
op = Op(
    name="my_corrective",
    reads={"OUTPUT_HI_THIS_STEP"},
    writes={"STACK0_BYTE0"},
    produces={"STACK0_byte0_lo": "is_byte"},
    requires={"after": "layer14_clear_output_corruption"},
    claims=frozenset({...}),
    bake=lambda block, dim_positions, ...: ...,
)
# No phase=, no head_idx=, no unit=, no slot=
# compile_full_vm_dynamic(strict=True) succeeds
# Byte-identical to before (modulo the new op's writes)
```
