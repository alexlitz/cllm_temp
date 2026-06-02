# Phase 7 — Close the "fully dynamic" gaps

## Why this exists

Phase 6 (PHASE_6_DECLARATIVE_WEIGHT_AUTHORING_PLAN.md) migrated most weight writes from imperative helpers into declarative `FFNRule` + `AttentionHeadIR`. That's the IR-plumbing piece of the original "fully dynamic" vision:

> "Once we have things fully dynamic there will be no more imperative setting and then all fixes will be fixes on the level of the compiler and no non-io dims will be hardcoded and no layer indices will be hardcoded and the weights will be just outputed via compilation of the declarative spec."

After Phase 6 we're ~85-95% there. Phase 7 closes the rest.

## Five gap categories (evidence from Wave 5E + census v2 + agent reports)

| Gap | Evidence | Severity |
|---|---|---|
| Allocator `pin=` is still a hardcoded layer/unit choice corpus-wide | Every migrated op pins to legacy offset | HIGH |
| Scheduler still phase-pinned: 83/121 ops in dep-cycle SCC, 4 self-pin bugs, 3 undeclared deps | Wave 5E report | HIGH |
| Partial migrations: L6 routing + L15 memory_lookup still drive weights from imperative helpers | Wave 4G, Wave 2F | MED |
| Model-level bakes (head/embedding/initial_pc) have no declarative analog | Wave 3L skipped initial_pc_bake | MED |
| Rule definitions reference dims by `+N` offset, not semantic role | All migrated ops | LOW |

## Acceptance criteria for Phase 7

After Phase 7:
1. Scheduler `phase_required_but_undeclared` = 0 and `phase_inconsistent_with_deps` = 0
2. `dep_graph_cycle_member` reduced from 83 to ≤ 10 (the genuine cross-step cycles)
3. B14 strict-flip lands cleanly on the corpus
4. Allocator `pin=` removed from all non-IO ops; behavior preserved (logits identical) under dynamic first-fit
5. L6 routing_ffn and L15 memory_lookup bakes cut from imperative helpers
6. `TokenEmbeddingRule` IR exists; head_bake / embedding_bake / initial_pc_bake migrated
7. Rules reference dims by `(category, semantic_role)` rather than `+N` offsets where possible
8. KV eviction (PHASE 7.F, formerly SAFE_KV_EVICTION_PLAN.md §6.5-6.7) lands

## Agent waves

### 7.A — Scheduler cycle decomposition (1-2 days, 5-7 agents)

**Goal**: Make B14 strict-flip work.

**7.A.1** — Fix 4 `phase_inconsistent_with_deps` self-pin bugs. Per Wave 5E: `phase_a_ffn`, `layer1_ffn`, `layer2_mem_byte_flags`, `convo_io_prtf_transport`. Each declares a same-layer `reads` edge that should be `requires_same_layer_as` instead. 1 agent, ~30 min.

**7.A.2** — Backfill `requires["after"]` for 3 `phase_required_but_undeclared` ops: `layer15_si_mem_addr0_from_stack0` (12-layer gap), `layer14_clear_addr_key_pollution` (10-layer gap), `layer10_sp_byte_passthrough` (6-layer gap). Each needs its real cross-layer pred op-names declared. 1 agent, ~45 min.

**7.A.3** — Decompose `OUTPUT_HI_THIS_STEP` back-edges (56 of them). Audit which writers/readers actually need cross-step access vs step-local; split into `OUTPUT_HI` (step-local) and `OUTPUT_HI_PREV_STEP` (cross-step read). The B9 OUTPUT_HI rename established the precedent. 1 agent per dim category × 5 (`OUTPUT_HI_THIS_STEP`, `OUTPUT_LO`, `TEMP`, `AX_CARRY_HI`, `ADDR_KEY`), parallel. ~2-3 hr each.

**7.A.4** — Re-run `analyze_scheduler.py` to verify SCC shrinks. 1 agent, ~10 min.

**7.A.5** — B14 strict-flip: set `compile_full_vm(strict=True)` as default, run corpus. 1 agent, ~30 min. **COMPLETE** (Phase 7.A.5 default-flip): `compile_full_vm_dynamic` now defaults to `strict=True, allow_sealed_cycles=True`; backfilled `requires["after"]` declarations on `layer14_demo_phase6_wave7` and `l12_alu_mul_getobd` so the strict admission gate has no non-cycle `phase_required_but_undeclared` ops on the lookup or efficient op sets. Byte-identity preserved across both ALU modes.

**7.A.6** — Closing-audit priority 5: route `compile_full_vm` through `compile_full_vm_dynamic` by default. **COMPLETE**: production callsites now hit the hybrid dynamic-layer scheduler transparently. The legacy phase-pruning implementation is preserved behind `compile_full_vm(use_static_path=True)` as a fallback while the migration completes (deletion candidate once one release passes without a static-path fallback in production callsites). The byte-identity gate (`compare_compile_paths` / `tests/test_compile_dynamic_byte_identical.py`) was updated to invoke the static implementation via `use_static_path=True` so the regression check keeps signal on both paths.

**Wave 7.A blocked on**: nothing (can start immediately).

### 7.A.6 follow-up — Delete the static phase-pruning implementation

Once one release passes without production callsites flipping `use_static_path=True`, the entire body of `compile_full_vm` past the dynamic-routing early-return becomes unreachable and can be removed in a dedicated commit. The TODO comment at `c4_release/neural_vm/unified_compiler/full_vm_compiler.py` (`# TODO(phase-7-audit-priority-5)`) marks the deletion site. The byte-identity tests (`compare_compile_paths`, `test_compile_full_vm_dynamic_byte_identical_lookup` / `_efficient`) explicitly exercise the static branch via `use_static_path=True` and must be deleted alongside it (they become redundant once the path itself is gone).

### 7.B — Pin removal corpus-wide (2-3 days, ~6 agents)

**Goal**: Allocator chooses layer/unit/head dynamically; behavior preserved (not byte-identity — pins moved, but logits identical because the network is invariant to permutation of FFN units within a layer).

**7.B.1** — Add `dynamic_first_fit` allocator strategy. Allocator currently has `pin=` mandatory for migrated ops; add a mode where `pin=` is hint-only and first-fit chooses. 1 agent, ~1 hr.

**7.B.2** — Drop pins layer-by-layer in waves. L0/L1/L2 first (smallest), then L3-L8, then L9-L17. After each layer, re-run a sample of the smoke + spec-decode tests to confirm logits unchanged. ~5 agents in 2 waves.

**7.B.3** — Final: re-run 1096 corpus with full dynamic allocation. Compare logits to legacy. Should be identical (modulo internal permutation). 1 agent, ~30 min.

**Wave 7.B blocked on**: 7.A.5 (cycle decomposition first; strict scheduler is the gate).

### 7.C — Cut partial migrations (1 day, 2 agents)

**Goal**: Remove legacy `_set_layer6_routing_ffn` and L15 memory_lookup helpers from bake paths.

**7.C.1** — L6 routing_ffn: migrate the remaining bands (branch_pc_byte1, all_step_jsr_pc_override, jsr_sp_decrement, delayed/first/all_step_jmp_pc_override, bz/bnz_pc_override) into the IR-lowered path. Cut the legacy `_set_layer6_routing_ffn` call. Byte-identity gate. 1 agent, ~4 hr.

**7.C.2** — L15 memory_lookup: the helpers branch on `attn.num_heads` at runtime (heads 4-11 only on LEV; head 9 wipe+rewrite when num_heads>9). Need a conditional `AttentionHeadIR` variant that supports runtime-shape branching. Either (a) extend `DeclarativeAttentionHeadSpec` with a `condition` field, or (b) keep the imperative helper but factor it into per-condition IR fragments and lower the right one based on shape. 1 agent, ~6 hr.

**Wave 7.C blocked on**: nothing (can run parallel to 7.A/7.B).

### 7.D — Model-level bake migration (1-2 days, 3 agents)

**Goal**: head_bake, embedding_bake, initial_pc_bake go declarative.

**7.D.1** — Design and implement `TokenEmbeddingRule` in `c4_release/neural_vm/unified_compiler/ir.py`. Mirrors `FFNRule.constant_write` but writes into `model.embed.embed.weight[token, dim]` or `model.head.weight[token, dim]`. 1 agent, ~3 hr.

**7.D.2** — Migrate `head_bake` (3615 cells), `embedding_bake` (1823 cells), `initial_pc_bake` (2 cells). Per-op commits with byte-identity. 1 agent, ~4 hr.

**7.D.3** — Validation: run sweep_compare_ffn equivalent for token embeddings. 1 agent, ~30 min.

**Wave 7.D blocked on**: nothing.

### 7.E — Semantic dim references (1 day, 2 agents)

**Goal**: Rules reference dims by `(category, semantic_role)` not `+N` offset.

**7.E.1** — Extend `dim_registry` with semantic categories: `register_lo/hi`, `memory_lo/hi`, `output_lo/hi`, `temp_scratch`, etc. Each rule writes a `(category, role)` pair. The lowerer resolves to concrete offsets at compile time based on the current `_SetDim` layout. 1 agent, ~4 hr.

**7.E.2** — Migrate rules to use category/role refs where the +N offset isn't semantically meaningful (e.g., `TEMP+k` for `k in range(32)` becomes `(temp_scratch, k)`). Skip cases where the +N is genuinely structural (e.g., `OUTPUT_LO+nibble_value` encodes a lookup table). 1 agent, ~6 hr.

**Wave 7.E blocked on**: 7.A (need dim variants from 7.A.3 to align with categories).

### 7.F — Safe KV eviction (1-2 days, 4 agents) — formerly Phase 6.5-6.7

Per `c4_release/docs/SAFE_KV_EVICTION_PLAN.md`:

**7.F.1** — Liveness analyzer over the declarative IR (use-def graph per `(step, position, dim)`). 1 agent.

**7.F.2** — Runtime eviction pass with `--kv-eviction-policy=static_liveness` flag. Spec-decode and main-decode agree on per-step eviction. 1 agent.

**7.F.3** — CI gate (`pytest c4_release/tests/test_kv_eviction_byte_identity.py`) — 100 sampled corpus inputs, byte-identical logits. 1 agent.

**7.F.4** — Quantify peak KV memory drop (target ≥30%). 1 agent.

**Wave 7.F blocked on**: 7.A (scheduler must be clean), 7.D (model-level state must be declarative to derive complete liveness).

## Dependency graph

```
7.A (scheduler cycles) ────────┐
                                ↓
7.B (pin removal) ←──── 7.A.5 (strict flip)
                                ↓
7.E (semantic dims)
                                
7.C (partial cut)  ⊥  7.D (model bakes)   ← parallel, no deps on 7.A

                                ↓
7.F (KV eviction) ←──── 7.A + 7.D done
```

## Total wave count: ~22 agents across 6 wave groups

Spread across ~5-7 working days, heavily parallelizable. Critical path: 7.A → 7.B → 7.F (~4 days). 7.C, 7.D, 7.E can interleave.

## Connection to today's work

Phase 6 = "weights from declarations." Phase 7 = "everything else from declarations." After Phase 7 lands:

- Adding a corrective op = one FFNRule edit, no allocator pin, compiler picks layer/slot/unit
- The scheduler is unconditional (no phase tiebreaker crutch)
- KV cache is shrunk by liveness analysis without correctness regression
- Model-level state (token embeddings, structural attention reshapes) is also declarative
- The original vision is realized

## Risk register

| Risk | Mitigation | Confidence |
|---|---|---|
| Dim decomposition in 7.A.3 breaks downstream readers | Per-dim commits with byte-identity gate; revert per-commit | MED |
| Pin removal in 7.B changes logits despite the "permutation invariant" theory | Spec-decode + main-decode disagreement gate; rollback per-layer | MED |
| L15 memory_lookup conditional shape is genuinely uncompilable into static spec | Document as the one structural exception; do not block strict-flip on it | MED |
| TokenEmbeddingRule semantics differ from FFNRule (no SwiGLU); design tax | Mirror constant_write only, no gated variant initially | HIGH |
| KV eviction analyzer false-dead bug ships | Byte-identity gate fails immediately; off-by-default flag | HIGH |
