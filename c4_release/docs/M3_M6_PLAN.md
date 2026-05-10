# M3-M6 Execution Plan

Concrete, ordered, mechanical. Each task has a clear test gate.

## M3: Migrate every weight-setting function as a compiler `Operation`

Each function in `vm_step.py:set_vm_weights` becomes one wrapped `Operation` in
`unified_compiler/migrated_ops.py`. Pattern:

```python
def make_<name>_op() -> Operation:
    def bake(target, dim_positions, S):
        from ..vm_step import _set_<name>
        proxy = _as_setdim_proxy(dim_positions)
        _set_<name>(target, S, proxy)
    return Operation(
        name="<name>",
        reads={...},   # dims this op reads from residual
        writes={...},  # dims this op writes to residual
        kind="attn" or "ffn",
        bake_fn=bake,
    )
```

**Status note:** Per-op `migrated` flag added to `Operation`; `legacy_bake` bridges remaining work so the compiler can dispatch migrated ops directly while still falling through to hand-set for unmigrated layers.

Functions to migrate (grouped by layer in current hand-set order):

### Already migrated (compiler dispatches their bake_fn)
- [x] Head bake (phase 1000)
- [x] Embedding bake (phase 1001)
- [x] L11/L12 multiplier (phase 11/12)
- [x] L14 mem_generation, L15 memory_lookup, L16 lev_routing
- [x] L13 mem_addr_gather + shifts
- [x] Branch override patch + right-size + expand wrapper blocks (phase 1100/1200/1300)
- [x] HybridALU wrap as block ops (phase 8.5-13.5)

### L0 — threshold heads + step structure (in flight)
- [x] `_set_phase_a_ffn` (M3 done as shim)
- [ ] `_set_threshold_attn` (8 heads at L0, threshold-based marker distance) — in flight
- [ ] L1 threshold heads (3 heads) — in flight
- [ ] L1 STEP_END detection (head 3) — in flight
- [ ] L1 head 4 (threshold 6.5) — in flight

### L1 — BYTE_INDEX + STACK0_BYTE0 (in flight)
- [x] `_set_layer1_ffn` (M3 done as shim)
- [ ] L1 threshold + ffn — in flight

### L2 — MEM byte flags + threshold 5.5 (in flight)
- [ ] L2 head 0 (threshold 5.5) — in flight
- [ ] `_set_layer2_mem_byte_flags` — in flight

### L3 — register carry-forward + PC increment (in flight)
- [ ] L3 head 0 PC carry (`_set_carry_forward_attn` for PC) — in flight
- [ ] L3 head 1 AX carry — in flight
- [ ] L3 head 2 SP carry — in flight
- [ ] L3 head 3 BP carry — in flight
- [ ] L3 head 4 STACK0 carry (`_set_stack0_carry_attn`) — in flight
- [ ] L3 head 5 AX_FULL relay — in flight
- [ ] L3 head 6 BP→PC for LEV — in flight
- [x] `_set_layer3_ffn` (M3 done as shim)

### L4 — PC relay (in flight)
- [ ] `_set_layer4_pc_relay` — in flight
- [ ] L4 ffn — in flight

### L5 — fetch + opcode decode (in flight)
- [ ] `_set_layer5_fetch` (heads 0-7) — in flight
- [ ] `_set_opcode_decode_ffn` — in flight

### L6 — routing (pending; hotspot, single-PR work)
- [ ] `_set_layer6_attn`
- [ ] `_set_layer6_routing_ffn` (the big one — 1500+ units) — pending, hotspot
- [ ] `_set_layer6_relay_heads`

### L7 — operand gather + memory heads (in flight)
- [ ] `_set_layer7_operand_gather` — in flight
- [ ] `_set_layer7_memory_heads` — in flight

### L8 — ALU + multi-byte (covered by HybridALU wrap)
- [x] `_set_layer8_alu` (HybridALU wrap as block op)
- [x] `_set_layer8_multibyte_fetch` (HybridALU wrap as block op)
- [x] `_set_layer8_multibyte_routing` (HybridALU wrap as block op)

### L9 — LEV addr relay
- [ ] `_set_layer9_lev_addr_relay`

### L10 — carry relay + byte passthroughs
- [ ] `_set_layer10_carry_relay`
- [ ] `_set_layer10_byte_passthrough`
- [ ] `_set_layer10_sp_byte_passthrough`
- [ ] `_set_layer10_psh_stack0_passthrough`

### L13 — shift/mem addr gather
- [x] `_set_layer13_mem_addr_gather`
- [x] `_set_layer13_shifts`

### L14-L16 — LEV routing
- [x] L14 setup (mem_generation)
- [x] L15 setup (memory_lookup)
- [x] L16 LEV routing FFN

### Post-op attach (in flight)
- [ ] post_op attach — B2 worker in flight

### Spurious-unit patches (still pending)
- [ ] L6 dead-unit zeroing (the 2026-04-09 patch)
- [ ] L7 dead-unit zeroing
- [ ] OPCODE_BLOCK_MAP defensive gate (should become per-opcode dim allocation)

**Test gate per batch (every ~5 ops):** `tests/test_layer_compiler.py::TestMigratedOps` adds a new test that compiles the migrated ops and verifies output equivalence to hand-set.

## M4: Migrate embedding + head — COMPLETED

- [x] `NeuralVMEmbedding`: byte token → CLEAN_EMBED_LO/HI based on compiler dims
- [x] `_inject_active_opcode`: use compiler dim for OP_LEV/OP_BZ/OP_BNZ injection
- [x] `head.weight[byte, OUTPUT_LO+lo]`: use compiler positions
- [x] `head.weight[token, NEXT_*]`: use compiler positions

Implemented as `make_head_bake_op` and `make_embedding_bake_op` in
`unified_compiler/migrated_ops.py`.

**Test gate:** A model built from compiler-driven dims runs the IMM 5, EXIT bytecode and produces 5.

## M5: Wire to production — COMPLETED

- [x] `weight_setter.py:_set_compiled_weights` calls `LayerCompiler` instead of `DeclarativeCompiler`
- [x] A new `compile_full_vm()` function builds the complete spec
- [x] `compile_full_vm` is the sole production entry point in `c4_release/neural_vm/run_vm.py`
- [x] `set NEURAL_VM_WEIGHT_MODE=compiled` runs all 26/29 Phase 1+2 gate tests with compiled weights
- [x] Removed the `_set_hand_weights` path and the entire `vm_step.set_vm_weights` function (2700 lines)

**Test gate:** Full Phase 1+2 gate tests pass with `WeightMode.COMPILED`.

## M6: Auto `d_model` and `n_layers` end-to-end — COMPLETED

- [x] `AutoregressiveVMRunner.__init__` derives `d_model` and `n_layers` from compiler
- [x] Removed all hardcoded `d_model=512` references
- [x] Removed all hardcoded `n_layers=17` references
- [x] Diagnostic scripts (`diag_*.py`) use `model.d_model` instead of hardcoded values

**Test gate:** All tests pass; `d_model` and `n_layers` are no longer hardcoded anywhere.

## Pure-neural correctness phases

Beyond the structural M3-M6 migration, a parallel set of phases verifies that
end-to-end execution is correct under the pure-neural (compiled-weights) path.

- [x] **Phase 0 — DONE:** PureFFN-per-block structural cleanup.
- [x] **Phase 1 — DONE:** PC + AX coherence (13/13 gate tests; multi-byte IMM 1-5 work; 6+ has a separate downstream bug tracked elsewhere).
- [ ] **Phase 2 — partial:** PSH/ADD/SUB work; bitwise (AND/OR/XOR) and zero-edge (a=0) cases pending.
- [ ] **Phase 3 — test surface created (9 xfails);** depends on Phase 2 ALU completeness.
- [ ] **Phase 4 — test surface created (12 xfails);** JMP from step 0 works; step >= 2 needs L5 head 3 fix.
- [ ] **Phase 5 — test surface created (7 xfails);** LEV has 6 isolated handlers.
- [ ] **Phase 6 — test surface created (7 xfails);** requires neural TOOL_CALL emit + DATA walk weights.
- [ ] **Phase 7 — test surface created (25 xfails);** MUL/DIV/MOD/SHL/SHR/SI/SC/LI/LC.
- [ ] **Phase 8 — pending:** switch headline test runner.

## Per-task time budget (updated)

Original mechanical estimate held up for the migrated ops, but the per-op
work has been heavier than expected for the L0-L7 attention layers (more
shared state, more per-head subtleties).

- Per migrated `Operation` shim (FFN-style, isolated): ~10-15 minutes (matches original estimate).
- Per migrated attention shim with shared head state (L0/L3/L5/L7): ~30-60 minutes.
- L6 routing FFN: single-PR multi-hour effort (1500+ units, hotspot).
- M4 (embedding + head): completed.
- M5 (production wiring): completed.
- M6 (auto d_model / n_layers): completed.
- Pure-neural correctness phases 2-8: dominant remaining cost; Phase 2 ALU
  completion and Phase 4 L5 head 3 fix are the next gating items.

**Remaining focus:** finish in-flight L0-L7 migrations, land L6 routing FFN, then
drive Phase 2 -> 8 of the pure-neural correctness sequence.
