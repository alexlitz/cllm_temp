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

Functions to migrate (grouped by layer in current hand-set order):

### L0 — threshold heads + step structure
- [x] `_set_phase_a_ffn` (M3 done as shim)
- [ ] `_set_threshold_attn` (8 heads at L0, threshold-based marker distance) — special: parameterized
- [ ] L1 threshold heads (3 heads)
- [ ] L1 STEP_END detection (head 3)
- [ ] L1 head 4 (threshold 6.5)

### L1 — BYTE_INDEX + STACK0_BYTE0
- [x] `_set_layer1_ffn` (M3 done as shim)

### L2 — MEM byte flags + threshold 5.5
- [ ] L2 head 0 (threshold 5.5)
- [ ] `_set_layer2_mem_byte_flags`

### L3 — register carry-forward + PC increment
- [ ] L3 head 0 PC carry (`_set_carry_forward_attn` for PC)
- [ ] L3 head 1 AX carry
- [ ] L3 head 2 SP carry
- [ ] L3 head 3 BP carry
- [ ] L3 head 4 STACK0 carry (`_set_stack0_carry_attn`)
- [ ] L3 head 5 AX_FULL relay
- [ ] L3 head 6 BP→PC for LEV
- [x] `_set_layer3_ffn` (M3 done as shim)

### L4 — PC relay
- [ ] `_set_layer4_pc_relay`

### L5 — fetch + opcode decode
- [ ] `_set_layer5_fetch` (heads 0-7)
- [ ] `_set_opcode_decode_ffn`

### L6 — routing
- [ ] `_set_layer6_attn`
- [ ] `_set_layer6_routing_ffn` (the big one — 1500+ units)
- [ ] `_set_layer6_relay_heads`

### L7 — operand gather + memory heads
- [ ] `_set_layer7_operand_gather`
- [ ] `_set_layer7_memory_heads`

### L8 — ALU + multi-byte
- [ ] `_set_layer8_alu`
- [ ] `_set_layer8_multibyte_fetch`
- [ ] `_set_layer8_multibyte_routing`

### L9 — LEV addr relay
- [ ] `_set_layer9_lev_addr_relay`

### L10 — carry relay + byte passthroughs
- [ ] `_set_layer10_carry_relay`
- [ ] `_set_layer10_byte_passthrough`
- [ ] `_set_layer10_sp_byte_passthrough`
- [ ] `_set_layer10_psh_stack0_passthrough`

### L13 — shift/mem addr gather
- [ ] `_set_layer13_mem_addr_gather`
- [ ] `_set_layer13_shifts`

### L14-L16 — LEV routing
- [ ] L14 setup
- [ ] L15 setup
- [ ] L16 LEV routing FFN

### Spurious-unit patches
- [ ] L6 dead-unit zeroing (the 2026-04-09 patch)
- [ ] L7 dead-unit zeroing
- [ ] OPCODE_BLOCK_MAP defensive gate (should become per-opcode dim allocation)

**Total: ~30 ops to migrate.**

**Test gate per batch (every ~5 ops):** `tests/test_layer_compiler.py::TestMigratedOps` adds a new test that compiles the migrated ops and verifies output equivalence to hand-set.

## M4: Migrate embedding + head

- [ ] `NeuralVMEmbedding`: byte token → CLEAN_EMBED_LO/HI based on compiler dims
- [ ] `_inject_active_opcode`: use compiler dim for OP_LEV/OP_BZ/OP_BNZ injection
- [ ] `head.weight[byte, OUTPUT_LO+lo]`: use compiler positions
- [ ] `head.weight[token, NEXT_*]`: use compiler positions

**Test gate:** A model built from compiler-driven dims runs the IMM 5, EXIT bytecode and produces 5.

## M5: Wire to production

- [ ] `weight_setter.py:_set_compiled_weights` calls `LayerCompiler` instead of `DeclarativeCompiler`
- [ ] A new `compile_full_vm()` function in `unified_compiler/full_vm.py` builds the complete spec
- [ ] `set NEURAL_VM_WEIGHT_MODE=compiled` runs all 26/29 Phase 1+2 gate tests with compiled weights
- [ ] Remove the `_set_hand_weights` path and the entire `vm_step.set_vm_weights` function (2700 lines)

**Test gate:** Full Phase 1+2 gate tests pass with `WeightMode.COMPILED`.

## M6: Auto `d_model` and `n_layers` end-to-end

- [ ] `AutoregressiveVMRunner.__init__` derives `d_model` and `n_layers` from compiler
- [ ] Remove all hardcoded `d_model=512` references
- [ ] Remove all hardcoded `n_layers=17` references
- [ ] Diagnostic scripts (`diag_*.py`) use `model.d_model` instead of hardcoded values

**Test gate:** All tests pass; `d_model` and `n_layers` are no longer hardcoded anywhere.

## Per-task time budget

- Each `Operation` migration shim: ~10 minutes (read function, identify reads/writes, write wrapper, run test)
- 30 migration shims = ~5 hours
- M4: ~2 hours
- M5: ~3 hours (rewiring + verification)
- M6: ~1 hour

**Total remaining: ~11 hours of focused work**, well under the 4-6 weeks I previously estimated because the actual work per migration is mechanical once the pattern is established. The big-ticket items are M5 (rewiring production) and ensuring equivalence at each step.
