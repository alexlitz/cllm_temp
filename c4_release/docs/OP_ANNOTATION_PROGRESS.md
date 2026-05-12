# Op-annotation progress: `claims` + staleness invariants

**Goal**: Annotate every `make_*_op` factory in
`c4_release/neural_vm/unified_compiler/ops/*.py` with
`claims` (dim-ownership 4-tuples) and `produces` / `consumes_fresh`
(staleness invariants) so the framework collision and staleness
detectors see full coverage and surface latent bugs at compile time.

Branch: `annotate-all-ops-claims-staleness`. See
[`DIM_OWNERSHIP_REGISTRY.md`](DIM_OWNERSHIP_REGISTRY.md) and
[`STALENESS_INVARIANTS.md`](STALENESS_INVARIANTS.md) for the bake-author
APIs.

## Annotation conventions

- **Attention scopes** (`attn_W_v/k/q/o`): identifier `"<head>_<slot>"`,
  column `"<INPUT_DIM>+<offset>"`. For attention V/O ops the load-bearing
  rows are V slot 1..32 (most carry-forward heads) or 32..63 (multibyte
  fetch / memory lookup heads). Q/K gate dims (33+) and biases are
  usually skipped unless they are themselves shared across ops.
- **FFN scopes** (`ffn_W_up/down/gate`): identifier `<unit_idx>`, column
  `"<INPUT_DIM>+<offset>"`. For units with stable indices (allocated at
  the bake-start of a known counter) we can encode them; for dynamic
  unit allocators (`block.ffn._lN_unit_counter`) we encode only the unit
  range start that the op is guaranteed to write into.

## Op-by-op status

Legend:
- ✅ Annotated this session
- ✅ (prior) Annotated before this branch (proof-of-concept ops)
- 🟡 Partially annotated (a subset of slots covered)
- ⏸️  Skipped: stub / no-op / dep-anchor only (no bake side-effect)
- 🔵 Todo: complex multi-head/unit bake that needs careful inspection

### `l0_ops.py`

| Op | Status | Notes |
|---|---|---|
| `make_phase_a_ffn_op` | ✅ | 7 W_down units → NEXT_* dims |
| `make_layer0_threshold_attn_op` | ✅ | 8 threshold heads, V slot 1..7 per head |

### `l1_ops.py`

| Op | Status | Notes |
|---|---|---|
| `make_layer1_ffn_op` | ✅ | 5 W_down units: STACK0_BYTE0 + BYTE_INDEX_0..3 |
| `make_layer1_threshold_attn_op` | ✅ | 5 heads (0,1,2,4 threshold + 3 STEP_END) |

### `l2_ops.py`

| Op | Status | Notes |
|---|---|---|
| `make_layer2_mem_byte_flags_op` | ✅ | 8 W_down units (MEM_VAL_B0..3 + BYTE_INDEX_0..3) |
| `make_layer2_initial_pc_bake_cancel_op` | ✅ | Units 8+9: EMBED_LO/HI cancel |
| `make_layer2_threshold_attn_op` | ✅ | 1 threshold head (0 → L2H0) |
| `make_layer2_lookback_detection_head_op` | ⏸️  | Convo-IO gated, body uses external helper |

### `l3_ops.py`

| Op | Status | Notes |
|---|---|---|
| `make_layer3_ffn_op` | 🔵 | Complex multi-section L3 FFN (PC/SP/BP defaults + increment) |
| `make_layer3_ffn_dep_anchor_op` | ⏸️  | No-op dep anchor |
| `make_layer3_carry_forward_attn_op` | ✅ | 7 heads (0-3 carry, 5 AX_FULL, 6 BP→PC LEV) |
| `make_layer3_convo_io_state_init_op` | ⏸️  | Convo-IO gated, body uses external helper |

### `l4_ops.py`

| Op | Status | Notes |
|---|---|---|
| `make_layer4_pc_relay_op` | ✅ | Heads 0 + 1, V slots 1..32 |
| `make_layer4_ffn_op` | 🔵 | Multi-section: PC+1..PC+4 rotation units |
| `make_layer4_sp_to_addr_key_op` | ✅ | Heads 2 + 3 SP staging (gated by `enable`) |

### `l5_ops.py`

| Op | Status | Notes |
|---|---|---|
| `make_layer5_fetch_op` | ✅ (prior) | 6 heads × 32 V slots — full proof-of-concept |
| `make_layer5_fetch_dep_anchor_op` | ⏸️  | No-op dep anchor |
| `make_opcode_decode_ffn_op` | 🔵 | Large opcode decoder FFN |
| `make_opcode_decode_ffn_dep_anchor_op` | ⏸️  | No-op dep anchor |

### `l6_ops.py`

| Op | Status | Notes |
|---|---|---|
| `make_layer6_attn_op` | 🔵 | Dep anchor; bake lives in `layer6_attn_bake` |
| `make_layer6_routing_ffn_op` | 🔵 | Huge per-opcode routing FFN |
| `make_layer6_relay_heads_op` | 🔵 | Dep anchor; bake lives in `layer6_relay_heads_bake` |
| `make_layer6_attn_bake_op` | 🔵 | Heads 0-5 with distinct semantics each |
| `make_layer6_relay_heads_bake_op` | 🔵 | Heads 6 + 7 PSH/JSR relay |
| `make_layer6_bz_bnz_relay_bake_op` | 🔵 | Head 4 BZ/BNZ relay |
| `make_binary_pop_sp_increment_op` | 🔵 | L6 FFN extension, ~unit 2200 |
| `make_putchar_think_protocol_op` | ⏸️  | Phase-1 no-op stub |
| `make_prtf_think_protocol_op` | ⏸️  | Phase-2a no-op stub |
| `make_open_clos_tool_call_op` | ⏸️  | Dep-graph anchor only |

### `l7_ops.py`

| Op | Status | Notes |
|---|---|---|
| `make_layer7_operand_gather_op` | ✅ | Heads 0 + 1 V/O slots; produces ALU_LO/HI at AX_byte0 |
| `make_layer7_memory_heads_op` | 🟡 | Heads 2-4 + scalar relays for heads 5, 7 |
| `make_format_pointer_extraction_op` | ⏸️  | Convo-IO gated |

### `l8_ops.py`

| Op | Status | Notes |
|---|---|---|
| `make_layer8_alu_op` | ✅ (prior + ext) | consumes_fresh AX_CARRY_LO + ALU_LO |
| `make_format_position_counter_op` | ⏸️  | Convo-IO gated |
| `make_layer8_multibyte_fetch_op` | ⏸️  | No-op dep anchor |
| `make_layer8_multibyte_fetch_bake_op` | ✅ | Head 3 V slots 32..63 |
| `make_layer8_multibyte_routing_op` | ✅ (prior) | produces OUTPUT_LO/HI at AX_byte0 |
| `make_layer8_sp_gather_op` | ⏸️  | No-op dep anchor |
| `make_layer8_sp_gather_bake_op` | ✅ | Heads 0-2 V slots 1..32 |
| `make_layer8_head6_ax_carry_refresh_op` | ✅ (prior) | produces AX_CARRY_LO/HI at AX_byte0 |
| `make_layer8_op_imm_relay_op` | ✅ | Head 4 V slot 0 → OP_IMM |
| `make_layer8_mem_to_alu_op` | ✅ | Head 5 V slots 1..32 (gated by `enable`) |

### `l9_ops.py`

| Op | Status | Notes |
|---|---|---|
| `make_layer9_alu_op` | ✅ | consumes_fresh ALU_HI |
| `make_layer9_lev_addr_relay_op` | ✅ | Head 0 V/O slots 1..32 |
| `make_layer9_lev_bp_to_pc_relay_op` | ✅ | Head 1 V/O slots 1..32 |
| `make_format_string_fetch_head_op` | ⏸️  | Convo-IO gated |
| `make_layer9_alibi_mem_attn_op` | ✅ | Head 2 V slots 1..32 (gated by `enable`) |
| `make_layer9_marker_suppress_op` | 🔵 | FFN marker suppression units |

### `l10_ops.py`

| Op | Status | Notes |
|---|---|---|
| `make_layer10_carry_relay_op` | ⏸️  | Dep anchor; bake lives in `*_bake` |
| `make_layer10_byte_passthrough_op` | ⏸️  | Dep anchor |
| `make_layer10_sp_byte_passthrough_op` | ⏸️  | Dep anchor |
| `make_layer10_psh_stack0_passthrough_op` | ⏸️  | Dep anchor |
| `make_layer10_carry_relay_bake_op` | ✅ | Head 0 CARRY[1]/[2] slots |
| `make_layer10_byte_passthrough_bake_op` | ✅ | Head 1 V slots 0..31 (passthrough chain) |
| `make_layer10_sp_byte_passthrough_bake_op` | ✅ | Head 2 V slots 0..31 |
| `make_layer10_psh_stack0_passthrough_bake_op` | ✅ | Head 3 V slots 0..31 |
| `make_layer10_stack0_byte_relay_bake_op` | ✅ | Head 4 V slots 0..31 |
| `make_layer10_alu_op` | ✅ | consumes_fresh ALU_LO/HI + AX_CARRY_LO/HI |
| `make_layer10_stack0_byte_relay_op` | ⏸️  | Dep anchor |
| `make_l10_post_ops_combined` | 🔵 | Composite post-op modules |
| `make_l10_post_op_attach_op` | ⏸️  | Block-level post-op attach |

### `l11_ops.py`

| Op | Status | Notes |
|---|---|---|
| `make_layer11_mul_partial_op` | ✅ | consumes_fresh ALU_LO/HI + AX_CARRY_LO/HI |

### `l12_ops.py`

| Op | Status | Notes |
|---|---|---|
| `make_layer12_mul_combine_op` | 🔵 | Multi-stage MUL combiner |

### `l13_ops.py`

| Op | Status | Notes |
|---|---|---|
| `make_layer13_mem_addr_gather_op` | ✅ | Heads 0-2 V slots 1..32 |
| `make_layer13_shifts_op` | 🔵 | Large SHL/SHR lookup table (4096 units) |

### `l14_ops.py`

| Op | Status | Notes |
|---|---|---|
| `make_layer14_mem_generation_op` | ✅ | 8 heads V slots 1..32 |
| `make_layer14_temp_clear_op` | 🔵 | Dynamic unit counter |
| `make_layer14_clear_addr_key_pollution_op` | 🔵 | Dynamic unit counter |
| `make_layer14_clear_output_corruption_op` | 🔵 | Dynamic unit counter |
| `make_layer14_clear_mem_marker_output_op` | 🔵 | Dynamic unit counter |
| `make_layer14_jsr_ax_bytes_zero_op` | 🔵 | Dynamic unit counter |
| `make_layer14_lc_ax_bytes_zero_op` | 🔵 | Dynamic unit counter |
| `make_layer14_addr_key_neural_decode_op` | 🔵 | Massive ADDR_KEY decode (~1184 units), gated |

### `l15_ops.py`

| Op | Status | Notes |
|---|---|---|
| `make_nibble_copy_ffn_op` | 🔵 | Conditional nibble copy FFN |
| `make_layer15_memory_lookup_op` | ✅ | Heads 0-3 V slots 32..63 |
| `make_layer15_nibble_copy_op` | 🔵 | Same as nibble_copy_ffn but pinned |
| `make_l15_attention_resize_op` | ⏸️  | Resize op (no weight writes) |

### `l16_ops.py`

| Op | Status | Notes |
|---|---|---|
| `make_layer16_lev_routing_op` | 🔵 | LEV SP = BP + 16 routing FFN |

### `model_ops.py`

| Op | Status | Notes |
|---|---|---|
| `make_io_putchar_routing_op` | 🔵 | L6 FFN ~unit 1500 onward |
| `make_function_call_weights_op` | ✅ (prior) | Multi-layer claim — full POC |
| `make_opcode_relay_head_op` | 🔵 | L6 head 6 opcode-relay weights |
| `make_residual_alibi_slopes_op` | ⏸️  | Only mutates alibi_slopes (no weight writes) |
| `make_branch_override_patch_op` | ⏸️  | Defensive post-pass; modifies existing units |
| `make_l6_dead_unit_zero_op` | ⏸️  | Defensive post-pass; zeros existing units |
| `make_l7_dead_unit_zero_op` | ⏸️  | Defensive post-pass; modifies existing units |
| `make_right_size_ffns_op` | ⏸️  | Structural FFN trim |
| `make_expand_wrapper_blocks_op` | ⏸️  | Structural block expansion |
| `make_head_bake_op` | ⏸️  | Output head (separate scope from `embed_row`) |
| `make_embedding_bake_op` | 🔵 | Per-token embedding bake (each token row) |
| `make_initial_pc_bake_op` | 🔵 | REG_PC token embedding bake (one row) |
| `make_contract_validation_op` | ⏸️  | Diagnostic only |

### `flag_gated_ops.py`

| Op | Status | Notes |
|---|---|---|
| Tool-call ops (3) | ⏸️  | Flag-gated; bakes only when `enable_tool_calling=True` |
| Convo-IO opcode decode / relay / state machine / output routing | ⏸️  | Flag-gated |
| V18 Phase-1 step-resume / PC-SP-latch / PRTF-capture / PRTF-transport | ⏸️  | Double-gated stubs |
| `make_null_terminator_detection_op` | 🔵 | L10 FFN units 1864+ when enabled |
| `make_conversational_io_output_routing_op` | 🔵 | L15 FFN units 1200+ when enabled |

### `alu_ops.py`

| Op | Status | Notes |
|---|---|---|
| L13 SHL/SHR composite (5 ops) | 🔵 | Composite-module installer chain |
| L8-L13 ALU postop attach (6 ops) | ⏸️  | Block.post_ops module attach |
| L8/L10/L11 efficient ALU wraps (3 ops) | ⏸️  | Block.ffn replacement (composite modules) |
| L11/L12 MUL stage installers (8 ops) | 🔵 | Composite-module installer chain (FlattenedALUMul) |

### `user_input_ops.py`

| Op | Status | Notes |
|---|---|---|
| `make_layer5_user_input_gather_op` | ⏸️  | Phase-1 stub (raises if enabled) |
| `make_layer6_getchar_routing_op` | ⏸️  | Phase-1 stub (raises if enabled) |

## Staleness invariants added

| Op | `produces` | `consumes_fresh` |
|---|---|---|
| `layer8_head6_ax_carry_refresh` | `AX_CARRY_LO/HI -> AX_byte0` | — |
| `layer8_multibyte_routing` | `OUTPUT_LO/HI -> AX_byte0` (IMM only) | — |
| `layer8_alu` | — | `AX_CARRY_LO -> AX_byte0`, `ALU_LO -> AX_byte0` |
| `layer7_operand_gather` | `ALU_LO/HI -> AX_byte0` | — |
| `layer9_alu` | — | `ALU_HI -> AX_byte0` |
| `layer10_alu` | — | `ALU_LO/HI -> AX_byte0`, `AX_CARRY_LO/HI -> AX_byte0` |
| `layer11_mul_partial` | — | `ALU_LO/HI -> AX_byte0`, `AX_CARRY_LO/HI -> AX_byte0` |

## Verification

After every batch the following gates must pass with zero new
collision or staleness warnings:

```bash
timeout 400 python -m pytest \
  c4_release/tests/test_smoke.py::TestSmokeBasic::test_imm_exit \
  c4_release/tests/test_dim_ownership.py \
  c4_release/tests/test_staleness_invariants.py \
  c4_release/tests/test_runtime_vanilla.py \
  c4_release/tests/test_layer_idx_consistency.py \
  c4_release/tests/test_compile_determinism.py \
  -v --tb=line --timeout=300
```

The dim-ownership and staleness suites assert
"no unexpected warnings in the production build", so any silent
collision / staleness regression is caught at gate-time.

## Latent bugs surfaced

None this session — all annotations were ratified by the
production-build gate without surfacing any new warnings.

## Follow-up TODOs

The 🔵 entries above are candidates for a follow-up annotation pass:

1. **Complex multi-section FFNs** (`l3_ffn`, `l4_ffn`, `l13_shifts`,
   `l14_addr_key_neural_decode`, `l16_lev_routing`, `layer6_routing_ffn`,
   `opcode_decode_ffn`). These allocate units across many ranges; each
   needs careful enumeration of (unit, output_dim) pairs from its
   helper.
2. **L6 attention heads** (`layer6_attn_bake`, `layer6_relay_heads_bake`,
   `layer6_bz_bnz_relay_bake`, `opcode_relay_head`). Each programs
   different head/slot patterns with distinct semantics; the bake
   helpers (~3000+ lines in `vm_step.py`) need to be carefully read
   slot-by-slot.
3. **Dynamic unit-counter ops** (L14 cleanup ops, L11/L12 MUL stages).
   These bake on top of a shared `_lN_unit_counter` so claims can only
   capture the start-unit guarantee, not the full range.
4. **Composite-module attach ops** (ALU shift composite, MUL stages,
   post_op attach ops). These don't write weights directly — they
   install Python modules onto `block.ffn` / `block.post_ops`. The
   weight writes happen at module-construction time, not at the op's
   `bake_fn` call. Annotating these would require introspecting the
   module's internal `W_up/W_down/...`.
5. **Embedding ops** (`embedding_bake`, `initial_pc_bake`). The
   `embed_row` scope is set up for per-token claims but the bake
   touches many token rows in a loop.
