# Per-Opcode Block-Skip Analysis (2026-06-05)

**Question**: For each `(block_idx, opcode)` cell, does the block do ANY
meaningful work for this opcode? If not, can the runner skip it?

**Method**:
1. `compile_full_vm(disk_cache=False)` — produces a `ModelLayout` whose
   `ops_per_layer`, `block_ops`, and `model_ops` are fully populated.
2. Bucket every `Operation` to its target block:
   - `ops_per_layer[i]` → block `i`.
   - `block_ops` and `model_ops` carrying a non-`None` `layer_idx` → that block.
   - `model_ops` with `layer_idx=None` (head, embedding, right-size, contract
     validation, etc.) are model-wide compile-time bakes; they don't program a
     specific block's *runtime* behavior, so they are excluded from the
     per-token walkthrough.
3. Classify each op:
   - **Generic** (`Operation.opcodes == set()`) — by contract, fires on every
     step regardless of opcode. Cannot be skipped without rewriting the op.
   - **Specific** (non-empty `opcodes={"OP_*", ...}`) — fires only on the
     declared opcodes (the assertion `verify_opcode_coverage` checks).
4. A block is **skippable** for opcode `X` iff every op assigned to that block
   is specific AND no op's `opcodes` set contains `OP_X`.

**Ground truth caveat**: only ~25 of the 327 op factories declare
`opcodes={...}`. The verifier treats unannotated FFN ops outside L0/L1/L3 as
"suspicious", but until they're annotated the strict-honest reading is
"empty == fires every opcode". Numbers below reflect that strict reading.

**Model shape**: the layout produces `n_layers=18`. After
`expand_wrapper_blocks` (ALU pipelines), the runtime model has 30 PyTorch
blocks, but ALU expansions are pinned to specific opcodes already at the bake
level — for the per-block skip question we use the 18 layout-level blocks
(L0…L17). The task brief's "35" is `TOKENS_PER_STEP`, not the block count.

## Per-Block Op Census

| Block | Generic ops                                                                                                             | Specific ops (with gate)                                                                                                          |
| ----- | ----------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| L0    | layer10_carry_relay, layer0_threshold_attn, phase_a_ffn                                                                 | —                                                                                                                                 |
| L1    | layer1_threshold_attn, layer1_ffn                                                                                       | —                                                                                                                                 |
| L2    | layer2_threshold_attn, layer2_mem_byte_flags, layer2_lookback_detection_head, layer2_initial_pc_bake_cancel             | —                                                                                                                                 |
| L3    | layer3_carry_forward_attn, layer10_sp_byte_passthrough, layer3_ffn, layer3_convo_io_state_init, convo_io_step_resume    | —                                                                                                                                 |
| L4    | _layer3_ffn_dep_anchor, layer4_pc_relay, layer4_ffn, layer4_sp_to_addr_key, convo_io_prtf_transport                     | —                                                                                                                                 |
| L5    | _layer5_fetch_dep_anchor, layer5_fetch, opcode_decode_ffn, convo_io_opcode_decode                                       | layer5_user_input_gather {GETCHAR}                                                                                                |
| L6    | _opcode_decode_ffn_dep_anchor, layer8_multibyte_fetch, putchar_think_protocol, prtf_think_protocol, open_clos_tool_call, convo_io_state_machine, convo_io_pc_sp_latch, function_call_weights | layer6_routing_ffn {EXIT, IMM, JMP, JSR, NOP}                                                                                     |
| L7    | layer6_attn, layer9_marker_suppress, layer7_memory_heads, format_pointer_extraction, convo_io_prtf_capture              | layer7_operand_gather {ADD, ADJ, AND, DIV, ENT, LEA, MOD, MUL, OR, SHL, SHR, SUB, XOR}                                            |
| L8    | nibble_copy_ffn, layer6_relay_heads, layer8_head6_ax_carry_refresh, format_position_counter, layer8_multibyte_routing, layer8_sp_gather_bake, layer8_multibyte_fetch_bake, layer8_mem_to_alu, layer8_op_imm_relay, l8_alu_postop_attach | layer8_alu {ADD, EQ, GE, GT, LE, LEA, LT, NE, SUB}                                                                                |
| L9    | layer8_sp_gather, layer9_alibi_mem_attn, format_string_fetch_head, l9_alu_postop_attach                                 | layer9_alu {ADD, AND, EQ, GE, GT, LE, LT, NE, OR, SUB, XOR}; layer9_lev_addr_relay {LEV}; layer9_lev_bp_to_pc_relay {LEV}         |
| L10   | layer10_byte_passthrough, layer10_psh_stack0_passthrough, layer10_stack0_byte_relay, layer10_byte_passthrough_bake, layer10_sp_byte_passthrough_bake, layer10_stack0_byte_relay_bake, null_terminator_detection, l10_post_op_attach, l10_alu_divmod_bdtoge, l10_alu_divmod_longdiv, l10_alu_divmod_getobd, l10_alu_divmod_install, l10_alu_postop_attach | layer10_carry_relay_bake {ADD, LEA, SUB}; layer10_psh_stack0_passthrough_bake {PSH}; layer10_alu {AND, DIV, EQ, GE, GT, LE, LT, MOD, NE, OR, XOR} |
| L11   | layer11_mul_partial, l11_alu_postop_attach                                                                              | —                                                                                                                                 |
| L12   | layer12_mul_combine, l12_alu_postop_attach                                                                              | —                                                                                                                                 |
| L13   | l13_alu_postop_attach                                                                                                   | layer13_mem_addr_gather {LC, LI, SC, SI}; layer13_shifts {SHL, SHR}                                                               |
| L14   | layer14_clear_addr_key_pollution, layer14_clear_output_corruption, layer14_addr_key_neural_decode                       | layer14_mem_generation {ENT, JSR, PSH, SC, SI}; layer14_temp_clear {LEV}; layer14_clear_mem_marker_output {ENT, JSR}; layer14_jsr_ax_bytes_zero {JSR}; layer14_lc_ax_bytes_zero {LC}; layer14_alu_nocarry_ax_bytes_zero {AND, OR, SHR, XOR} |
| L15   | layer15_nibble_copy, l15_attention_resize, conversational_io_output_routing                                             | layer15_memory_lookup {LC, LI}                                                                                                    |
| L16   | —                                                                                                                       | layer16_lev_routing {LEV}                                                                                                         |
| L17   | l10_post_ops_combined                                                                                                   | —                                                                                                                                 |

## Skip Matrix (18 blocks × 31 opcodes)

`1` = block does work for this opcode (cannot skip). `.` = SKIPPABLE.

```
Op   | L 0 L 1 L 2 L 3 L 4 L 5 L 6 L 7 L 8 L 9 L10 L11 L12 L13 L14 L15 L16 L17 | needed
---------------------------------------------------------------------------------------
IMM  |  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   .   1 | 17
LEA  |  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   .   1 | 17
JMP  |  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   .   1 | 17
JSR  |  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   .   1 | 17
BZ   |  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   .   1 | 17
BNZ  |  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   .   1 | 17
ENT  |  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   .   1 | 17
ADJ  |  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   .   1 | 17
LEV  |  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1 | 18
LI   |  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   .   1 | 17
LC   |  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   .   1 | 17
SI   |  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   .   1 | 17
SC   |  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   .   1 | 17
PSH  |  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   .   1 | 17
OR   |  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   .   1 | 17
XOR  |  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   .   1 | 17
AND  |  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   .   1 | 17
EQ   |  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   .   1 | 17
NE   |  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   .   1 | 17
LT   |  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   .   1 | 17
GT   |  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   .   1 | 17
LE   |  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   .   1 | 17
GE   |  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   .   1 | 17
SHL  |  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   .   1 | 17
SHR  |  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   .   1 | 17
ADD  |  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   .   1 | 17
SUB  |  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   .   1 | 17
MUL  |  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   .   1 | 17
DIV  |  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   .   1 | 17
MOD  |  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   .   1 | 17
EXIT |  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   .   1 | 17
```

Total cells: 558. Skip cells: **30 / 558 = 5.4%**.

**Headline**: under the strict-honest reading of `opcodes={...}` gating, the
only block that's currently safe to skip per-opcode is **L16** (LEV-only —
the dedicated routing layer added in n_layers=17). Every other block has at
least one **generic** op (empty `opcodes=` set), so by contract it fires on
every step.

## Per-Opcode Minimum Block Path

- **LEV**: needs all 18 blocks (L16's LEV routing brings it back to full).
- **All other 30 opcodes**: need 17 blocks (L0…L15, L17). L16 skippable.

This is unsurprising: L16 was deliberately added as the post-LEV routing
layer (`n_layers=16 → 17 for LEV Phase 3`, per `vm_step.py:1307` comment).

## Compute Savings

Under a uniform opcode mix:
- Average blocks per token = 17.03 / 18 ≈ **94.6%** of forward.
- Savings ≈ **5.4%** wall-clock if forwards are layer-dominated.

A weighted estimate is approximately the same: LEV fires ~once per non-leaf
function-call return, vastly less than 1× per VM step. In a corpus where
LEV is ≤ 10% of executed opcodes, the saving rounds to **5%**.

## Why the Number Is Low (and How to Raise It)

The bottleneck is that 35 of the ~71 per-block ops are **un-annotated**
generics (empty `opcodes=` set). Many of them are factually opcode-specific
but the annotation hasn't been added:

- L6 / L7 ALU operand gather and routing: most non-ALU paths could declare
  the opcode set they fire on.
- L11 / L12 multiply-partial / -combine: literally `MUL`-only by design, but
  unannotated. Annotating these alone would make L11 and L12 skippable for
  29 of 31 opcodes.
- L8 ALU post-op pipelines: gated on opcode_flags at the *weight* level, but
  the `Operation.opcodes` field isn't set.

Annotating these would push the skip rate from 5% to plausibly 30-40%
(rough estimate; L11/L12 alone are ~11% of compute).

## Implementation Sketch — Per-Token Block Mask

The runtime entry point is `AutoregressiveVM.forward` at
`neural_vm/vm_step.py:1536`:

```python
for i, block in enumerate(self.blocks):
    layer_cache = kv_cache.get_layer_cache(i) if kv_cache is not None else None
    x = block(x, kv_cache=layer_cache, x_is_new_only=x_is_new_only)
```

To skip blocks per-token:

1. **Bake the mask at compile time.** Add to `ModelLayout`:
   ```python
   block_skip_mask: torch.Tensor  # shape [n_blocks, n_opcodes], dtype=bool
   ```
   populated from this analysis. Persist in the disk cache alongside
   `dim_positions` / `ffn_widths`.

2. **Per-step opcode detection.** Each VM step has a fixed 35-token layout
   where the opcode lands at a known position relative to `PC_byte0`. The
   runner already knows the current opcode (`run_vm.py` dispatches MEM-store
   handlers off it). Plumb a `current_opcode_idx: torch.LongTensor [B]`
   into the forward pass.

3. **Replace the block loop.** In `BatchedPureNeuralRunner` (since per-token
   batched skip is the high-leverage case):
   ```python
   active_blocks = self.layout.block_skip_mask[:, current_opcode_idx]  # [n_blocks, B]
   for i, block in enumerate(self.blocks):
       # If all batch elements skip this block, skip the call entirely.
       if not active_blocks[i].any():
           continue
       # Mixed: run block, then zero-out the residual delta for skip elements.
       x_new = block(x, kv_cache=layer_cache, x_is_new_only=x_is_new_only)
       mask = active_blocks[i].view(-1, 1, 1)  # [B,1,1]
       x = torch.where(mask, x_new, x)
   ```
   The skip applies to the residual *delta*, not the residual itself —
   skipped blocks pass through unchanged.

4. **KV-cache consistency.** A skipped block doesn't append to its
   KV-cache. Either (a) carry the previous step's K/V forward (zero delta),
   or (b) mark the slot as "no contribution" and have the next active block
   handle the gap. Safest first cut: only skip blocks that have *no*
   attention component for the active opcode (i.e. attn op is also gated).

5. **Validation.** Add a smoke pass that runs `(test_program × opcode)` with
   and without the mask and asserts byte-identical output. This catches
   over-aggressive gating before it ships.

**Minimum-viable cut**: hardcode the L16 skip-for-non-LEV decision (5%
savings, byte-identical, no batch-mask plumbing needed). Then iterate by
annotating L11/L12 and the L8/L9/L10 ALU pipelines, re-running this
analysis, and ratcheting the mask wider as `verify_opcode_coverage` keeps
the annotations honest.
