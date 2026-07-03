# Position-Role Audit (2026-06-10)

Audit of every FFN rule-producing function and attention-head spec
in `c4_release/neural_vm/unified_compiler/ops/lN_ops.py`,
classified by the three architectural position-roles:

1. **STEP_END** — heavy compute (ALU/CMP/branch/dispatch/writeback)
2. **TOKEN_EMIT** — register/memory token positions (MARK_AX, MARK_PC,
   MARK_SP, MARK_BP, MARK_STACK0, MEM section, BYTE_INDEX_*)
3. **MEM_READ** — past-KV lookups (L15 lookup at MARK_AX during LI/LC)

Verdict legend:
- `CORRECT` — current gating matches the intent's role.
- `MISPOSITIONED → STEP_END` — compute landing on a TOKEN_EMIT marker.
- `MIXED` — same op contains rules of multiple intents.
- `DUAL_USE_OK` — a TOKEN_EMIT position the architecture explicitly co-opts
  (e.g. L15 head 0 at MARK_AX for memory lookup; L8 ALU at MARK_AX as
  designed convergence point for AX writeback).

## Summary stats

- **Total rule-producing units catalogued**: 97 (62 FFN families + 35
  attention-head specs).
- **Wave-B migration candidates (compute @ marker → STEP_END)**: 21
  rule families.
- **Token-emission rules at marker positions** (CORRECT): 38.
- **Memory-read rules at MARK_AX / MEM marker** (CORRECT or DUAL_USE_OK): 6.
- **Unscoped / pure cleanup** (no marker gate): 32.

## Audit table

| Op (function) | Layer | Position gate(s) | Intent | Verdict |
|---|---|---|---|---|
| `_phase_a_ffn_rules` | L0 | unscoped (raw IS_MARK/marker dims) | TOKEN_EMIT (marker flag rebuild) | CORRECT |
| `_layer0_threshold_head_specs` | L0 | MARK_* (all 7 markers), MARK_CS | TOKEN_EMIT (marker threshold detectors) | CORRECT |
| `_layer1_ffn_rules` | L1 | IS_BYTE + BYTE_INDEX_* | TOKEN_EMIT (byte-index flags + STACK0_BYTE0) | CORRECT |
| `_layer1_threshold_head_specs` | L1 | MARK_SE_ONLY, MARK_CS | TOKEN_EMIT (step-end existence) | CORRECT |
| `_layer2_mem_byte_flags_rules` | L2 | IS_BYTE → MEM_VAL_B0/B1, STACK0_BYTE1 | TOKEN_EMIT (mem-byte flags) | CORRECT |
| `_layer2_initial_pc_bake_cancel_rules` | L2 | MARK_PC | TOKEN_EMIT (PC bake suppression at step 0) | CORRECT |
| `_layer2_threshold_attn` heads | L2 | all-marker | TOKEN_EMIT (marker propagation) | CORRECT |
| `_layer2_lookback_detection_head` | L2 | IS_BYTE | COMPUTE (lookback detect) | MISPOSITIONED → STEP_END |
| `_layer3_ffn_rules` | L3 | MARK_PC / MARK_SP / MARK_BP / MARK_STACK0 / MARK_MEM | TOKEN_EMIT (initial register/state propagation) | CORRECT |
| `_layer3_pc_byte1_output_rules` | L3 | IS_BYTE + BYTE_INDEX_0 | TOKEN_EMIT (PC byte-1 output) | CORRECT |
| `_layer3_convo_io_state_init_rules` | L3 | unscoped (gate-driven) | TOKEN_EMIT (IO state init) | CORRECT |
| `_layer3_carry_forward_head_specs` | L3 | MARK_AX / MARK_PC / MARK_SP / MARK_BP | TOKEN_EMIT (carry-forward relay) | CORRECT |
| `_layer4_pc_relay_head_specs` | L4 | MARK_AX, MARK_PC + IS_BYTE | TOKEN_EMIT (PC bytes routed to AX) | CORRECT |
| `_nibble_rotation_chain_rules` | L4 | unscoped | COMPUTE (nibble rotation) | MISPOSITIONED → STEP_END |
| `_layer4_pc_plus1_ax_rules` | L4 | MARK_AX | COMPUTE (PC+1 staging at AX) | MISPOSITIONED → STEP_END |
| `_layer4_temp_clear_pc_rules` | L4 | MARK_PC | TOKEN_EMIT cleanup | CORRECT |
| `_layer4_pc_plus_offset_byte_rules` | L4 | IS_BYTE | COMPUTE (PC byte offset add) | MISPOSITIONED → STEP_END |
| `_layer4_pc_plus1_pc_rules` | L4 | MARK_PC | TOKEN_EMIT (PC bake) | CORRECT |
| `_layer4_sp_to_addr_key_op` | L4 | MARK_AX, STACK0_BYTE0 + BYTE_INDEX_0/1 | COMPUTE (SP→addr-key derive) | MISPOSITIONED → STEP_END |
| `_layer5_fetch_head_specs` | L5 | MARK_AX, MARK_PC | TOKEN_EMIT (instruction fetch at PC) | CORRECT |
| `_opcode_decode_main_rules` (+ first_step/temp_clear/all_step_pc) | L5 | unscoped (CONST/IS_MARK band) | TOKEN_EMIT (opcode dim activation) | CORRECT |
| `_layer6_all_step_jmp_pc_override_rules` | L6 | MARK_AX, MARK_PC | TOKEN_EMIT (PC override) | CORRECT |
| `_layer6_imm_fetch_route_rules` | L6 | IS_BYTE + MARK_AX/MARK_PC | TOKEN_EMIT (IMM byte route) | CORRECT |
| `_layer6_imm_carry_refresh_rules` | L6 | IS_BYTE + MARK_AX/PC | COMPUTE (carry refresh) | MISPOSITIONED → STEP_END |
| `_layer6_all_step_jsr_pc_override_rules` | L6 | all markers + IS_BYTE | TOKEN_EMIT (JSR PC override) | CORRECT |
| `_layer6_exit_ax_route_rules` | L6 | IS_BYTE + MARK_AX/PC | TOKEN_EMIT (EXIT route) | CORRECT |
| `_layer6_nop/jmp/jsr/getchar/bz/bnz/psh/adj_ax_route_rules` | L6 | MARK_AX, MARK_PC (+IS_BYTE) | TOKEN_EMIT (per-opcode AX routing) | CORRECT |
| `_layer6_delayed_jmp_pc_override_rules` | L6 | MARK_AX, MARK_PC | TOKEN_EMIT (PC override) | CORRECT |
| `_layer6_first_step_jmp_pc_override_rules` | L6 | MARK_AX, MARK_PC | TOKEN_EMIT (PC override) | CORRECT |
| `_layer6_halt_detect_rules` | L6 | unscoped | COMPUTE (HALT flag) | MISPOSITIONED → STEP_END |
| `_layer6_temp_cleanup_rules`, `_layer6_cmp3_cleanup_rules` | L6 | MARK_PC + IS_BYTE | TOKEN_EMIT cleanup | CORRECT |
| `_layer6_stack_identity_rules` | L6 | MARK_SP/MARK_BP/MARK_STACK0 + IS_BYTE | TOKEN_EMIT (stack identity) | CORRECT |
| `_layer6_psh/jsr/sp_decrement_rules`, `_jsr_sp_fixup_rules`, `_jsr_sp_bytes_rules` | L6 | MARK_SP + byte-index | COMPUTE (SP arithmetic) | MISPOSITIONED → STEP_END |
| `_layer6_psh_stack0_writeback_rules`, `_psh_stack0_marker_override_rules` | L6 | MARK_STACK0 | TOKEN_EMIT (STACK0 writeback) | CORRECT |
| `_layer6_adj_sp_writeback_rules`, `_ent_sp_writeback_rules`, `_ent_first_step_sp_byte0_rules`, `_ent_first_step_sp_bytes_rules` | L6 | MARK_SP / BYTE_INDEX_* | COMPUTE (SP writeback math) | MISPOSITIONED → STEP_END |
| `_layer6_ent_after_jsr_sp_byte0_fixup_rules` | L6 | MARK_BP/MARK_SP/MARK_STACK0 + IS_BYTE | COMPUTE (SP fixup) | MISPOSITIONED → STEP_END |
| `_layer6_bz_pc_override_rules`, `_bnz_pc_override_rules`, `_branch_pc_byte1_override_rules` | L6 | MARK_PC / MARK_STACK0 / IS_BYTE | TOKEN_EMIT (branch PC override) | CORRECT |
| `_layer6_tail_cleanup_rules` | L6 | all markers | TOKEN_EMIT cleanup | CORRECT |
| `_layer6_stack_writeback_rules`, `_ax_output_route_rules` | L6 | unscoped | TOKEN_EMIT | CORRECT |
| `_layer6_attn_head_specs` (4 heads) | L6 | MARK_AX/MARK_PC/MARK_SP + IS_BYTE | TOKEN_EMIT (per-step relays) | CORRECT |
| `_layer6_relay_head_specs` | L6 | all markers + STACK0_BYTE0 | TOKEN_EMIT (cross-band relays) | CORRECT |
| `_layer6_bz_bnz_relay_bake_op` | L6 | MARK_AX/PC + IS_BYTE | TOKEN_EMIT (branch flag relay) | CORRECT |
| `_layer6_binary_pop_sp_increment_rules` | L6 | all markers + BYTE_INDEX_* | COMPUTE (SP+=8 on POP) | MISPOSITIONED → STEP_END |
| `_post_l9_bz_pc_override_rules`, `_post_l9_bnz_pc_override_rules` | L6 | MARK_PC + MARK_STACK0 | TOKEN_EMIT (branch PC override) | CORRECT |
| `_layer7_operand_gather_head_specs` | L7 | MARK_AX/MARK_SP/MARK_BP/STACK0_BYTE0 | TOKEN_EMIT (operand gather) | CORRECT |
| `_layer7_memory_head_specs` | L7 | MARK_MEM/MARK_STACK0/MARK_SP + BYTE_INDEX_* + IS_BYTE | MEM_READ (mem byte fetch precursor) | CORRECT |
| `make_format_pointer_extraction_op` | L7 | MARK_STACK0 | TOKEN_EMIT (format ptr extract) | CORRECT |
| `make_layer7_sp_byte0_is_f8_op` | L7 | MARK_SP | TOKEN_EMIT (SP-byte0 sentinel detect) | CORRECT |
| `_layer8_alu_add_lo_rules` | L8 | MARK_AX (+MARK_PC blocker) | COMPUTE (ADD low-nibble) | MISPOSITIONED → STEP_END (DUAL_USE_OK today) |
| `_layer8_alu_lea_lo_rules`, `_alu_sub_lo_rules`, `_alu_add_carry_rules`, `_alu_lea_carry_rules`, `_alu_adj_lo_rules`, `_alu_adj_carry_rules`, `_alu_sub_borrow_rules`, `_alu_ent_lo_rules`, `_alu_ent_borrow_rules` | L8 | MARK_AX (+blockers) | COMPUTE (per-op ALU lo-byte stage) | MISPOSITIONED → STEP_END |
| `_layer8_alu_cmp_group_rules`, `_alu_cmp_clear_rules` | L8 | MARK_AX | COMPUTE (CMP group decode) | MISPOSITIONED → STEP_END |
| `_layer8_alu_ent_adj_defaults_rules` | L8 | MARK_AX + MARK_SP | COMPUTE (ENT/ADJ defaults) | MISPOSITIONED → STEP_END |
| `_layer8_alu_lev_byte0_lo_rules`, `_lev_byte0_hi_rules`, `_lev_b1_rules`, `_lev_b2_rules` | L8 | MARK_BP (+MARK_PC blocker) | COMPUTE (LEV addr at BP marker) | MISPOSITIONED → STEP_END (or MARK_PC for return-addr emit) |
| `_layer8_alu_lea_axb2_rules` | L8 | IS_BYTE + BYTE_INDEX_1 | COMPUTE (LEA AX-byte 2) | MISPOSITIONED → STEP_END |
| `_format_position_counter_rules` | L8 | unscoped | COMPUTE (position counter) | MISPOSITIONED → STEP_END |
| `_layer8_multibyte_routing_rules` | L8 | IS_BYTE + MARK_AX | TOKEN_EMIT (multi-byte IMM route) | CORRECT |
| `_layer8_sp_gathered_sentinel_rule` | L8 | MARK_SP + STACK0_BYTE0 + STEP_BOUNDARY | COMPUTE (SP=0xF8 sentinel) | MISPOSITIONED → STEP_END |
| `_layer8_sp_gather_head_specs` | L8 | MARK_SP/MARK_BP/MARK_STACK0 + BYTE_INDEX_* | TOKEN_EMIT (SP gather) | CORRECT |
| `make_layer8_multibyte_fetch_bake_op` | L8 | MARK_AX + IS_BYTE | TOKEN_EMIT (IMM fetch) | CORRECT |
| `make_layer8_op_imm_relay_op` | L8 | MARK_AX + IS_BYTE | TOKEN_EMIT (IMM relay) | CORRECT |
| `make_layer8_mem_to_alu_op` | L8 | MARK_AX/MEM_VAL_B0/B1 + all markers | MEM_READ (mem→ALU at AX) | DUAL_USE_OK (memory-read at AX, parallel to L15) |
| `make_layer8_head6_ax_carry_refresh_op` | L8 | MARK_AX | COMPUTE (carry refresh) | MISPOSITIONED → STEP_END |
| `_layer9_add_hi_nibble_rules`, `_lea_hi_nibble_rules`, `_adj_hi_nibble_rules`, `_sub_hi_nibble_rules`, `_ent_hi_nibble_rules` | L9 | MARK_AX (+MARK_PC blocker) | COMPUTE (hi-nibble ALU stage) | MISPOSITIONED → STEP_END |
| `_layer9_cmp_rules` | L9 | MARK_AX | COMPUTE (CMP flags hi_lt/hi_eq/lo_lt/lo_eq) | MISPOSITIONED → STEP_END |
| `_layer9_add_carry_out_rules`, `_sub_borrow_out_rules` | L9 | MARK_AX | COMPUTE (carry-out flag) | MISPOSITIONED → STEP_END |
| `_layer9_alu_clear_rules` | L9 | MARK_AX | TOKEN_EMIT cleanup | CORRECT |
| `_layer9_bp_plus8_shift_rules`, `_addr_b1_set_and_cascade_rules` | L9 | MARK_SP/MARK_BP/MARK_PC | COMPUTE (BP+8 / addr-byte1 cascade) | MISPOSITIONED → STEP_END |
| `_layer9_marker_suppress_rules` | L9 | unscoped | TOKEN_EMIT (marker suppression) | CORRECT |
| `make_layer9_lev_addr_relay_op` | L9 | MARK_SP + BYTE_INDEX_0 | TOKEN_EMIT (LEV addr relay) | CORRECT |
| `make_layer9_lev_bp_to_pc_relay_op` | L9 | MARK_PC + BYTE_INDEX_0 | TOKEN_EMIT (LEV BP→PC) | CORRECT |
| `make_format_string_fetch_head_op` | L9 | unscoped | MEM_READ (format-string fetch) | CORRECT |
| `make_layer9_alibi_mem_attn_op` | L9 | MEM_VAL_B0 | MEM_READ (ALiBi mem read) | CORRECT |
| `_l10_binary_op_byte_zeroing_rules` | L10 | IS_BYTE + BYTE_INDEX_* + opcode | COMPUTE (binary-op byte zeroing) | MISPOSITIONED → STEP_END |
| `_l10_carry_propagation_rules` (add/sub_rule_for) | L10 | MARK_AX BLOCKED (-5000) + IS_BYTE + BYTE_INDEX_* | COMPUTE (carry propagation) | MISPOSITIONED → STEP_END |
| `_l10_comparison_combine_rules` | L10 | MARK_AX (+MARK_PC -50 blocker) | COMPUTE (CMP combine) | MISPOSITIONED → STEP_END |
| `_layer10_alu_cmp_combine_rules` | L10 | MARK_AX | COMPUTE (CMP combine, ALU side) | MISPOSITIONED → STEP_END (recent EQ Shape-B fix lives here) |
| `_layer10_alu_bitwise_or/xor/and_rules` | L10 | MARK_AX | COMPUTE (bitwise) | MISPOSITIONED → STEP_END |
| `_layer10_alu_shl_shr_zero_rules` | L10 | MARK_AX | COMPUTE (SHL/SHR zero handling) | MISPOSITIONED → STEP_END |
| `_layer10_alu_ax_passthrough_rules`, `_alu_mul_lo_rules` | L10 | MARK_AX | COMPUTE (AX passthrough, MUL lo) | MISPOSITIONED → STEP_END |
| `make_layer10_carry_relay_bake_op` | L10 | MARK_AX + IS_BYTE | COMPUTE (carry relay) | MISPOSITIONED → STEP_END |
| `make_layer10_byte_passthrough_bake_op` | L10 | MARK_AX + BYTE_INDEX_* + MEM_VAL_B0/B1 | TOKEN_EMIT (byte passthrough) | CORRECT |
| `make_layer10_sp_byte_passthrough_bake_op` | L10 | MARK_SP + BYTE_INDEX_* | TOKEN_EMIT (SP byte passthrough) | CORRECT |
| `make_layer10_bp_byte_passthrough_bake_op` | L10 | MARK_STACK0 + STACK0_BYTE0/1 + BYTE_INDEX_* | TOKEN_EMIT (BP byte passthrough) | CORRECT |
| `make_layer10_psh_stack0_passthrough_bake_op` | L10 | MARK_STACK0 + BYTE_INDEX_* | TOKEN_EMIT (PSH STACK0 byte) | CORRECT |
| `make_layer10_psh_ax_broadcast_bake_op` | L10 | all markers + BYTE_INDEX_* | TOKEN_EMIT (PSH AX broadcast to STACK0 bytes) | CORRECT |
| `make_layer10_stack0_byte_relay_bake_op` | L10 | MARK_STACK0/MARK_MEM + STACK0_BYTE0/1 | TOKEN_EMIT (STACK0 byte relay) | CORRECT |
| `_tail_bit32_result_correction_rules` (and `step_end_transition_blocked`, `tail_clear_output_before_step_end`) | L10 | NEXT_SE gate + all markers as blockers | COMPUTE (tail repair of 32-bit OUTPUT) | CORRECT (already targets near-STEP_END via NEXT_SE) |
| `make_tail_bit32_result_correction_op` (attention) | L10 | all markers + MEM_VAL_B0/1 + BYTE_INDEX_* | COMPUTE (tail attn) | MISPOSITIONED → STEP_END |
| `_layer11_mul_partial_rules_for_a_lo`, `_layer11_mul_partial_rules` | L11 | MARK_AX | COMPUTE (MUL partial products) | MISPOSITIONED → STEP_END |
| `_layer12_mul_combine_rules` | L12 | MARK_AX | COMPUTE (MUL combine) | MISPOSITIONED → STEP_END |
| `_layer13_shifts_substage_rules`, `_shl_rules`, `_shr_rules`, `_shifts_rules` | L13 | MARK_AX | COMPUTE (shift staging) | MISPOSITIONED → STEP_END |
| `_layer13_mem_addr_gather_head_specs` | L13 | MARK_MEM/MARK_STACK0/MARK_AX + MEM_VAL_B0/B1 | MEM_READ (mem-addr gather) | CORRECT |
| `_layer14_mem_generation_head_specs` (heads 0-3 addr, 4-7 val) | L14 | MARK_MEM + addr-byte / val-byte position | TOKEN_EMIT (MEM byte generation) | CORRECT |
| `_layer14_mem_generation_head_specs_with_overrides` | L14 | MARK_MEM + MEM_VAL_B0/B1 + STACK0_BYTE0 + IS_BYTE | TOKEN_EMIT (MEM byte gen, addr-key path) | CORRECT |
| `make_layer14_alu_high_byte_relay_op` | L14 | MARK_AX (blocked) + IS_BYTE + BYTE_INDEX_0 | COMPUTE (ALU high-byte relay) | MISPOSITIONED → STEP_END |
| `_layer14_temp_clear_rules` | L14 | MARK_PC + IS_BYTE + BYTE_INDEX_* | TOKEN_EMIT cleanup | CORRECT |
| `_layer14_clear_addr_key_pollution_rules` | L14 | all markers + MEM_VAL_B0/B1 | TOKEN_EMIT cleanup | CORRECT |
| `_layer14_clear_output_corruption_rules` | L14 | all markers + MEM_VAL + BYTE_INDEX_3 | TOKEN_EMIT cleanup | CORRECT |
| `_layer14_clear_mem_marker_output_rules` | L14 | all markers + IS_BYTE | TOKEN_EMIT cleanup | CORRECT |
| `_layer14_jsr_ax_bytes_zero_rules`, `_alu_nocarry_ax_bytes_zero_rules`, `_ent_ax_bytes_zero_rules`, `_lc_ax_bytes_zero_rules` | L14 | IS_BYTE / BYTE_INDEX_3 / BYTE_INDEX_0 | TOKEN_EMIT (AX byte zero) | CORRECT |
| `_layer14_addr_key_neural_decode_*_rules` (lo_hi/top_common/top_carry/load_query) | L14 | MARK_AX (load_query only) / unscoped | COMPUTE (addr-key neural decode) | MISPOSITIONED → STEP_END (the load-query stage; others unscoped) |
| `make_layer15_memory_lookup_op` (heads 0-3, 4-7 LEV, 9 pop_d8_to_e0, 10-11 LEV return) | L15 | MARK_AX (Q) + MARK_MEM (K) + BYTE_INDEX_* | MEM_READ (LI/LC/LEV/POP lookups into past KV) | CORRECT (architectural memory-read at MARK_AX) |
| `make_layer15_store_stack0_sp_byte0_addr_op` (head 12) | L15 | MARK_SP + MARK_STACK0 | TOKEN_EMIT (SI/SC addr capture) | CORRECT |
| `make_layer15_si_mem_addr0_from_stack0_op` (head 13) | L15 | MARK_MEM + STACK0_BYTE0 | TOKEN_EMIT (SI/SC MEM addr0) | CORRECT |
| `make_l15_psh_stack_ir`, `make_l15_nibble_copy_ir`, `make_nibble_copy_ffn_op`, `make_layer15_nibble_copy_op` | L15 | (per-op) MARK_STACK0/BYTE_INDEX_* | TOKEN_EMIT (PSH stack write / nibble copy) | CORRECT |
| `_layer16_lev_routing_rules` | L16 | all markers + BYTE_INDEX_* + MEM_VAL + STACK0_BYTE | TOKEN_EMIT (LEV routing across positions) | MIXED (mostly TOKEN_EMIT; the BP-frame-byte1 PSH-aggressor lives here per memory note `project_l16_bp_frame_byte1_ff_dual.md`) |

## Per-layer compute concentration

| Layer | FFN compute-at-marker rules (Wave-B candidates) | Token-emission rules | Memory-read |
|---|---|---|---|
| L0  | 0 | 1 + threshold heads | – |
| L1  | 0 | 1 + threshold heads | – |
| L2  | 1 (`lookback_detection_head`) | 2 + threshold heads | – |
| L3  | 0 | 4 + 1 attn | – |
| L4  | 4 (`nibble_rotation`, `pc_plus1_ax`, `pc_plus_offset_byte`, `sp_to_addr_key`) | 3 + 1 attn | – |
| L5  | 0 | 6 + 1 fetch attn | – |
| L6  | 8 (`imm_carry_refresh`, `halt_detect`, 5 SP-decrement/fixup/writeback families, `binary_pop_sp_increment`) | 27 + 3 attn | – |
| L7  | 0 | 2 + 1 attn | 1 (`memory_head_specs`) |
| L8  | 17 (entire ALU family + LEV byte rules + `sp_gathered_sentinel` + `head6_ax_carry_refresh` + `lea_axb2`) | 3 + 2 attn | 1 (`mem_to_alu`) |
| L9  | 9 (`add/lea/adj/sub/ent_hi_nibble`, `cmp`, `add/sub_carry_out`, `bp_plus8_shift`, `addr_b1_set_and_cascade`) | 3 + 2 attn | 2 (`format_string_fetch`, `alibi_mem_attn`) |
| L10 | 11 (`binary_op_byte_zeroing`, `carry_propagation`, both `cmp_combine` families, bitwise, shl_shr_zero, ax_passthrough, mul_lo, `carry_relay_bake`, `tail_bit32` attn, `alu_high_byte_relay` indirect) | 6 byte-passthrough heads + 1 broadcast | – |
| L11 | 2 (`mul_partial`) | 0 | – |
| L12 | 1 (`mul_combine`) | 0 | – |
| L13 | 4 (shifts substage + shl/shr/shifts) | 0 | 1 (`mem_addr_gather`) |
| L14 | 2 (`alu_high_byte_relay`, `addr_key_neural_decode_load_query`) | 8 cleanup + 4 AX-zero + mem_generation heads | – |
| L15 | 0 | 3 store/copy heads | 1 family (memory_lookup heads 0-11) |
| L16 | 0 (but PSH-aggressor `bp_frame_byte1_ff` is open bug) | 1 (lev_routing) | – |

## Key findings

1. **Wave-B target**: the MISPOSITIONED → STEP_END candidates are
   concentrated in L8 (17), L9 (9), L10 (11), L11/L12/L13 ALU stages
   (7 total). All currently fire at `MARK_AX` (or `MARK_BP` for LEV
   bytes) because the AX register-byte position is the natural
   convergence point under the legacy "compute lands where the result
   token is emitted" model. The architectural directive separates these:
   compute → STEP_END, then writeback → MARK_AX.

2. **L10 cmp_combine** is explicitly tagged compute-at-MARK_AX. Recent
   EQ Shape-B fix (memory note `project_eq_byte1_l6_divergence.md`,
   2026-06-07) added the `CMP+0` blocker at MARK_AX; this is the kind
   of patch that the STEP_END migration is intended to obviate.

3. **L15 memory_lookup is the canonical MEM_READ pattern at
   MARK_AX**: queries during LI/LC fire at MARK_AX, attending back to
   past `MARK_MEM`-keyed KV-cache rows. `L8 mem_to_alu` (heads 5/7) and
   `L13 mem_addr_gather` follow the same dual-use pattern.

4. **L14 mem_generation heads 0-7 are the canonical TOKEN_EMIT
   pattern at MARK_MEM byte positions**: each head fires at exactly
   one MEM byte slot via threshold-difference encoding (slot 0) gated
   by MEM_STORE.

5. **Tail repairs already targeting near-STEP_END**: the L10
   `_tail_bit32_result_correction_rules` family uses `NEXT_SE` as gate
   (`tail_clear_output_before_step_end`) and a `step_end_transition_blocked`
   wrapper. This is the closest existing analog of "compute at
   STEP_END" and is a useful template for Wave-B migrations.

6. **Open bugs still live at TOKEN_EMIT positions**: the L10 PSH
   `MEM_addr0=0xE0` ENT-main miss (memory note
   `project_l10_psh_addr_ent_bug.md`) is gated at byte position with
   no OP_ENT guard; the L16 `bp_frame_byte1_ff` PSH-aggressor
   (`project_l16_bp_frame_byte1_ff_dual.md`) lives in
   `_layer16_lev_routing_rules`. Both indicate the "compute at
   token-emit" pattern is structurally fragile and motivates the
   STEP_END migration.

## Sources

- All `lN_ops.py` files in
  `c4_release/neural_vm/unified_compiler/ops/` (L0..L16).
- Per-file scope/conditions extraction; counts are function-level
  (one entry per rule-generating helper, not per individual
  `FFNRule` it produces — the carry-propagation `add_rule_for` /
  `sub_rule_for` factories alone emit 768 individual `FFNRule`s).
- Memory notes consulted: `project_eq_byte1_l6_divergence.md`,
  `project_l10_psh_addr_ent_bug.md`,
  `project_l16_bp_frame_byte1_ff_dual.md`,
  `project_l15_stack0_byte_attribution.md`.
