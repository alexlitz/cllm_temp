# Phase 11.C audit — DeclarativeAttentionHeadSpec K-gate classification

- Total heads scanned: **63**
- K-side categories:
  - `k_gates_on_op`: **1**
  - `k_gates_on_mark_only`: **19**
  - `k_no_mark_no_op`: **43**
- Q-side has OP_* gate: **7**
- **Easy migrations** (K=MARK_only AND Q implies OP_): **2**

## Per-head verdicts

| file | line | head_idx | enclosing | category | q_implies_op | q_marks | q_ops | k_marks | k_ops |
|---|---|---|---|---|---|---|---|---|---|
| control_flow_heads.py | 172 | head_idx | _lev_detector_head_spec | k_no_mark_no_op | no | - | - | - | - |
| flag_gated_ops.py | 373 | 5 | _tool_call_relay_head_spec | k_gates_on_mark_only | no | MARK_AX | - | MARK_AX | - |
| flag_gated_ops.py | 1294 | 4 | _convo_io_prtf_transport_spec | k_gates_on_mark_only | no | - | - | MARK_AX | - |
| flag_gated_ops.py | 459 | 4 | _convo_io_relay_head_specs | k_gates_on_mark_only | no | MARK_AX | - | MARK_AX | - |
| flag_gated_ops.py | 470 | 5 | _convo_io_relay_head_specs | k_gates_on_mark_only | no | MARK_AX | - | MARK_AX | - |
| l10_ops.py | 1058 | _l10_head_idx('layer10_carry_relay_bake.head_0') | _layer10_carry_relay_head_spec | k_gates_on_mark_only | no | - | - | MARK_AX | - |
| l10_ops.py | 1130 | head_idx | _byte_passthrough_chain_spec | k_no_mark_no_op | no | - | - | - | - |
| l10_ops.py | 1347 | spec.head_idx | _layer10_sp_byte_passthrough_head_spec | k_no_mark_no_op | YES | MARK_SP | OP_ENT,OP_JSR | - | - |
| l10_ops.py | 1557 | _l10_head_idx('layer10_psh_stack0_passthrough_bake.head_3') | _layer10_psh_stack0_passthrough_head_spec | k_no_mark_no_op | no | - | - | - | - |
| l10_ops.py | 1631 | _l10_head_idx('layer10_stack0_byte_relay_bake.head_4') | _layer10_stack0_byte_relay_head_spec | k_no_mark_no_op | no | - | - | - | - |
| l10_ops.py | 1693 | _l10_head_idx('layer10_stack0_byte_relay_bake.head_5') | _layer10_nonbitwise_stack0_byte_relay_head_spec | k_no_mark_no_op | no | - | - | - | - |
| l10_ops.py | 1805 | _l10_head_idx('layer10_stack0_byte_relay_bake.head_6') | _layer10_stack0_persistence_head_spec | k_no_mark_no_op | no | - | - | - | - |
| l13_ops.py | 367 | head_idx | _layer13_mem_addr_gather_head_specs | k_no_mark_no_op | no | - | - | - | - |
| l14_ops.py | 854 | 8 | _layer14_alu_high_byte_relay_spec | k_no_mark_no_op | no | - | - | - | - |
| l14_ops.py | 512 | _L14_HEAD_LAYOUT_BY_NAME[f'layer14_mem_generation.head_{h}'] | _layer14_mem_generation_head_specs | k_no_mark_no_op | no | - | - | - | - |
| l14_ops.py | 606 | _L14_HEAD_LAYOUT_BY_NAME[f'layer14_mem_generation.head_{head_idx}'] | _layer14_mem_generation_head_specs | k_no_mark_no_op | no | - | - | - | - |
| l15_ops.py | 625 | _l15_head_idx('layer15_store_stack0_sp_byte0_addr') | _layer15_store_stack0_sp_byte0_addr_spec | k_no_mark_no_op | no | - | - | - | - |
| l15_ops.py | 734 | _l15_head_idx('layer15_si_mem_addr0_from_stack0') | _layer15_si_mem_addr0_from_stack0_spec | k_no_mark_no_op | no | - | - | - | - |
| l1_ops.py | 492 | h_has_se | _layer1_threshold_ir | k_no_mark_no_op | no | - | - | - | - |
| l1_ops.py | 503 | h_in_step_fresh | _layer1_threshold_ir | k_no_mark_no_op | no | - | - | - | - |
| l1_ops.py | 365 | h_has_se | bake | k_no_mark_no_op | no | - | - | - | - |
| l1_ops.py | 386 | h_in_step_fresh | bake | k_no_mark_no_op | no | - | - | - | - |
| l2_ops.py | 713 | _l2_head_idx('layer2_lookback_detection_head') | _layer2_lookback_detection_head_spec | k_no_mark_no_op | no | - | - | - | - |
| l3_ops.py | 1286 | head_idx | _carry_forward_head_spec | k_no_mark_no_op | no | - | - | - | - |
| l3_ops.py | 1389 | 4 | _stack0_carry_head_spec | k_no_mark_no_op | no | - | - | - | - |
| l3_ops.py | 1416 | 5 | _ax_full_relay_head_spec | k_no_mark_no_op | no | - | - | - | - |
| l3_ops.py | 1447 | 6 | _lev_bp_to_pc_head_spec | k_no_mark_no_op | no | - | - | - | - |
| l3_ops.py | 1482 | 7 | _pc_byte1_prev_head_spec | k_no_mark_no_op | no | MARK_PC | - | - | - |
| l4_ops.py | 323 | _l4_head_idx('layer4_pc_relay.head_0') | _layer4_pc_relay_head_specs | k_gates_on_mark_only | no | MARK_AX | - | MARK_PC | - |
| l4_ops.py | 342 | _l4_head_idx('layer4_pc_relay.head_1') | _layer4_pc_relay_head_specs | k_gates_on_mark_only | no | - | - | MARK_PC | - |
| l5_ops.py | 258 | 0 | _layer5_fetch_head_specs | k_no_mark_no_op | no | MARK_AX | - | - | - |
| l5_ops.py | 276 | 1 | _layer5_fetch_head_specs | k_no_mark_no_op | no | MARK_AX | - | - | - |
| l5_ops.py | 294 | 2 | _layer5_fetch_head_specs | k_no_mark_no_op | no | MARK_PC | - | - | - |
| l5_ops.py | 312 | 3 | _layer5_fetch_head_specs | k_no_mark_no_op | no | MARK_PC | - | - | - |
| l5_ops.py | 330 | 4 | _layer5_fetch_head_specs | k_no_mark_no_op | no | MARK_AX | - | - | - |
| l5_ops.py | 348 | 5 | _layer5_fetch_head_specs | k_no_mark_no_op | no | MARK_PC | - | - | - |
| l6_ops.py | 3795 | _L6_HEAD_LAYOUT_BY_NAME['layer6_bz_bnz_relay_bake.head_4'] | _layer6_bz_bnz_relay_head_spec | k_no_mark_no_op | YES | MARK_AX,MARK_PC | OP_BNZ,OP_BZ | - | - |
| l6_ops.py | 3038 | _L6_HEAD_LAYOUT_BY_NAME['layer6_attn_bake.later_step_jmp_relay'] | _layer6_attn_head_specs | k_no_mark_no_op | no | - | - | - | - |
| l6_ops.py | 3047 | _L6_HEAD_LAYOUT_BY_NAME['layer6_attn_bake.exit_relay'] | _layer6_attn_head_specs | k_gates_on_mark_only | no | MARK_AX | - | MARK_AX | - |
| l6_ops.py | 3074 | _L6_HEAD_LAYOUT_BY_NAME['layer6_attn_bake.first_step_jmp_relay'] | _layer6_attn_head_specs | k_no_mark_no_op | no | - | - | - | - |
| l6_ops.py | 3083 | _L6_HEAD_LAYOUT_BY_NAME['layer6_attn_bake.first_step_jsr_relay'] | _layer6_attn_head_specs | k_gates_on_mark_only | no | MARK_AX,MARK_PC | - | MARK_AX | - |
| l6_ops.py | 3168 | _L6_HEAD_LAYOUT_BY_NAME['layer6_attn_bake.first_step_fetch_relay'] | _layer6_attn_head_specs | k_no_mark_no_op | no | - | - | - | - |
| l6_ops.py | 3274 | _L6_HEAD_LAYOUT_BY_NAME['layer6_relay_heads_bake.psh_ax_carry_lo'] | _layer6_relay_head_specs | k_no_mark_no_op | no | - | - | - | - |
| l6_ops.py | 3346 | _L6_HEAD_LAYOUT_BY_NAME['layer6_relay_heads_bake.psh_ax_carry_hi'] | _layer6_relay_head_specs | k_no_mark_no_op | no | - | - | - | - |
| l7_ops.py | 657 | _L7_HEAD_LAYOUT_BY_NAME['layer7_memory_heads.head_7'] | _format_pointer_extraction_spec | k_gates_on_mark_only | no | - | - | MARK_STACK0 | - |
| l7_ops.py | 781 | _L7_HEAD_LAYOUT_BY_NAME['layer7_memory_heads.head_6'] | _layer7_sp_byte0_is_f8_spec | k_no_mark_no_op | no | - | - | - | - |
| l7_ops.py | 240 | _L7_HEAD_LAYOUT_BY_NAME['layer7_operand_gather.head_0'] | _layer7_operand_gather_head_specs | k_no_mark_no_op | YES | MARK_AX | OP_ADJ,OP_ENT,OP_LEA | - | - |
| l7_ops.py | 263 | _L7_HEAD_LAYOUT_BY_NAME['layer7_operand_gather.head_1'] | _layer7_operand_gather_head_specs | k_gates_on_mark_only | YES | MARK_AX | OP_ADJ,OP_ENT,OP_LEA | MARK_BP,MARK_SP | - |
| l7_ops.py | 423 | _L7_HEAD_LAYOUT_BY_NAME['layer7_memory_heads.head_7'] | _layer7_memory_head_specs | k_gates_on_mark_only | no | MARK_MEM | - | MARK_MEM | - |
| l7_ops.py | 491 | _L7_HEAD_LAYOUT_BY_NAME['layer7_memory_heads.head_5'] | _layer7_memory_head_specs | k_gates_on_mark_only | no | MARK_AX | - | MARK_AX | - |
| l7_ops.py | 532 | _L7_HEAD_LAYOUT_BY_NAME['layer7_memory_heads.head_6'] | _layer7_memory_head_specs | k_gates_on_mark_only | no | MARK_SP,MARK_STACK0 | - | MARK_SP,MARK_STACK0 | - |
| l7_ops.py | 461 | head | _layer7_memory_head_specs | k_no_mark_no_op | no | MARK_AX | - | - | - |
| l8_ops.py | 1444 | _L8_HEAD_LAYOUT_BY_NAME['layer8_multibyte_fetch_bake.head_3'] | _layer8_multibyte_fetch_head_spec | k_gates_on_mark_only | no | - | - | MARK_AX | - |
| l8_ops.py | 2091 | _L8_HEAD_LAYOUT_BY_NAME['layer8_op_imm_relay.head_4'] | _layer8_op_imm_relay_head_spec | k_gates_on_mark_only | no | - | - | MARK_AX | - |
| l8_ops.py | 1770 | _L8_HEAD_LAYOUT_BY_NAME[_sp_gather_main_names[j]] | _layer8_sp_gather_head_specs | k_no_mark_no_op | no | MARK_BP,MARK_STACK0 | - | - | - |
| l8_ops.py | 1832 | head_idx | _layer8_sp_gather_head_specs | k_no_mark_no_op | no | MARK_BP,MARK_SP | - | - | - |
| l9_ops.py | 1423 | _l9_head_idx('layer9_lev_addr_relay') | _layer9_lev_addr_relay_head_spec | k_no_mark_no_op | YES | MARK_SP | OP_LEV | - | - |
| l9_ops.py | 1459 | _l9_head_idx('layer9_lev_bp_to_pc_relay') | _layer9_lev_bp_to_pc_relay_head_spec | k_no_mark_no_op | YES | MARK_PC | OP_LEV | - | - |
| l9_ops.py | 1574 | 0 | _format_string_fetch_head_spec | k_no_mark_no_op | no | - | - | - | - |
| model_ops.py | 692 | 5 | _function_call_l5_head_specs | k_gates_on_mark_only | no | MARK_STACK0 | - | MARK_BP | - |
| model_ops.py | 707 | 6 | _function_call_l5_head_specs | k_gates_on_mark_only | YES | MARK_BP | OP_ENT | MARK_SP | - |
| model_ops.py | 753 | 7 | _function_call_l6_head_spec | k_gates_on_op | no | MARK_AX,MARK_STACK0 | - | MARK_PC | OP_JSR |
| model_ops.py | 1080 | 6 | _opcode_relay_head_spec | k_gates_on_mark_only | no | MARK_AX,MARK_BP,MARK_MEM,MARK_PC,MARK_SP,MARK_STACK0 | - | MARK_AX | - |

## Easy migration candidates

Heads where K-side is `MARK_*` only AND Q-side already pins an OP_*. 
These are *structural* candidates -- a direct K-side OP_* addition would
further constrain the K-gate to opcode-discriminative behaviour, matching
the Phase 11.C goal. **NOTE:** structural candidacy does not imply
byte-identity-safe migration: the OP_* dim must be non-zero at the
K-attended residual position at the head's layer. Many candidates fail
this check because the OP_* marker is only written by a *later* layer.

| file | line | head_idx | enclosing | q_ops | k_marks (current) | notes |
|---|---|---|---|---|---|---|
| l7_ops.py | 263 | _L7_HEAD_LAYOUT_BY_NAME['layer7_operand_gather.head_1'] | _layer7_operand_gather_head_specs | OP_ADJ,OP_ENT,OP_LEA | MARK_BP,MARK_SP | structural only -- check OP_* residual timing |
| model_ops.py | 707 | 6 | _function_call_l5_head_specs | OP_ENT | MARK_SP | structural only -- check OP_* residual timing |
