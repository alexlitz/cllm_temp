"""Aggregator: builds the full list of migrated core-VM ops.

See ../migrated_ops.py for history.
"""

# Wildcard imports pull every factory name from the per-layer modules.
from .l0_ops import *  # noqa: F401,F403
from .l1_ops import *  # noqa: F401,F403
from .l2_ops import *  # noqa: F401,F403
from .l3_ops import *  # noqa: F401,F403
from .l4_ops import *  # noqa: F401,F403
from .l5_ops import *  # noqa: F401,F403
from .l6_ops import *  # noqa: F401,F403
from .l7_ops import *  # noqa: F401,F403
from .l8_ops import *  # noqa: F401,F403
from .l9_ops import *  # noqa: F401,F403
from .l10_ops import *  # noqa: F401,F403
from .l11_ops import *  # noqa: F401,F403
from .l12_ops import *  # noqa: F401,F403
from .l13_ops import *  # noqa: F401,F403
from .l14_ops import *  # noqa: F401,F403
from .l15_ops import *  # noqa: F401,F403
from .l16_ops import *  # noqa: F401,F403
from .alu_ops import *  # noqa: F401,F403
from .flag_gated_ops import *  # noqa: F401,F403
from .model_ops import *  # noqa: F401,F403


def all_core_ops(
    alu_mode: str = "lookup",
    *,
    enable_conversational_io: bool = False,
    enable_tool_calling: bool = False,
) -> list:
    """Return the full list of migrated core-VM operations.

    Doesn't include I/O operations (those need their own migration pass).
    Flag-gated ops are always registered to keep the dep graph stable; their
    bake bodies are no-ops when the flag is False.

    ``alu_mode`` is forwarded to ops whose bake action depends on whether the
    legacy lookup ALU or the efficient neural ALU owns a given operation
    (currently: SHL/SHR via ``make_layer13_shifts_op``).

    ``enable_conversational_io`` is forwarded to flag-gated convo-io ops
    (L2 lookback detection head, L3 convo-I/O state init,
    ``make_format_pointer_extraction_op``, ``make_format_position_counter_op``,
    ``make_format_string_fetch_head_op``). All are always registered; their
    bake_fn is a no-op when the flag is False.

    ``enable_tool_calling`` is forwarded to the 3 tool-call ops
    (`tool_call_opcode_decode`, `tool_call_relay_head`, `tool_call_detection`)
    which are always registered but no-op when the flag is False. This keeps
    the registration list (and hence the dep graph / layer count) stable
    across modes.

    Conversational-I/O tail ops (L10 null-terminator detection,
    L15 output routing) are also flag-gated via ``enable_conversational_io``;
    L10 null-terminator additionally requires ``alu_mode == 'lookup'`` since
    ffn10 is replaced with ``ALUAndOrXor`` in efficient mode.
    """
    return [
        make_layer0_threshold_attn_op(),
        make_layer1_threshold_attn_op(),
        make_layer2_threshold_attn_op(),
        # Conversational-I/O L2 lookback head: phase=2.1 so it bakes after
        # the L2 threshold attention (phase=2). Body is a no-op unless
        # ``enable_conversational_io`` is True.
        make_layer2_lookback_detection_head_op(
            enable_conversational_io=enable_conversational_io,
        ),
        make_layer3_carry_forward_attn_op(),
        make_phase_a_ffn_op(),
        make_layer1_ffn_op(),
        make_layer2_mem_byte_flags_op(),
        # `make_nibble_copy_ffn_op` (kind="ffn", not migrated) is kept as a
        # placeholder in the dep graph: removing it shifts the dep-graph layout
        # for L11/L12 ops. The actual L15 nibble-copy bake is performed by
        # `make_layer15_nibble_copy_op` (kind="block", layer_idx=15,
        # migrated=True) below; this kind="ffn" entry is skipped at dispatch
        # because legacy_bake is present and migrated=False.
        make_nibble_copy_ffn_op(),
        make_layer3_ffn_op(),
        make_layer3_ffn_dep_anchor_op(),
        # Conversational-I/O L3 state init: phase=3.1 so it bakes after
        # the L3 FFN (phase=3) and writes into FFN units above the L3 /
        # L6-routing unit ranges. Body is a no-op unless
        # ``enable_conversational_io`` is True.
        make_layer3_convo_io_state_init_op(
            enable_conversational_io=enable_conversational_io,
        ),
        make_layer4_pc_relay_op(),
        make_layer4_ffn_op(),
        make_layer5_fetch_op(),
        make_layer5_fetch_dep_anchor_op(),
        make_opcode_decode_ffn_op(),
        make_opcode_decode_ffn_dep_anchor_op(),
        make_layer6_attn_op(),
        make_layer6_routing_ffn_op(),
        make_layer6_relay_heads_op(),
        # 3 model-level bake ops (phase 998.5/.6/.7): the actual
        # _set_layer6_attn / _set_layer6_relay_heads / _set_bz_bnz_relay
        # weight bakes. They run AFTER function_call_weights (998) but
        # BEFORE legacy_bake (999), preserving the legacy override
        # contract on attn6 head 7 Q slots. See docstrings on the
        # individual bake ops for details.
        make_layer6_attn_bake_op(),
        make_layer6_relay_heads_bake_op(),
        make_layer6_bz_bnz_relay_bake_op(),
        make_layer7_operand_gather_op(),
        make_layer7_memory_heads_op(),
        # Convo-I/O L7 attn bake (phase=7.5). Always registered; bake is a
        # no-op when enable_conversational_io is False. See docstring for
        # phase/ordering rationale.
        make_format_pointer_extraction_op(
            enable_conversational_io=enable_conversational_io
        ),
        make_layer8_alu_op(),
        # Convo-I/O L8 FFN bake (phase=8.5). Always registered; bake is a
        # no-op when enable_conversational_io is False. Fires regardless of
        # alu_mode (the original lookup-mode nesting was incidental).
        make_format_position_counter_op(
            enable_conversational_io=enable_conversational_io
        ),
        # `make_layer8_multibyte_fetch_op` and `make_layer8_sp_gather_op` are
        # kept as kind="attn" dep anchors so the LayerCompiler topology
        # remains stable. The actual attn bakes are performed by the
        # kind="block" migrated ops below (phases 8.0 / 8.1, layer_idx=8).
        make_layer8_multibyte_fetch_op(),
        make_layer8_multibyte_routing_op(),
        make_layer8_sp_gather_op(),
        make_layer8_sp_gather_bake_op(),
        make_layer8_multibyte_fetch_bake_op(),
        make_layer9_alu_op(),
        make_layer9_lev_addr_relay_op(),
        make_layer9_lev_bp_to_pc_relay_op(),
        # ALiBi-based memory propagation attention head (phase=9.2).
        # PROOF-OF-CONCEPT for replacing _inject_mem_store / runner shadow
        # memory with attention. Registered always so the dep graph and
        # layer_idx gates see it; bake is a no-op by default (`enable=False`)
        # so existing tests are byte-identical. See l9_ops.py docstring
        # for the full design and slope-tuning analysis.
        make_layer9_alibi_mem_attn_op(enable=False),
        # Convo-I/O L9 attn bake (phase=9.5). Always registered; bake is a
        # no-op when enable_conversational_io is False. Fires regardless of
        # alu_mode; runs AFTER the L9 LEV bakes (phase 9.0/9.1) since the
        # legacy code also re-filled alibi_slopes after LEV setup.
        make_format_string_fetch_head_op(
            enable_conversational_io=enable_conversational_io
        ),
        make_layer10_carry_relay_op(),
        make_layer10_byte_passthrough_op(),
        make_layer10_sp_byte_passthrough_op(),
        make_layer10_psh_stack0_passthrough_op(),
        # 5 block-level bake ops (phases 10.0..10.4): the actual attn weight
        # bakes for L10 heads 0-4. Inline calls in set_vm_weights have been
        # removed; these own the bake. The five kind="attn" placeholders above
        # remain as dep-graph anchors. See docstrings on the individual bake
        # ops for details.
        make_layer10_carry_relay_bake_op(),
        make_layer10_byte_passthrough_bake_op(),
        make_layer10_sp_byte_passthrough_bake_op(),
        make_layer10_psh_stack0_passthrough_bake_op(),
        make_layer10_stack0_byte_relay_bake_op(),
        make_layer10_alu_op(),
        make_layer11_mul_partial_op(),
        make_layer12_mul_combine_op(),
        make_layer13_mem_addr_gather_op(),
        make_layer13_shifts_op(alu_mode=alu_mode),
        # 4-stage SHL/SHR composite (replaces ALUShift wrapper). Only
        # meaningful in efficient mode. The 5 ops returned by
        # ``make_alu_shift_composite_ops`` share a single
        # ``_ALUShiftCompositeBuilder`` so the kind="block" install op can
        # hand the fully-constructed composite to ``model.blocks[13].ffn``
        # regardless of which block the dep analyser placed the kind="ffn"
        # stage ops at.
        *(
            make_alu_shift_composite_ops()
            if alu_mode == "efficient"
            else []
        ),
        make_layer14_mem_generation_op(),
        make_layer14_temp_clear_op(),
        make_layer14_clear_addr_key_pollution_op(),
        make_layer14_clear_output_corruption_op(),
        make_layer14_clear_mem_marker_output_op(),
        make_layer15_memory_lookup_op(),
        make_layer15_nibble_copy_op(),
        make_layer16_lev_routing_op(),
        # Critical additional ops (M3+ continuation)
        make_binary_pop_sp_increment_op(),
        make_layer10_stack0_byte_relay_op(),
        make_layer9_marker_suppress_op(),
        # L8 ALU ADD/SUB flatten (5 ops replacing the monolithic ALUAddSub)
        make_l8_alu_addsub_bdtoge_op(),
        make_l8_alu_addsub_stage1_op(),
        make_l8_alu_addsub_stage2_op(),
        make_l8_alu_addsub_stage3_op(),
        make_l8_alu_addsub_getobd_op(),
        # L10 post_ops merged into a single phase-10.5 ffn
        make_l10_post_ops_combined(),
        # L15 attention resize: 8 heads -> 12 heads for LEV (phase=14.9 so it
        # fires before _set_layer15_memory_lookup populates the heads).
        make_l15_attention_resize_op(),
        # Model-level bake that runs BEFORE legacy_bake (phase 998) so its
        # FFN unit writes survive the rightsize pass at end of legacy_bake.
        make_function_call_weights_op(),
        # Model-level bake that runs BEFORE legacy_bake (phase 998) so its
        # L6 FFN unit writes survive the rightsize pass at end of legacy_bake.
        make_io_putchar_routing_op(),
        # Tool-call bakes (phase 998.8): flag-gated. Always registered so the
        # registration list is stable; bake_fn is a no-op when
        # `enable_tool_calling=False`. The 3 ops replace inline calls in
        # set_vm_weights at L5 FFN (opcode decode units 400-405), L6 attn
        # head 5 (relay), and L6 FFN unit 1300 (NEXT_TOOL_CALL detection).
        # The L6 attn alibi_slopes[5]=5.0 mutation is folded into the relay
        # op so the bake is self-contained.
        make_tool_call_opcode_decode_op(enable_tool_calling=enable_tool_calling),
        make_tool_call_relay_head_op(enable_tool_calling=enable_tool_calling),
        make_tool_call_detection_op(enable_tool_calling=enable_tool_calling),
        # Conversational-I/O L5/L6 bakes (3 block ops, all gated by
        # ``enable_conversational_io``). When the flag is False each bake_fn
        # is a no-op so the dep-graph layout stays stable. Phases 5.6 / 999.5 /
        # 6.6 place these alongside the existing L5/L6 FFN + L6 attn bakes.
        make_convo_io_opcode_decode_op(
            enable_conversational_io=enable_conversational_io
        ),
        make_convo_io_relay_heads_op(
            enable_conversational_io=enable_conversational_io
        ),
        make_convo_io_state_machine_op(
            enable_conversational_io=enable_conversational_io
        ),
        # Model-level bakes (run after legacy_bake's per-layer/head/embed work)
        make_head_bake_op(),
        make_embedding_bake_op(),
        # Initial-PC bake: writes the PC_OFFSET pattern into the REG_PC
        # token-embedding row (replaces the runtime `_inject_initial_pc`).
        # Phase=1001.5 so it runs just AFTER embedding_bake (1001).
        make_initial_pc_bake_op(),
        # L6 attn head 6 opcode relay (phase=1002): runs AFTER legacy_bake's
        # `attn6.alibi_slopes.fill_(0.0)` so the alibi_slopes[6]/[7]=5.0 writes
        # survive. See docstring on make_opcode_relay_head_op for details.
        make_opcode_relay_head_op(),
        # Model-level post-passes — run in phase order after legacy_bake (999):
        #   branch_override_patch (1100): suppress spurious branch/LEV-override
        #     units across all blocks.
        #   l6_dead_unit_zero    (1160): zero L6 FFN units misreading
        #     OUTPUT_BYTE residuals.
        #   l7_dead_unit_zero    (1170): suppress L7 FFN units firing at PC
        #     marker.
        #   right_size_ffns      (1200): trim each block's FFN hidden dim to
        #     actually-programmed unit count.
        #   expand_wrapper_blocks(1300): split HybridALU + post_op composites
        #     into separate transformer blocks.
        # All five carry migrated=True; their inline counterparts inside
        # set_vm_weights have been removed to avoid double-bake.
        make_branch_override_patch_op(),
        make_l6_dead_unit_zero_op(),
        make_l7_dead_unit_zero_op(),
        make_right_size_ffns_op(),
        make_expand_wrapper_blocks_op(),
        # Conversational-I/O tail bakes (gated by enable_conversational_io).
        # Registered unconditionally; bake_fn is a no-op when the flag is off
        # so the dep-graph topology is identical regardless of flag state.
        # See ``make_null_terminator_detection_op`` /
        # ``make_conversational_io_output_routing_op`` for details.
        make_null_terminator_detection_op(
            enable_conversational_io=enable_conversational_io,
            alu_mode=alu_mode,
        ),
        make_conversational_io_output_routing_op(
            enable_conversational_io=enable_conversational_io,
        ),
    ]


def all_alu_postop_attach_ops() -> list:
    """Return the L8-L13 ALU post-op attach ops.

    These are NOT included in all_core_ops() because they attach the ALU to
    ``block.post_ops`` AFTER set_vm_weights' legacy bake has populated the
    FFN weights. Use _dispatch_migrated_block_ops in vm_step.py to fire them
    at the right time (post-legacy-bake, pre-right-size).
    """
    return [
        make_l8_alu_postop_attach_op(),
        make_l9_alu_postop_attach_op(),
        make_l10_alu_postop_attach_op(),
        make_l11_alu_postop_attach_op(),
        make_l12_alu_postop_attach_op(),
        make_l13_alu_postop_attach_op(),
    ]
