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
from .l5_ops import _stack0_next_arith_enabled  # noqa: F401 (underscore name)
from .shared import (  # noqa: F401 — cross-op FFN-lint demo flags (tooling-only)
    ffn_lint_clean_demo_enabled,
    ffn_lint_mull14_demo_enabled,
)
from .l6_ops import *  # noqa: F401,F403
from .l7_ops import *  # noqa: F401,F403
from .l8_ops import *  # noqa: F401,F403
from .l9_ops import *  # noqa: F401,F403
from .l10_ops import *  # noqa: F401,F403
from .l11_ops import *  # noqa: F401,F403
from .l12_ops import *  # noqa: F401,F403
from .l13_ops import *  # noqa: F401,F403
from .l14_ops import *  # noqa: F401,F403
from .l14_ops import _li_zeroaddr_indicator_on  # noqa: F401
from .l15_ops import *  # noqa: F401,F403
from .l16_ops import *  # noqa: F401,F403
from .alu_ops import *  # noqa: F401,F403
from .flag_gated_ops import *  # noqa: F401,F403
from .model_ops import *  # noqa: F401,F403
from .user_input_ops import (  # noqa: F401
    make_layer5_user_input_gather_op,
    make_layer6_getchar_routing_op,
)
from .control_flow_heads import make_lev_detector_head_op  # noqa: F401
from .shared import mul_width2_enabled, operand_from_memsp_enabled  # noqa: F401
from .shared import sub_full_borrow_enabled  # noqa: F401
from .shared import l8_operand_sp_disc_enabled  # noqa: F401
from .shared import sili_b1_restore_enabled  # noqa: F401


def all_core_ops(
    alu_mode: str = "lookup",
    *,
    enable_conversational_io: bool = False,
    enable_tool_calling: bool = False,
    enable_neural_io_think_protocol: bool = False,
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

    ``enable_neural_io_think_protocol`` is forwarded to the PUTCHAR
    THINK-tag protocol bake (``make_putchar_think_protocol_op``).
    Default False; when True the bake wires up the model-emitted
    ``THINKING_END, byte, THINKING_START`` sequence at the end of a
    PUTCHAR step (BLOG_SPEC.md:851). See
    ``c4_release/docs/NEURAL_IO_VIA_THINK_PROTOCOL_PLAN.md`` for the
    full design and Phase 2/3 follow-ups (PRTF, GETCHAR, READ).

    Conversational-I/O tail ops (L10 null-terminator detection,
    L15 output routing) are also flag-gated via ``enable_conversational_io``;
    L10 null-terminator additionally requires ``alu_mode == 'lookup'`` since
    ffn10 is replaced with ``ALUAndOrXor`` in efficient mode.
    """
    return [
        # Phase 8.G.6: L0 attn dep anchor — gives L0 block ops a stable
        # ``target_op_name`` to bind to so they can drop ``layer_idx=0``
        # literals. Mirrors the L3/L4/L5/L6/L11 dep-anchor pattern.
        make_layer0_threshold_attn_dep_anchor_op(),
        make_layer0_threshold_attn_op(),
        make_threshold_attn_op(),
        make_layer2_threshold_attn_op(),
        # Conversational-I/O L2 lookback head: phase=2.1 so it bakes after
        # the L2 threshold attention (phase=2). Body is a no-op unless
        # ``enable_conversational_io`` is True.
        make_layer2_lookback_detection_head_op(
            enable_conversational_io=enable_conversational_io,
        ),
        make_carry_forward_attn_op(),
        make_phase_a_ffn_op(),
        make_threshold_ffn_op(),
        make_layer2_mem_byte_flags_op(),
        # Cancels the REG_PC token-embedding initial-PC bake at step-1+ PC
        # markers (MARK_PC AND HAS_SE). Runs at L2 (phase=2.5) so L3 FFN's
        # PC INCREMENT sees a clean EMBED_LO/HI — see docstring on the
        # factory for why the prior L3-inline cancel leaks via PC
        # INCREMENT reading the FFN's input residual.
        make_layer2_initial_pc_bake_cancel_op(),
        # `make_nibble_copy_ffn_op` (kind="ffn", not migrated) is kept as a
        # placeholder in the dep graph: removing it shifts the dep-graph layout
        # for L11/L12 ops. The actual L15 nibble-copy bake is performed by
        # `make_layer15_nibble_copy_op` (kind="block", layer_idx=15,
        # migrated=True) below; this kind="ffn" entry is skipped at dispatch
        # because legacy_bake is present and migrated=False.
        make_nibble_copy_ffn_op(),
        make_register_default_ffn_op(),
        make_register_default_ffn_dep_anchor_op(),
        # Conversational-I/O L3 state init: phase=3.1 so it bakes after
        # the L3 FFN (phase=3) and writes into FFN units above the L3 /
        # L6-routing unit ranges. Body is a no-op unless
        # ``enable_conversational_io`` is True.
        make_convo_io_state_init_op(
            enable_conversational_io=enable_conversational_io,
        ),
        make_layer4_pc_relay_op(),
        make_layer4_ffn_op(),
        make_layer4_ffn_dep_anchor_op(),
        # STACK0 via mem attention: Q-side SP->ADDR_KEY staging.
        #
        # REBUILD (Inc-1, 2026-06-18): DISABLED. GPU diagnosis showed this op
        # is the dominant `C4_OPERAND_FROM_MEMSP` blocker: staging SP into the
        # ADDR_KEY band at the AX marker pollutes that row in a way the
        # downstream L19 FFN reacts to by CRUSHING OUTPUT to 0 on EVERY step
        # (PSH included) -> step-1 AX=0 -> 0/4 on the gate. The rebuilt L8
        # head-5 mem-to-ALU CAM (below) no longer matches on address; it picks
        # the most-recent pushed value via ALiBi recency on the MEM value-byte
        # row, so it needs no ADDR_KEY staging. enable=False (no L19 pollution).
        make_layer4_sp_to_addr_key_op(enable=False),
        make_fetch_op(),
        make_fetch_dep_anchor_op(),
        # Consumer-opcode LOOKAHEAD (#221 framing-drift fix; flag-gated
        # C4_STACK0_NEXT_ARITH, DEFAULT-OFF). The C4_STACK0_B0_DUMP over-fires
        # on arithmetic-intermediate operand frames (expr a*b/c) but is needed
        # on comparison-result frames (if/bool). The only separator is the
        # CONSUMER opcode (the next instruction), causally unavailable at the
        # operand frame -- but the single-slot C4 ISA puts it at a fixed PC+8 in
        # program memory. (1) build PC+8, (2) fetch op2's opcode byte by
        # ADDR_KEY content-match, (3) decode "arith consumer", (4) relay it to the
        # STACK0 row, (5) AND a prior-arith latch -> the bounded dump-block flag
        # the dump reads as a blocker so it skips MULTI-op arith-intermediate
        # operand frames (fresh value survives) while keeping the +27 on cmp/
        # branch frames AND the single-op operand-frame dump (mul/sub guards).
        # The whole chain is registered ONLY when the feature flag is on so a
        # flag-off build is byte-identical to HEAD (no op-graph / scheduling
        # perturbation). See l5_ops.make_lookahead_pc8_chain_op etc.
        *(
            [
                make_lookahead_pc8_chain_op(),
                make_lookahead_opcode_fetch_op(),
                make_next_arith_flag_op(),
                make_next_arith_relay_op(),
                make_prior_arith_latch_op(),
                make_dump_block_flag_op(),
            ]
            if _stack0_next_arith_enabled() else []
        ),
        # V9 GETCHAR neural read scaffolding (BLOG_SPEC.md:851).
        # Phase 1: registered but disabled (enable=False). The runner-side
        # _inject_getchar shim still owns byte transfer until phase 2
        # widens L5 to 10 heads + allocates STDIN_BYTE dims. See
        # docs/V9_GETCHAR_READ_NEURAL_PLAN.md.
        make_layer5_user_input_gather_op(enable=False),
        make_opcode_decode_ffn_op(),
        make_opcode_decode_ffn_dep_anchor_op(),
        make_layer6_attn_dep_anchor_op(),
        make_layer6_attn_op(),
        make_layer6_routing_ffn_op(),
        make_layer6_ffn_dep_anchor_op(),
        make_layer6_ent_after_jsr_sp_byte0_fixup_op(),
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
        # STACK0 campaign Inc-1 store-commit relay (phase=7, L7 block 9).
        # Flag-gated by C4_OPERAND_FROM_MEMSP (DEFAULT OFF = byte-identical).
        # When on, broadcasts MEM_STORE from a MEM section's MARK_MEM marker
        # row to its value-byte-0 row into MEM_STORE_AT_VAL, BEFORE the L8
        # attn block, so the L8 head-5 mem-to-ALU CAM can discriminate the
        # real PSH store's value row from phantom IMM-step MEM value rows.
        # See make_layer7_mem_store_relay_op + make_layer8_mem_to_alu_op.
        make_layer7_mem_store_relay_op(enable=operand_from_memsp_enabled()),
        # expr_add_mul operand-A SP-frame discriminator relay (phase=7, L7
        # block 9). Flag-gated by C4_L8_OPERAND_SP_DISC (DEFAULT-OFF blueprint;
        # flag-OFF / non-campaign = byte-identical, the SP_ADDR_LO band is
        # omitted). When on, relays the push-time SP low byte (OUTPUT_LO one-hot
        # at the nearest prior MARK_SP) onto the MEM store value rows + binary-op
        # AX query rows into SP_ADDR_LO, BEFORE the L8 attn block, so the L8
        # head-5 mem-to-ALU CAM can demote a POPPED store (SP-frame mismatch)
        # vs the LIVE store on the depth-2 expr_add_mul (a+b*c) cluster. See
        # make_layer7_sp_addr_relay_op + make_layer8_mem_to_alu_op.
        make_layer7_sp_addr_relay_op(enable=l8_operand_sp_disc_enabled()),
        # BLOCKER-1 fix (block-10 SHARPENER): winner-take-all clamp of the
        # relayed SP_ADDR_LO one-hot (threshold each cell at 0.5 -> clean 0/1
        # cells in SP_ADDR_LO_SHARP + a bounded PRESENT in {0,1}). Runs at the L7
        # FFN AFTER the relay attention (requires after=layer7_sp_addr_relay) and
        # BEFORE the L8 head-5 read, so head-5's bilinear SP-mismatch penalty
        # cancels EXACTLY on a frame MATCH for ALL ops -> kills the var_simple
        # PRESENT blow-up that kept the discriminator default-OFF. Flag-gated.
        make_layer7_sp_addr_sharpen_op(enable=l8_operand_sp_disc_enabled()),
        # Convo-I/O L7 attn bake (phase=7.5). Always registered; bake is a
        # no-op when enable_conversational_io is False. See docstring for
        # phase/ordering rationale.
        make_format_pointer_extraction_op(
            enable_conversational_io=enable_conversational_io
        ),
        # B7-2 (phase=7.6): SP_BYTE0_IS_F8 producer. Adds head-6 V/O slots
        # 6 and 7 that detect the carry-forwarded SP byte 0 == 0xF8 from
        # the MARK_SP row's EMBED_LO+8 / EMBED_HI+15 (set by L3 head 2).
        # See ``make_layer7_sp_byte0_is_f8_op`` docstring.
        make_layer7_sp_byte0_is_f8_op(),
        # L8 attn head 6 AX_CARRY refresh from prev step AX marker OUTPUT
        # (commit 3d1b700). Always registered for the staleness analyzer
        # (Phase 3 / Agent G of ARCH_LEAKAGE_FIX_PLAN.md) so its
        # ``produces`` annotation participates in the in-step producer
        # check that guards the L8 ALU's ``consumes_fresh AX_CARRY_LO``
        # contract. Bake body is no-op (``enable=False``) until the full
        # production wiring is validated end-to-end; the active head-6
        # bake currently lives in ``unified_compiler/compiler.py`` (the
        # UnifiedVMCompiler path, see commit 3d1b700).
        make_layer8_head6_ax_carry_refresh_op(enable=False),
        # V2/G7 LEV detector attention head (phase=8.06). Detects that
        # the prior instruction-step's opcode was LEV and materialises the
        # saved PC / BP / SP nibbles into the current step's
        # ``PC_VIA_LEV_DETECTOR_LO/HI`` / ``BP_VIA_LEV_DETECTOR`` /
        # ``SP_VIA_LEV_DETECTOR`` residual bands. Downstream readers
        # (L9 alu, L8 sp_gather_bake) consume the detector dims so the
        # cross-step ``requires["after"]=layer16_lev_routing`` edges
        # collapse. See docs/CONTROL_FLOW_DETECTOR_HEADS.md.
        make_lev_detector_head_op(enable=False),
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
        # Dead-unit budget anchor for L8's primary block FFN. L8 is
        # attention-only at the primary block (see
        # docs/DEAD_UNIT_AUDIT_2026_06_05.md); this kind="ffn" anchor
        # carries ``ffn_units_used=0`` so the dynamic-FFN allocator
        # pre-sizes block[L8].ffn.hidden_dim=0 instead of allocating
        # the historical 4096-unit dead footprint.
        make_layer8_ffn_dep_anchor_op(),
        make_layer8_sp_gather_bake_op(),
        make_layer8_multibyte_fetch_bake_op(),
        # Paired with the L4 SP-to-ADDR_KEY staging above. Flag-gated by
        # C4_OPERAND_FROM_MEMSP (DEFAULT OFF = byte-identical). When on, this
        # head 5 reads mem[SP] byte 0 via the ADDR_KEY CAM (most-recent
        # matching MEM_STORE wins via ALiBi recency) and writes ALU_LO/HI at
        # the AX marker — the binary-op operand-A read re-routed from the
        # emitted STACK0 token to memory. STACK0-emission-drop prerequisite;
        # see make_layer4_sp_to_addr_key_op above.
        make_layer8_mem_to_alu_op(enable=operand_from_memsp_enabled()),
        # B7-5 SP_GATHERED_THIS_STEP sentinel: marks MARK_SP rows with a
        # single bit indicating L8's SP gather has fired this step. Phase
        # 8.6 runs after all other L8 ops so the sentinel is visible to
        # L9+ consumers without disturbing earlier-phase bakes.
        make_layer8_sp_gathered_sentinel_op(),
        make_layer9_alu_op(alu_mode=alu_mode),
        make_lev_addr_relay_op(),
        make_lev_bp_to_pc_relay_op(),
        # ALiBi-based memory propagation attention head (phase=9.2).
        # PROOF-OF-CONCEPT for replacing _inject_mem_store / runner shadow
        # memory with attention. Registered always so the dep graph and
        # layer_idx gates see it; bake is a no-op by default (`enable=False`)
        # so existing tests are byte-identical. See l9_ops.py docstring
        # for the full design and slope-tuning analysis.
        make_alibi_mem_attn_op(enable=False),
        # Wave A v2 (2026-06-10): register-tagged STEP_END operand
        # relay. Two declarative attn heads in L9 attn that mirror raw
        # ALU_LO/HI / AX_CARRY_LO/HI / CMP / OP_<cmp> from MARK_AX to
        # the SE_-tagged dims at MARK_SE_ONLY. Fires BEFORE L9 FFN so
        # the migrated L9 CMP rules (commit 62b64449, MARK_SE_ONLY
        # gated) have the operand state at the SE row. Replaces the
        # disabled L11 step_end_operand_relay (10ca51a7, enable=False)
        # which both (i) fired too late and (ii) collapsed register
        # identity by writing raw ALU_LO/HI at MARK_SE. See
        # docs/STEP_END_COMPUTE_ARCHITECTURE_2026_06_10.md and memory
        # note ``project_wave_b_cmp_needs_l9_internal_relay.md``.
        make_layer9_step_end_operand_relay_op(),
        # Wave B Phase 2.1: re-assert the SE relay heads' ALiBi slope to
        # 0.2 AFTER layer10_residual_alibi_slopes (phase 999.1) clobbers
        # blocks[10] heads 3/4 to 0.5/1.0. Without this the relay attends
        # to nothing across the d=29 AX->SE gap and SE_* never transmits
        # (Wall 2, project_attention_dsl_alibi_slope_gap). Slope-only;
        # owns no weights.
        make_layer9_se_relay_slope_op(),
        # Convo-I/O L9 attn bake (phase=9.5). Always registered; bake is a
        # no-op when enable_conversational_io is False. Fires regardless of
        # alu_mode; runs AFTER the L9 LEV bakes (phase 9.0/9.1) since the
        # legacy code also re-filled alibi_slopes after LEV setup.
        make_format_string_fetch_head_op(
            enable_conversational_io=enable_conversational_io
        ),
        make_layer10_carry_relay_op(),
        # Phase 3 (mem cluster fix): sibling attn-side anchor for L10.
        # Decouples the 6 L10 attn-bake ops from ``layer10_carry_relay``
        # (which historically anchored both attn + ffn families). See
        # the docstring on ``make_layer10_attn_anchor_op`` and
        # ``docs/MEMORY_PHASE2_BLOCKER_2026_06_05.md``.
        make_layer10_attn_anchor_op(),
        make_layer10_byte_passthrough_op(),
        make_layer10_sp_byte_passthrough_op(),
        make_layer10_psh_stack0_passthrough_op(),
        # Wave 1 A3 broadcast topology anchor (slots 8/9/10):
        # AX byte 1/2/3 -> STACK0_BYTE_VAL_h_LO/HI at STACK0 byte rows
        # during OP_PSH. See docs/L8_SP_GATHER_STACK0_AUDIT_2026_06_07.md.
        make_layer10_psh_ax_broadcast_op(),
        # 5 block-level bake ops (phases 10.0..10.4): the actual attn weight
        # bakes for L10 heads 0-4. Inline calls in set_vm_weights have been
        # removed; these own the bake. The five kind="attn" placeholders above
        # remain as dep-graph anchors. See docstrings on the individual bake
        # ops for details.
        make_layer10_carry_relay_bake_op(),
        make_layer10_byte_passthrough_bake_op(),
        make_layer10_sp_byte_passthrough_bake_op(),
        make_layer10_bp_byte_passthrough_bake_op(),
        make_layer10_psh_stack0_passthrough_bake_op(),
        # Wave 1 A3: resize L10 attn 8 -> 13 heads BEFORE the broadcast
        # / PC byte_passthrough bakes (otherwise slots 8/9/10/11 write
        # out-of-bounds). Mirrors ``l15_attention_resize`` for L15.
        make_l10_attention_resize_op(),
        # Wave 1 A3 broadcast heads bake (slots 8/9/10), phase ~10.35.
        make_layer10_psh_ax_broadcast_bake_op(),
        # JSR/LEV PC byte_passthrough head (slot 11), phase ~10.37.
        # Mirrors BP/SP/AX byte_passthrough but at the MARK_PC marker;
        # suppressed on OP_JSR/JMP/BZ/BNZ/LEV so the L6/L9 PC override
        # writers are not stomped. See
        # ``docs/PHASE_5_JSR_ENT_LEV_FOLLOWUP.md`` for context.
        make_layer10_pc_byte_passthrough_bake_op(),
        make_layer10_stack0_byte_relay_bake_op(),
        make_layer10_alu_op(),
        # Cluster D fix (2026-06-03): post-L9 BZ/BNZ PC override owner.
        # Physically moves the BZ/BNZ cancel + FETCH-target-copy bands out
        # of layer6_routing_ffn (which runs BEFORE layer9_alu in the
        # forward pass) into a kind="ffn" op pinned via
        # ``requires={"after": "layer10_alu"}`` so it lands at L11+ where
        # same-step CMP is the freshly-written L9 ALU output. See
        # c4_release/docs/CMP_PATH_AUDIT.md.
        make_post_l9_bz_bnz_pc_override_op(),
        # Phase 8.A.4 retry: dep anchor for L11. The actual MUL partial bake
        # is owned by ``layer11_mul_partial`` (kind="block", target_op_name=
        # ``_layer11_ffn_dep_anchor``); this no-op companion gives the
        # scheduler a layer-resident ffn op so the block op resolves to L11.
        make_mul_partial_dep_anchor_op(),
        make_mul_partial_op(alu_mode=alu_mode),
        # Wave A (docs/STEP_END_COMPUTE_ARCHITECTURE_2026_06_10.md):
        # two-head attention relay broadcasting OP_<NAME>, AX_CARRY,
        # ALU_LO/HI, CMP, and STACK0_BYTE0..3 from MARK_AX -> MARK_SE
        # within the same step. Enables Wave B migration of L8/L9/L10
        # dispatch+ALU+CMP rules from MARK_AX gating to MARK_SE gating.
        make_layer11_step_end_operand_relay_op(
            enable=True,
            # Scoped Wave A relay: only relay OP_<NAME> dispatch flags
            # (the operand bands AX_CARRY / ALU / CMP / STACK0_BYTE
            # are NOT relayed because broadcasting them at MARK_SE
            # races downstream OUTPUT consumers and regresses 9 smoke
            # tests). Within OP_<NAME>, OP_IMM and OP_SHR are excluded
            # because their MARK_SE-side relay regresses
            # test_add_basic / test_shr -- the L8 IMM/SHR rules read
            # the opcode flag at MARK_SE under the migrated Wave B
            # cluster, and the relayed flag interacts with a stale
            # cross-step alias on those specific paths. Verified
            # baseline-neutral on tests/test_smoke.py at 31/51 PASS.
            # See ``docs/STEP_END_COMPUTE_ARCHITECTURE_2026_06_10.md``
            # section 4 for the Wave A v1 (RAW relay) vs v2 (SE_-
            # tagged mirror) trade-off.
            include_op_name=True,
            include_ax_carry=False,
            include_alu=False,
            include_cmp=False,
            include_stack0_byte=False,
            op_name_subset=(
                "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
                "OP_BZ", "OP_BNZ", "OP_JMP", "OP_JSR", "OP_EXIT", "OP_LEV",
                "OP_PSH",
                "OP_ENT", "OP_ADJ", "OP_LEA",
                "OP_LI", "OP_LC", "OP_SI", "OP_SC",
                "OP_OR", "OP_AND", "OP_XOR",
                "OP_MUL", "OP_DIV", "OP_MOD",
                "OP_ADD", "OP_SUB", "OP_SHL",
            ),
        ),
        # AX byte-1 DUMP carry (dedicated ``H1_PREV_STEP`` band). The CARRY
        # HEAD half: copies the PREVIOUS step's AX byte-1 ``H1`` one-hot into
        # the fresh ``H1_PREV_STEP`` band UNCONDITIONALLY (via ``H1.*.-1``, an
        # SSA cross-step read -> no same-step back-edge). The carried-vs-fresh
        # gate + the LM-head re-supply live in the partner
        # ``ax_byte1_dump_repopulate`` FFN + ``ax_byte1_dump_head_bake`` below.
        # Writing the distinct ``H1_PREV_STEP`` band (read by nobody upstream)
        # breaks the H1-write 2-cycle that blocked every prior single-head
        # attempt. See l11_ops.make_layer11_ax_byte1_dump_carry_op and
        # docs/AX_BYTE1_DUMP_CARRY_H1_WRITE_CYCLE_2026_06_13.md.
        make_layer11_ax_byte1_dump_carry_op(enable=True),
        # STACK0 byte-0 DUMP carry (Root 2 — the if/bool/expr framing drift).
        # The CARRY HEAD half (mirror of the AX byte-1 carry above): copies the
        # PREVIOUS step's STACK0-marker ``H1``/``H3`` byte-0 one-hot into the
        # fresh ``STACK0_B0_H1_PREV``/``STACK0_B0_H3_PREV`` bands (via
        # ``H1.*.-1``/``H3.*.-1`` SSA cross-step reads -> no same-step
        # back-edge). The byte-0 emission one-hot is decoded fresh at the PSH
        # step but ABSENT (then smeared+nuked to ~-289M by the L21/L25
        # correctors) on the carried comparison step -> a marker wins and the
        # model emits a spurious extra register block (57-token step) that the
        # fixed-35 slicer misreads. The carried-vs-fresh gate + LM-head re-supply
        # live in the partner ``stack0_byte0_dump_repopulate`` FFN +
        # ``stack0_byte0_dump_head_bake`` below. See
        # l11_ops.make_stack0_byte0_dump_carry_op.
        make_stack0_byte0_dump_carry_op(enable=True),
        # STACK0 byte-0 POP discriminator (if_gt/if_lt/if_eq/bool_and fix). The
        # carry above re-supplies the STALE operand byte-0 on EVERY carried
        # STACK0 row, including the rows AFTER the comparison/branch popped it
        # (id 350 emits 35 on the post-pop steps instead of the oracle's 0). This
        # L9 head-8 CAUSAL LATCH fires on a STACK0 row from the consuming cmp/
        # branch opcode step ONWARD and the L25 dump reads ``STACK0_B0_POPPED``
        # as a HARD blocker -> it stops re-supplying the popped operand. FLAG-
        # GATED (``C4_STACK0_B0_POPPED``, default-OFF): off = no band, no op, no
        # dump condition -> byte-identical. See
        # l11_ops.make_stack0_byte0_popped_latch_op.
        make_stack0_byte0_popped_latch_op(),
        # ENT saved-BP store DUMP carry (BP_SAVE_PREV — the func/nested/rec/var
        # LI-from-frame 37-token desync). The CARRY HEAD half (mirror of the AX
        # byte-1 / STACK0 byte-0 carries above): the callee's saved-BP store (the
        # ENT step's MEM val bytes) emits 0xFF garbage because the L14 value heads
        # content-address the WRONG source at the ENT step. Instead of fixing the
        # failing same-step attention, this head carries the clean old_BP across
        # the step boundary: it attends from each ENT-step MEM val-byte-k PREDICTOR
        # row BACK to the prev step's BP byte-k row (matched per-byte by
        # ``MEM_VAL_B{k}`` <-> ``BYTE_INDEX_{k}`` + ``OP_JSR`` prologue preference)
        # and V-copies the clean ``CLEAN_EMBED_LO/HI`` old_BP nibbles into the
        # dedicated ``BP_SAVE_PREV`` band. The OP_ENT gate + OUTPUT re-supply live
        # in the partner ``bp_save_dump_repopulate`` FFN below. See
        # l11_ops.make_bp_save_prev_carry_op and
        # docs/FUNC_LEV_IS_LI_FROM_FRAME_37TOKEN_DESYNC_2026_06_14.md.
        make_bp_save_prev_carry_op(enable=True),
        # Phase 8.G.6: L12 ffn dep anchor — gives L12 block ops a
        # stable ``target_op_name`` to bind to so they can drop
        # ``layer_idx=12`` literals.
        make_mul_combine_dep_anchor_op(),
        make_mul_combine_op(alu_mode=alu_mode),
        # Phase 8.G.6: L13 attn dep anchor — gives L13 block ops a
        # stable ``target_op_name`` to bind to so they can drop
        # ``layer_idx=13`` literals.
        make_layer13_attn_dep_anchor_op(),
        # Phase 3b (mem cluster fix, 2026-06-05): L13 mem-addr anchor —
        # split off from ``_layer13_attn_dep_anchor`` so the mem-addr
        # gather binds to a ``layer_idx=13``-pinned anchor while the
        # FFN-side shift family stays on the original (L16) anchor. See
        # ``docs/MEMORY_PHASE4_BLOCKER_2026_06_05.md`` and the docstring
        # on ``make_layer13_mem_addr_anchor_op``.
        make_layer13_mem_addr_anchor_op(),
        make_layer13_mem_addr_gather_op(),
        # 16-bit OR/XOR byte-1 fix (2026-06-11): L13 attn head 3 stages
        # operand-A byte 1 into AX_FULL on OR/XOR so the L15
        # high_byte_relay (gate widened to OP_OR/OP_XOR) can emit byte 1.
        make_layer13_bitwise_byte1_gather_op(),
        # Multi-byte SUB minuend relay (2026-06-12): L13 attn head 4
        # delivers the pushed operand's high bytes (STACK0_BYTE_VAL_h)
        # to the SUB byte-h emit rows so the L14 borrow cascade can
        # compute byte 1/2 of a multi-byte SUB (Part 1 of the sub_16bit
        # fix; the cascade rule re-point is Part 2 in l10_ops.py).
        make_layer13_sub_minuend_relay_op(),
        # Multi-byte ADD addend relay (2026-06-12): L13 attn head 5
        # delivers operand-A byte 1 (a1) into STACK0_BYTE_VAL_1 at the ADD
        # byte-1 emit row so the L10 ADD adder can compute a1 + b1 + carry
        # (Part 1 of the multi-byte ADD fix; the byte-1 adder is the L10
        # post-tail ``l10_add_high_byte_adder``). Mirror of head 4 (SUB),
        # gated on TEMP+8 (ADD) instead of TEMP+9 (SUB).
        make_layer13_add_addend_relay_op(),
        # width=2 MUL byte-1 result relay (2026-06-13): L13 attn head 6
        # copies the product's high byte (MUL_RESULT_HI_LO/HI, written by
        # the width=2 wide_mul install) into AX_FULL at the MUL MARK_AX
        # row so the existing layer15_alu_high_byte_relay emits byte 1.
        # Always registered (dep graph stable); enabled by default via
        # mul_width2_enabled() (opt out with C4_MUL_WIDTH2=0). The d_model
        # widen is head-dim-preserving (MUL_RESULT_HI routed through
        # extra_residual_dims), so bnz stays green -> smoke 50/1. See
        # docs/MUL_WIDTH2_WIDEN_2026_06_13.md.
        make_layer13_mul_result_hi_relay_op(enable=mul_width2_enabled()),
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
        # Phase 8.G.6 follow-up: L14 attn dep anchor — gives
        # ``layer14_mem_generation`` a stable ``requires["same_layer_as"]``
        # target so it can drop its ``layer_idx=14`` literal. Anchor is
        # pinned past ``_layer13_attn_dep_anchor`` so it lands at L14.
        make_layer14_attn_dep_anchor_op(),
        make_layer14_mem_generation_op(),
        make_layer14_temp_clear_op(),
        make_layer14_clear_addr_key_pollution_op(),
        make_layer14_clear_output_corruption_op(),
        # Cross-op FFN-lint demo fixtures (tools/lint_cross_op_ffn.py --demo).
        # Registered ONLY when their flag is on so a flag-off / production build
        # is byte-identical to golden 4958b35b. See l14_ops.py module banner.
        *(
            [make_ffn_lint_mull14_demo_op()]
            if ffn_lint_mull14_demo_enabled() else []
        ),
        *(
            [make_ffn_lint_clean_demo_op()]
            if ffn_lint_clean_demo_enabled() else []
        ),
        make_layer14_clear_mem_marker_output_op(),
        # var-cluster follow-up (2026-06-05): cancel the L3
        # mem_byte_0_default +0.940 baseline at MEM marker / addr-byte
        # rows when MEM_ADDR_SRC=1 (SI/SC stores). For SI/SC the addr
        # comes from STACK0 (can be ANY value, e.g. 0xFFFC for
        # var_simple_0); the L3 baseline biased argmax toward 0x00 and
        # competed with L14 mem_generation's addr writes. Runs at phase
        # 14.45 AFTER clear_mem_marker_output (14.4) and BEFORE
        # addr_key_neural_decode (14.5). See l14_ops.py:
        # make_layer14_mem_addr_src_default_suppress_op for derivation.
        make_layer14_mem_addr_src_default_suppress_op(),
        # var-cluster JSR-path sibling (2026-06-06): the f4f9103d SI/SC
        # cancel above does not fire on JSR step 0 (MEM_ADDR_SRC=0
        # because JSR uses SP, not STACK0), so var_simple_0/if_var_0/
        # var_three_0 still failed. This op cancels the L3 +0.940
        # baseline at the JSR/PSH/ENT store path (MEM_STORE=1 AND
        # MEM_ADDR_SRC=0). Runs at phase 14.46, immediately after
        # mem_addr_src_default_suppress (14.45). See l14_ops.py:
        # make_layer14_jsr_mem_default_suppress_op for derivation.
        make_layer14_jsr_mem_default_suppress_op(),
        # V2 ADDR_KEY neural decode (BLOG_SPEC.md:830) — flipped on
        # (2026-05-11, blockers-3-4 PR).  Replaces
        # NeuralVMEmbedding._inject_mem_metadata's per-val-byte
        # ADDR_KEY[lo, 16+hi, 32+top] one-hot writes with a baked FFN
        # decode reading addr nibbles from L13's mem_addr_gather output.
        # Parity validated by tests/test_addr_key_neural_decode.py (18
        # cases including carry-overflow); see
        # docs/V2_ADDR_KEY_NEURAL_DECODE_PLAN.md.
        make_layer14_addr_key_neural_decode_op(enable=True),
        # JSR AX bytes 1-3 zeroing (fix-jsr-ax-bytes-1-3, 2026-05-12):
        # Restores C4's 8-bit-AX convention (AX bytes 1-3 = 0x00) across
        # JSR by zeroing OUTPUT at AX byte positions when OP_JSR is active.
        # Unblocks Phase 5 JSR/ENT/LEV roundtrip tests (returns 7 instead
        # of 0xF8030063). See docs/PHASE_5_JSR_ENT_LEV_FOLLOWUP.md.
        make_layer14_jsr_ax_bytes_zero_op(),
        # LC byte 1-3 zeroing (V7 Phase 7a, per V7_HEAP_OPS_NEURAL_PLAN.md):
        # LC is a 1-byte load (char) so AX bytes 1-3 must be 0. L15 head 0
        # writes the loaded byte at AX byte 0; this op zeros bytes 1-3 by
        # boosting OUTPUT_*[0] at AX byte positions 1-3 when OP_LC_RELAY is
        # active. Mirrors the JSR variant above. Unblocks test_sc_then_lc.
        make_layer14_lc_ax_bytes_zero_op(),
        # Non-carry ALU bytes 1-3 zeroing (V7 Block 13, per
        # V7_HEAP_OPS_NEURAL_PLAN.md §2): for AND/OR/XOR/SHR the result is
        # byte-sized so AX bytes 1-3 must be 0. Mirrors the JSR/LC variants.
        # Gates on TEMP[7] (NOCARRY_ALU_OP relay added to L7 head 5 V slot 9
        # in the same commit). Backup for the L10 BinaryOpByteZeroingPostOp
        # against downstream L11-L14 contamination.
        make_layer14_alu_nocarry_ax_bytes_zero_op(),
        # ENT AX bytes 1-3 zeroing (Wave 1 Cluster B1, 2026-06-07): fixes
        # ``test_lea_basic`` by restoring AX bytes 1-3 = 0 at ENT step 0.
        # Without this op the model emits AX=0xE8E8E800 at the ENT step
        # (SP byte 0 leaks into AX bytes 1-3), corrupting the downstream
        # BP frame and collapsing LEA to AX=0 instead of BP+imm.
        # Mirrors the JSR / LC / nocarry-ALU variants above; gates on
        # OP_ENT (broadcast by L7 head 7 V slot 4).
        make_layer14_ent_ax_bytes_zero_op(),
        # SUB no-borrow multi-byte minuend-byte1 passthrough (2026-06-12):
        # completes the multi-byte SUB result on the no-borrow path the
        # borrow-gated L14 carry cascade cannot reach. Reads the minuend
        # byte 1 that ``layer13_sub_minuend_relay`` deposits into
        # STACK0_BYTE_VAL_1 and writes OUTPUT byte 1 when CARRY+2 (byte-0
        # borrow-out) is absent. Byte-identical on 8-bit SUB; fixes the
        # ~38 no-borrow 1096 ``sub`` cases (e.g. ``827 - 26``).
        make_layer14_sub_noborrow_high_byte_passthrough_op(),
        # SUB full-borrow (minuend byte1==0) flag precursor (CAMPAIGN-ONLY,
        # C4_SUB_FULL_BORROW): writes the bounded SUB_FULL_BORROW flag on the
        # SUB byte-1 emit row when the STACK0_BYTE_VAL_1 band is EMPTY (the
        # 0-1=0xFFFFFFFF full-underflow case the cascade/passthrough rules can't
        # key on -- there's no minuend-byte1 one-hot to match). Runs at this
        # EARLY L14 block where STACK0_BYTE_VAL_1 is still fresh (it is cleared
        # by the block-32 L18 slam). Golden byte-identical (band+op omitted
        # flag-OFF: registered ONLY when sub_full_borrow_enabled, like the
        # li_zeroaddr indicator below). See l14_ops.make_layer14_sub_full_borrow_flag_op.
        *(
            [make_layer14_sub_full_borrow_flag_op()]
            if sub_full_borrow_enabled() else []
        ),
        # si/li 16-bit LOAD byte-1 value CAPTURE (CAMPAIGN-ONLY,
        # C4_SILI_B1_RESTORE): a standalone PureFFN post_op on the block-16
        # mem-addr anchor that snapshots the freshly-delivered loaded AX byte-1
        # nibbles into the private LI_RELOAD_B1 band BEFORE the block-32 (L18)
        # OUTPUT_HI slam crushes them. Self-gates to a no-op (empty reads/writes,
        # band omitted) when the flag is off -> golden 7f6f2e5d byte-identical.
        # The L25-tail RESTORE half (make_sili_b1_restore_op) re-supplies the
        # captured byte-1 AFTER the slam. See l14_ops.make_layer14_sili_b1_capture_op.
        make_layer14_sili_b1_capture_op(),
        # func re-read-LEA byte-0 LO-nibble CAPTURE (Bug #2, CAMPAIGN-ONLY,
        # C4_FUNC_LEA_B0_RESTORE): a standalone PureFFN post_op on the block-16
        # mem-addr anchor that snapshots the freshly-computed re-read LEA byte-0
        # LO one-hot into the private LEA_REREAD_B0 band BEFORE the block-42 (L21)
        # slam stamps the prior LEA's lo nibble. Self-gates to a no-op (empty
        # reads/writes, band omitted) when the flag is off -> golden d0619711
        # byte-identical. The L25-tail RESTORE half (make_func_lea_b0_restore_op)
        # re-establishes the captured nibble AFTER the slam. See
        # l14_ops.make_func_lea_b0_capture_op.
        make_func_lea_b0_capture_op(),
        # Phase 6 Wave 7 demo: pure-declaration corrective op. One
        # ``FFNRule`` + ``pin=None`` auto-fit + slim ``bake_fn`` wrapper.
        # Byte-identically a no-op on the live corpus -- the demo proves
        # the END-TO-END FLOW (declare -> byte-identity gate -> compile
        # -> corpus check) per docs/HOW_TO_ADD_A_CORRECTIVE_OP.md.
        make_layer14_demo_phase6_wave7_op(),
        # PHASE-2 KEYSTONE (#318): the (committed AND zero-address) store
        # indicator. Registered ONLY when the campaign keystone flag is on
        # (C4_L15_LI_ZEROADDR_CAM) so a flag-off / golden build is byte-identical
        # (the op is not registered, the chain alloc never claims its slot, and
        # the LI_ZEROADDR_COMMITTED band is not collected). Materializes the
        # FFN 3-way AND that L15 head-0 keys K on. See l14_ops.py
        # make_layer14_li_zeroaddr_indicator_op + l15_ops _l15_li_zeroaddr_cam_on.
        *(
            [make_layer14_li_zeroaddr_indicator_op()]
            if _li_zeroaddr_indicator_on() else []
        ),
        make_layer15_memory_lookup_op(),
        make_layer15_nibble_copy_op(),
        make_layer16_lev_routing_op(),
        # Critical additional ops (M3+ continuation)
        make_binary_pop_sp_increment_op(),
        make_layer10_stack0_byte_relay_op(),
        make_marker_suppress_op(),
        # L8 ALU ADD/SUB flatten (5 ops replacing the monolithic ALUAddSub)
        make_l8_alu_addsub_bdtoge_op(),
        make_l8_alu_addsub_stage1_op(),
        make_l8_alu_addsub_stage2_op(),
        make_l8_alu_addsub_stage3_op(),
        make_l8_alu_addsub_getobd_op(),
        # L10 post_ops merged into a single phase-10.5 ffn.
        #
        # TODO: replace this dependency-assigned tail layer with a single
        # immediate L10 post-op block. For now the combined layer remains
        # load-bearing for SUB/DIV smoke, but bitwise propagation is owned
        # only by the attached L10 post-op block so 16-bit XOR is not
        # reprocessed in the late tail layer.
        make_l10_post_ops_combined(),
        # Non-first-PSH SP byte-0 fix (flag C4_NONFIRST_PSH_SP_FIX, default
        # ON): AND helper computing NONFIRST_PSH_SP_SUPPRESS (SP-decrement
        # result == 0xF0 at the SP marker). Scheduled BEFORE tail_bit32
        # (which reads the band) so its NOT-blocker suppresses the 0xF8 SP
        # exactness writer on non-first pushes. Flag-off => no-op + no band
        # (byte-identical).
        make_l10_nonfirst_psh_sp_helper_op(),
        make_tail_bit32_result_correction_op(),
        # L10 EXIT/no-clean-opcode AX_CARRY -> OUTPUT source fix (flag
        # C4_L10_EXIT_AXCARRY, default OFF): a post_op attached AFTER
        # tail_bit32_result_correction that, on the post-LEV EXIT step (MARK_AX
        # + LEAKED OP_LEA + no OUTPUT-owning opcode + no MEM_ADDR_SRC), routes
        # the (correct, uncorrupted) AX_CARRY byte into OUTPUT, out-voting the
        # spurious LEA effective-address materializer that otherwise stamps the
        # stale frame-pointer high nibble (0xF0 instead of 42). Ships
        # test_simple_function under the 6 LEV flags. Flag-off => no rules, no
        # post_op (byte-identical to HEAD).
        make_l10_exit_axcarry_op(),
        # L10 multilocal-ENT AX byte-0 source fix (flag C4_L10_ENT_AXCARRY,
        # default ON in the campaign config): a post_op attached AFTER
        # l10_exit_axcarry that, on the multilocal main-ENT step (MARK_AX +
        # LEAKED OP_LEA ~0.81 + no OUTPUT-owning opcode + no MEM_ADDR_SRC),
        # routes the carried prior AX (AX_CARRY) into OUTPUT, out-voting the
        # spurious LEA effective-address materializer that otherwise stamps the
        # ENT frame-size immediate's high nibble (16 -> AX byte-0 = 0x10).
        # Fixes loop_sum step-1 / loop_mul/loop_pow2 step-0 (~75 programs).
        # Flag-off / non-campaign => no rules, no post_op (byte-identical to
        # golden f2b040aa).
        make_l10_ent_axcarry_op(),
        # Multi-byte ADD high-byte adder (2026-06-12): appends a post_op
        # AFTER tail_bit32_result_correction that writes OUTPUT byte 1 =
        # a1 + b1 + carry at the ADD byte-1 row, completing the multi-byte
        # ADD result the carry-only tail rules leave at byte1=carry. Reads
        # a1 (relayed by ``layer13_add_addend_relay``), b1 (ADDR_B1_LO) and
        # the byte-0 carry (CARRY+1, the clean autoregressive
        # discriminator). Byte-identical on 8-bit ADD; add 1096 4/50->39/50.
        make_l10_add_high_byte_adder_op(),
        # AX byte-1 DUMP band-pass UPPER-cut precursor: writes
        # ``AX_CARRY_OVERFLOW = step(AX_CARRY_HI+2 >= 3.0)`` so the dump FFN
        # below can BAND-PASS Σ AX_CARRY (exclude SHL ~+12.85 / JMP ~+47.86),
        # not just lower-bound it. Must bake BEFORE the dump (the dump's
        # requires-after pins it). See l11_ops.make_ax_byte1_carry_overflow_flag_op.
        make_ax_byte1_carry_overflow_flag_op(),
        # AX byte-1 DUMP repopulate FFN: the carried-vs-fresh GATE half of the
        # AX byte-1 register-dump carry. Copies the prev-step one-hot from
        # ``H1_PREV_STEP`` (filled by ``layer13_ax_byte1_dump_carry``) into the
        # emission band ``H1_DUMP_OUT`` ONLY on carried (non-AX-writing) steps
        # (gated on the AX_CARRY fresh ~-988 / carried ~+2.7 separation +
        # the AX_CARRY_OVERFLOW upper-cut, held to the final block). Standalone
        # PureFFN post_op on the L25 tail block after tail_bit32. See
        # l11_ops.make_ax_byte1_dump_repopulate_op.
        make_ax_byte1_dump_repopulate_op(),
        # AX byte-2/3 ENT-frame zero cap (THE callee-ENT prologue blocker for
        # func/nested/rec/var): on the callee ENT step the AX dump leaks garbage
        # into bytes 2/3 (func_identity_0 step-1 ax=0x0a0a0000) because the L18
        # PC-byte default (OUTPUT_LO+10) wins on the high-byte rows when the L9
        # AX-result zero default does not fire (opcode is ENT, not IMM/ADD).
        # Restores byte=0 on the byte-2/3 dump rows gated on OP_ENT (the clean,
        # program-stable ENT-frame discriminator; callee AX is always 0 there).
        # Standalone PureFFN post_op on the L25 tail block after tail_bit32.
        # Gated by C4_AX_BYTE23_DUMP (default ON). See
        # l11_ops.make_ax_byte23_dump_zero_op.
        make_ax_byte23_dump_zero_op(),
        # AX byte-2/3 LI-LOAD zero cap (THE if_var/var LI-load high-byte blocker):
        # on an ``LI x`` LOAD step the AX dump leaks 0x01 into byte 2 (id425 x=96
        # -> ax=0x010060) because OUTPUT_LO+1 narrowly beats OUTPUT_LO+0 at the
        # byte-2 dump row for x=0x60 (value-dependent ~+1.0 swap). Restores byte=0
        # on the byte-2/3 dump rows gated on OP_LI (the clean, value/address-
        # invariant load-step discriminator; MARK_AX blocks the byte-0 OP_LI
        # spike). Standalone PureFFN post_op on the L25 tail block after the ENT
        # cap. Gated by C4_AX_LI_BYTE23_ZERO (default ON). See
        # l11_ops.make_ax_li_byte23_zero_op.
        make_ax_li_byte23_zero_op(),
        # SUB full-borrow byte-1 0xFF writer (CAMPAIGN-ONLY, C4_SUB_FULL_BORROW):
        # the POST-SLAM half of the sub_borrow_cascade fix. On the row where the
        # L14 precursor lit SUB_FULL_BORROW, overwrites OUTPUT byte 1 = 0xFF.
        # Appended AFTER tail_bit32_result_correction so it is the LAST OUTPUT
        # writer before the LM head and DOMINATES the block-32 (L18) OUTPUT_HI
        # slam (which adds +664 to the 0x00 hi default). Golden byte-identical
        # (registered ONLY when sub_full_borrow_enabled; reads the flag band that
        # is itself flag-gated). See l14_ops.make_sub_full_borrow_byte1_ff_op.
        *(
            [make_sub_full_borrow_byte1_ff_op()]
            if sub_full_borrow_enabled() else []
        ),
        # si/li 16-bit LOAD byte-1 value RESTORE (CAMPAIGN-ONLY,
        # C4_SILI_B1_RESTORE): the POST-SLAM half of the si_li_16bit fix. A
        # standalone PureFFN post_op appended AFTER tail_bit32_result_correction
        # (the LAST OUTPUT writer before the LM head) that re-supplies the
        # captured loaded byte-1 from LI_RELOAD_B1 into OUTPUT, DOMINATING the
        # block-32 (L18) OUTPUT_HI slam additively. Self-gates to a no-op when the
        # flag is off (golden 7f6f2e5d byte-identical). See
        # l14_ops.make_sili_b1_restore_op (+ the CAPTURE half above).
        make_sili_b1_restore_op(),
        # func re-read-LEA byte-0 LO-nibble RESTORE (Bug #2, CAMPAIGN-ONLY,
        # C4_FUNC_LEA_B0_RESTORE): the POST-SLAM half. A standalone PureFFN post_op
        # appended AFTER tail_bit32_result_correction (the LAST OUTPUT writer
        # before the LM head) that re-establishes the captured re-read LEA byte-0
        # LO nibble from LEA_REREAD_B0 into OUTPUT_LO via a per-cell
        # winner-take-all that DOMINATES the block-42 (L21) ~4.13e9 LO slam.
        # Self-gates to a no-op when the flag is off (golden d0619711
        # byte-identical). See l14_ops.make_func_lea_b0_restore_op (+ the CAPTURE
        # half above).
        make_func_lea_b0_restore_op(),
        # STACK0 byte-0 carried-step flag precursor (Root 2): writes the BOUNDED
        # ``STACK0_B0_CARRIED`` gate flag at an EARLY block (L7 anchor) where the
        # same-step H3 byte-0 one-hot is still bounded (fresh ~3.3 present,
        # carried ~0 absent) -- BEFORE the L25 corruptor nukes H1/H3 to ~-289M.
        # The flag persists to the L25 tail where the dump FFN gates on it (a
        # raw H1/H3 read there would drive the dump's silu gate to ~+10^9). See
        # l11_ops.make_stack0_byte0_carried_flag_op.
        make_stack0_byte0_carried_flag_op(),
        # STACK0 byte-0 PREV-sharpness flag precursor (Root 2 re-point gate):
        # writes the BOUNDED ``STACK0_B0_SHARP`` flag on the L25 tail, reading the
        # PREV band the carry head populated. SHARP=1 only when PREV is a CLEAN
        # single-slot one-hot (the framing-drift case) and 0 on a SMEAR (a
        # multi-byte arithmetic result). The dump's DIRECT H1/H3 re-point ANDs
        # this flag so it fires ONLY on the genuinely-corrupted rows and stays a
        # no-op on healthy emissions. See l11_ops.make_stack0_byte0_sharp_flag_op.
        make_stack0_byte0_sharp_flag_op(),
        # STACK0 byte-0 RATIO-based PREV-dominant flag precursor (Root 2 smear
        # gate): writes ``STACK0_B0_PREV_DOM`` = 1 when ONE PREV slot dominates (a
        # clean carried one-hot of ANY magnitude). Unlike SHARP (absolute per-slot
        # margin -> misses a small clean one-hot like the comparison-result byte
        # 0x01) this RATIO test is magnitude-independent, so the NON-COMPARISON
        # blocker's smear rule darkens the add_16bit smear WITHOUT darkening small
        # comparison results. See l11_ops.make_stack0_byte0_prev_dom_flag_op.
        make_stack0_byte0_prev_dom_flag_op(),
        # STACK0 byte-0 NON-COMPARISON blocker precursor (Root 2 DEFAULT-ON gate):
        # writes the BOUNDED ``STACK0_B0_NOT_CMP`` flag on the L25 tail. NOT_CMP = 1
        # on an arithmetic-result / JMP STACK0 row (rule 1: per-step arith/JMP
        # opcode) OR a SMEARED-PREV row (rule 2: the add_16bit over-fire), and 0 on
        # the genuine comparison drift rows. Reads the WIDENED-layout opcode bands
        # (the static registry mismaps them — the same dim-map error Root 3
        # corrected). The dump's re-point reads this flag as a -1000 BLOCKER so it
        # darkens the add_16bit + jmp_forward over-fire while leaving the if/bool/
        # expr drift rows firing -> the carry ships DEFAULT-ON. See
        # l11_ops.make_stack0_byte0_not_cmp_flag_op.
        make_stack0_byte0_not_cmp_flag_op(),
        # STACK0 byte-0 DUMP repopulate FFN (Root 2): the carried-vs-fresh GATE
        # half. On carried STACK0-marker rows (gated on the bounded
        # ``STACK0_B0_CARRIED`` flag AND, for the re-point path, ``STACK0_B0_SHARP``)
        # it re-supplies the prev-step byte-0 one-hot from ``STACK0_B0_{H1,H3}_PREV``
        # (filled by ``stack0_byte0_dump_carry``). Flag OFF: into the inert
        # ``STACK0_B0_DUMP_{H1,H3}`` bands (byte-identical). Flag ON: DIRECTLY into
        # the byte's own ``H1``/``H3`` LM-head emission cells (the re-point fix --
        # reaches the byte token THROUGH the block-38 corruption it runs after).
        # Standalone PureFFN post_op on the L25 tail block after tail_bit32. See
        # l11_ops.make_stack0_byte0_dump_repopulate_op.
        make_stack0_byte0_dump_repopulate_op(),
        # ENT saved-BP store DUMP repopulate FFN: the OP_ENT gate + OUTPUT
        # re-supply half. On the ENT-store val-byte predictor rows (gated on the
        # high OP_ENT broadcast ~10.7 + the MEM_VAL_B markers, so SI/SC/PSH/JSR
        # stores stay dark) it re-supplies the carried old_BP nibbles from
        # ``BP_SAVE_PREV`` (filled by ``bp_save_prev_carry``) DIRECTLY into
        # ``OUTPUT_LO/HI`` -- overwriting the 0xFF garbage AFTER the tail
        # corruptor runs, so the LM head emits the clean old_BP byte tokens.
        # Flag OFF (``C4_BP_SAVE_DUMP=0``): writes nothing into OUTPUT
        # (byte-identical). Standalone PureFFN post_op on the L25 tail block after
        # tail_bit32. See l11_ops.make_bp_save_dump_repopulate_op.
        make_bp_save_dump_repopulate_op(),
        # No-STACK0 (30-token) STEP_END OUTPUT-clear FFN: drives OUTPUT_LO/HI
        # hugely negative at the MARK_SE_ONLY row so the LM head emits REG_PC (not
        # a stray byte) after STEP_END, killing the +1-token/step frame drift that
        # desyncs the 30-token fixed-stride decode. Standalone PureFFN post_op on
        # the L25 tail block after tail_bit32 (LAST OUTPUT writer before the head).
        # Gated by C4_NO_STACK0_EMIT; flag-OFF bakes NO units (byte-identical).
        # See l0_ops.make_no_stack0_se_output_clear_op.
        make_no_stack0_se_output_clear_op(),
        # No-STACK0 (30-token) MEM-MARKER-row OUTPUT-clear FFN: drives OUTPUT_LO/HI
        # hugely negative at the bounded NEXT_MEM one-hot row (the BP-byte3 row
        # whose logits decide the MEM marker) so the LM head emits Token.MEM (not
        # a stray ALU-result value byte) -> the NEXT_MEM->NEXT_SE->NEXT_PC marker
        # chain fires and the next step's REG_PC is emitted. Fixes the
        # var_update SI-store / if_var BZ-branch SILENCE collapse (the model went
        # quiet after the store/branch because the leaked result byte broke the
        # marker chain). Standalone PureFFN post_op on the L25 tail block after
        # no_stack0_se_output_clear (LAST OUTPUT writer at the MEM-marker row).
        # Gated by C4_NO_STACK0_EMIT + C4_MEM_MARKER_OUTPUT_CLEAR (default ON);
        # flag-OFF bakes NO units (byte-identical to HEAD's 35-token golden).
        # See l0_ops.make_no_stack0_mem_marker_output_clear_op.
        make_no_stack0_mem_marker_output_clear_op(),
        # No-STACK0 (30-token) PC value-byte OUTPUT-clear FFN (Inc 0): at the
        # PC value-byte rows (H1+0 PC-marker proximity + IS_BYTE + per-byte
        # BYTE_INDEX one-hot) sinks OUTPUT high nibbles so the PC HIGH bytes
        # default to 0x00 and the byte-0 nibble-1 leak is cleared, killing the
        # self-reinforcing 0x01 PC-replication that is the dominant 30-token
        # blocker (288/375 PC-wrong). Appended AFTER no_stack0_se_output_clear
        # (LAST OUTPUT writer at the PC rows), so it overrides every upstream
        # leak source (block 33 L15 nibble_copy + block 41 L25 tail). Gated by
        # C4_NO_STACK0_EMIT; flag-OFF bakes NO units (byte-identical).
        # See l0_ops.make_no_stack0_pc_highbyte_clear_op.
        make_no_stack0_pc_highbyte_clear_op(),
        # L15 attention resize: add LEV/ALU/store-disambiguation heads
        # (phase=14.9 so it fires before _set_layer15_memory_lookup populates
        # the heads).
        make_l15_attention_resize_op(),
        make_layer14_alu_high_byte_relay_op(),
        make_layer15_store_stack0_sp_byte0_addr_op(),
        make_layer15_si_mem_addr0_from_stack0_op(),
        # Model-level bake that runs BEFORE legacy_bake (phase 998) so its
        # FFN unit writes survive the rightsize pass at end of legacy_bake.
        make_function_call_weights_op(),
        # Model-level bake that runs BEFORE legacy_bake (phase 998) so its
        # L6 FFN unit writes survive the rightsize pass at end of legacy_bake.
        make_io_putchar_routing_op(),
        # Neural-I/O THINK-tag protocol PUTCHAR bake (phase 6.6): flag-gated.
        # Always registered for dep-graph stability; bake_fn is a no-op when
        # ``enable_neural_io_think_protocol=False`` (the default). When
        # enabled, the model emits THINKING_END, byte token, THINKING_START
        # at the end of a PUTCHAR step (BLOG_SPEC.md:851 — canonical neural
        # I/O mode). See c4_release/docs/NEURAL_IO_VIA_THINK_PROTOCOL_PLAN.md.
        make_putchar_think_protocol_op(
            enable_neural_io_think_protocol=enable_neural_io_think_protocol,
        ),
        # Neural-I/O THINK-tag protocol PRTF bake (phase 6.6): flag-gated
        # under the same ``enable_neural_io_think_protocol`` switch as
        # PUTCHAR. Phase 2a: bake_fn is a no-op stub — the full
        # PRTF byte-emission chain is already baked under
        # ``enable_conversational_io`` (L5/L6/L7/L8/L9/L10/L15 helpers
        # in setup_helpers.py + vm_step.py). This op anchors the
        # PRTF-specific reads/writes in the dep graph and exposes the
        # scaffolding so a Phase-2b worker can flip the flag to True
        # and bake any additional cleanup units (e.g. IO_FORMAT_POS
        # reset) without restructuring the migration chain.
        # See c4_release/docs/V9_PRTF_NEURAL_PLAN.md for the full design.
        make_prtf_think_protocol_op(
            enable_neural_io_think_protocol=enable_neural_io_think_protocol,
        ),
        # OPEN/CLOS TOOL_CALL boundary-opcode dep anchor (phase 6.7).
        # Per BLOG_SPEC.md:853, these two opcodes are not candidates
        # for neural I/O (file descriptors cross the host boundary).
        # The canonical design is to emit a TOOL_CALL token at step end
        # so the runner can perform os.open/os.close — already baked
        # under ``enable_tool_calling=True`` via make_tool_call_*_op
        # below. This op's bake_fn is always a no-op; it exists to
        # document the OPEN/CLOS reads/writes in the dep graph and to
        # anchor a future Phase A bake (e.g. an OPEN/CLOS-specific
        # marker dim). See c4_release/docs/V9_PRTF_NEURAL_PLAN.md
        # § Phase A.
        make_open_clos_tool_call_op(
            enable_tool_calling=enable_tool_calling,
        ),
        # V9 GETCHAR routing (BLOG_SPEC.md:851). Phase 998.9: mirrors
        # io_putchar_routing in shape — routes STDIN_BYTE_LO/HI -> AX_CARRY
        # at GETCHAR & MARK_AX so AX byte 0 receives the gathered USER_INPUT
        # byte. Phase 1: registered but disabled. See
        # docs/V9_GETCHAR_READ_NEURAL_PLAN.md §3.3.
        make_layer6_getchar_routing_op(enable=False),
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
        # V18 Phase 2 bakes (step resumption + PC/SP latch + PRTF capture +
        # PRTF transport). All four are double-gated:
        # ``enable_conversational_io`` AND ``enable`` must be True. Inner
        # ``enable=True`` flipped (Phase 2 entry, 2026-05-12) so the bakes
        # form the full ``capture(3c) → transport(3d) → replay(3b)`` chain
        # plus the ``step_resume(3a)`` head-routing flip whenever
        # ``enable_conversational_io=True`` reaches the compiler.
        #
        # Status: bake unit tests pass, but the end-to-end round-trip with
        # ``enable_conversational_io=True`` does not yet emit THINKING_END
        # from the model — the V18 Python handler block at run_vm.py:776-825
        # therefore remains the production path. Removing it is gated on
        # the smoke loop emitting THINKING_END → bytes → THINKING_START →
        # REG_PC autonomously. See V18_CONVO_IO_NEURAL_PLAN.md §3 / Phase 2.
        make_convo_io_step_resume_op(
            enable_conversational_io=enable_conversational_io,
            enable=True,
        ),
        make_convo_io_pc_sp_latch_op(
            enable_conversational_io=enable_conversational_io,
            enable=True,
        ),
        make_convo_io_prtf_capture_op(
            enable_conversational_io=enable_conversational_io,
            enable=True,
        ),
        make_convo_io_prtf_transport_op(
            enable_conversational_io=enable_conversational_io,
            enable=True,
        ),
        # Model-level bakes (run after legacy_bake's per-layer/head/embed work)
        make_head_bake_op(),
        # AX byte-1 DUMP emission columns: mirrors the H1 high-byte one-hot
        # columns onto H1_DUMP_OUT so the LM head re-emits the carried high
        # byte. Phase=1002 (additive, AFTER head_bake). Byte-identical on fresh
        # steps (H1_DUMP_OUT == 0). See model_ops.make_ax_byte1_dump_head_bake_op.
        make_ax_byte1_dump_head_bake_op(),
        # AX byte-1 FULL-WIDTH emission columns (ISA-DSL full_width_byte_emission):
        # a 256-cell wide value band + un-aliased LM-head columns 16-255 that
        # break the H-band mod-16 cap. Gated by C4_AX_BYTE1_FULL_WIDTH
        # (default-OFF -> band omitted -> byte-identical). The emission half the
        # deferred IMM-decode relay plugs into to fix edge_literal / byte1 >= 16.
        # See model_ops.make_ax_byte1_full_width_emission_op.
        make_ax_byte1_full_width_emission_op(),
        # AX byte-1 FULL-WIDTH band FILL (the value-source re-point FFN):
        # reconstructs byte-1's full value (16-255) into AX_BYTE1_FULL_WIDE by
        # AND-ing the ALU_LO/ALU_HI nibble pair at the byte-1 predictor row, so
        # the un-aliased LM-head column emits the correct high byte (fixes
        # edge_literal / IMM byte1 >= 16). L25-tail block FFN; gated by
        # C4_AX_BYTE1_FULL_WIDTH (flag-off => zero rules => byte-identical).
        # See model_ops.make_ax_byte1_full_width_fill_op.
        make_ax_byte1_full_width_fill_op(),
        # AX byte-1 HIGH-NIBBLE emission columns (narrow alias-break): a 16-cell
        # AX_BYTE1_HINIB band + un-aliased HIGH-nibble LM-head columns 16-255
        # that break the byte-1 mod-16 cap with only +1 head (d_model
        # 1090->1199, n_heads 10->11) vs the 256-cell FULL_WIDTH's +3 heads.
        # Phase=1002 (additive, AFTER head_bake). Gated by C4_AX_BYTE1_HINIB
        # (default-OFF -> band omitted -> byte-identical). See
        # model_ops.make_ax_byte1_hinib_emission_op.
        make_ax_byte1_hinib_emission_op(),
        # AX byte-1 HIGH-NIBBLE band FILL: lights AX_BYTE1_HINIB+hi from the
        # carried H3 high-nibble one-hot (H3_PREV_STEP+(4+hi)) at the byte-1
        # predictor row, present on BOTH the fresh IMM and carried persisting-AX
        # steps -> emits the correct byte1>=16 high nibble on every step.
        # L25-tail block FFN; gated by C4_AX_BYTE1_HINIB (flag-off => zero rules
        # => byte-identical). See model_ops.make_ax_byte1_hinib_fill_op.
        make_ax_byte1_hinib_fill_op(),
        # STACK0 byte-0 DUMP emission columns (Root 2): mirrors the byte-value
        # H1/H3 one-hot columns onto STACK0_B0_DUMP_{H1,H3} so the LM head
        # re-emits the carried STACK0 byte-0 on carried steps. Phase=1002
        # (additive, AFTER head_bake). Byte-identical on fresh steps (DUMP bands
        # == 0). Gated by C4_STACK0_B0_DUMP (default-on). See
        # model_ops.make_stack0_byte0_dump_head_bake_op.
        make_stack0_byte0_dump_head_bake_op(),
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
        # Qwen R1: NORM_COMPENSATOR seed. Registered unconditionally so
        # the dep-graph topology is stable across the
        # ``C4_QWEN_EXPORT_COMPAT`` flag; the bake_fn is a no-op when the
        # flag is OFF (and an additional no-op when the dim was not
        # declared). Phase=1400 — runs after every structural model
        # bake (expand_wrapper_blocks=1300).
        make_norm_compensator_seed_op(),
    ]


def all_alu_postop_attach_ops() -> list:
    """Return the L8-L13 ALU post-op attach ops.

    These are NOT included in all_core_ops() because they attach the ALU to
    ``block.post_ops`` AFTER set_vm_weights' legacy bake has populated the
    FFN weights. Use _dispatch_migrated_block_ops in vm_step.py to fire them
    at the right time (post-legacy-bake, pre-right-size).

    2026-06-03 (L17 tail MUL double-fire fix): ``make_l11_alu_postop_attach_op``
    is intentionally OMITTED. The L11 MUL ``FlattenedALUMul`` post-op attach
    was redundant: it installed a from-scratch MUL pipeline (reads
    ALU_LO/HI, OP_MUL, MARK_AX from BD; writes OUTPUT_LO/HI += 2.0 indicators
    via ``GEToBDConverter``) at L11, then ``make_l12_alu_postop_attach_op``
    installed an identical pipeline at L12. After ``_expand_wrapper_blocks``
    (Phase 10.B) split both post_ops into adjacent wrapper blocks 25 and 27,
    the second run re-fired ``OUTPUT_HI/LO += 2.0`` on the same MARK_AX
    row, doubling the signal and clobbering OUT_LO[expected]. Margin was
    uniformly -16.00 across 58 / 384 of the 1096 failures at L17.post_ops[0]
    block=27 ``expected-token-never-wins`` (see
    docs/L17_TAIL_MUL_DOUBLE_FIRE.md). The L11 ``layer11_mul_partial`` FFN
    rules still own the partial-product staging in block 11's FFN; only the
    redundant post-op wrap is removed.
    """
    return [
        make_l8_alu_postop_attach_op(),
        make_l9_alu_postop_attach_op(),
        make_l10_alu_postop_attach_op(),
        # make_l11_alu_postop_attach_op() — removed; L12's attach covers
        # the single FlattenedALUMul fire. See docstring above.
        make_l12_alu_postop_attach_op(),
        make_l13_alu_postop_attach_op(),
    ]
