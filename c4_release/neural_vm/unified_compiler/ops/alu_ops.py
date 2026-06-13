"""ALU composite/wrapping op factories. See ../migrated_ops.py for history."""

from ..ir import CompilerIR
from ..layer_compiler import Operation
from .shared import _as_setdim_proxy, _make_alu_postop_attach_op, _ensure_l11_mul_module


def _mark_structural_declarations(op: Operation) -> Operation:
    """Mark module-assembly ops as safe for declarations-only dispatch."""
    op.declarative_bake_fn = op.bake_fn
    op.declarative_authority = "structural_model"
    return op


# ---------------------------------------------------------------------------
# 4-stage SHL/SHR ops (efficient-mode replacement for ALUShift wrapper).
# See shared._ALUShiftCompositeBuilder for the shared builder class.
# ---------------------------------------------------------------------------

from .shared import _ALUShiftCompositeBuilder

def make_alu_shift_composite_ops():
    """Build the 5 cooperating ops (4 ffn stages + 1 block install).

    Returns ``[bdtoge, precompute, select, getobd, install]`` — all sharing
    the same ``_ALUShiftCompositeBuilder`` so the install op can hand the
    fully-constructed composite to ``model.blocks[13].ffn``.
    """
    builder = _ALUShiftCompositeBuilder()

    def make_bdtoge():
        def bake(ffn, dim_positions, S):
            from ...efficient_alu_neural import ShiftBDToGEStage
            BD = _as_setdim_proxy(dim_positions)
            composite = builder.ensure(S, BD)
            composite.bdtoge_stage = ShiftBDToGEStage(S, BD)

        return Operation(
            name="l13_alu_shift_bdtoge",
            reads={"MARK_AX", "ALU_LO", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI",
                   "OP_SHL", "OP_SHR"},
            writes=set(),
            kind="ffn",
            declarative_bake_fn=bake,
            declarative_authority="structural_model",
            migrated=True,
            # Dim-ownership claims: empty. ``bake`` attaches the BD->GE
            # stage to the shared ``_ALUShiftCompositeBuilder``; the
            # install op below ultimately swaps ``model.blocks[13].ffn``
            # for the assembled composite (module replacement), not
            # per-cell ``(layer, scope, identifier, column)`` writes.
            claims=set(),
            # Module-replacement sentinel: dynamic verifier (Mode B) skips
            # drift detection; static (Mode A) snapshot diffing unaffected.
            produces={'__module_replacement': 'L13.ffn[ALUShiftComposite]'},
        smoke_tests={
            "TestSmoke32Bit::test_shl_8bit",
            "TestSmoke32Bit::test_shr_8bit",
            "TestSmokeShift::test_shl",
            "TestSmokeShift::test_shr",
        },
        spec_section="BLOG_SPEC.md#shifts",
            # Tier A opcode gating: stage of the L13 ALUShiftComposite SHL/SHR
            # pipeline. ``ALUShiftComposite.forward`` (efficient_alu_neural.py)
            # computes op_shl / op_shr from BD OP_SHL / OP_SHR flags and zeros
            # all residual writes outside those opcodes via opcode_mask.
            opcodes={"OP_SHL", "OP_SHR"},
        )

    def make_precompute():
        def bake(ffn, dim_positions, S):
            from ...efficient_alu_neural import ShiftPrecomputeStage
            BD = _as_setdim_proxy(dim_positions)
            composite = builder.ensure(S, BD)
            composite.precompute_stage = ShiftPrecomputeStage(S, BD)

        return Operation(
            name="l13_alu_shift_precompute",
            reads={"MARK_AX", "ALU_LO", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI",
                   "OP_SHL", "OP_SHR"},
            writes=set(),
            kind="ffn",
            declarative_bake_fn=bake,
            declarative_authority="structural_model",
            migrated=True,
            # Dim-ownership claims: empty. ``bake`` attaches the
            # SHL/SHR sub-chunk precompute stage to the shared
            # ``_ALUShiftCompositeBuilder``; module assembly, not
            # per-cell writes.
            claims=set(),
            # Module-replacement sentinel: dynamic verifier (Mode B) skips
            # drift detection; static (Mode A) snapshot diffing unaffected.
            produces={'__module_replacement': 'L13.ffn[ALUShiftComposite]'},
        smoke_tests={
            "TestSmoke32Bit::test_shl_8bit",
            "TestSmoke32Bit::test_shr_8bit",
            "TestSmokeShift::test_shl",
            "TestSmokeShift::test_shr",
        },
        spec_section="BLOG_SPEC.md#shifts",
            # See l13_alu_shift_bdtoge gating note: SHL/SHR pipeline stage.
            opcodes={"OP_SHL", "OP_SHR"},
        )

    def make_select():
        def bake(ffn, dim_positions, S):
            from ...efficient_alu_neural import ShiftSelectStage
            BD = _as_setdim_proxy(dim_positions)
            composite = builder.ensure(S, BD)
            composite.select_stage = ShiftSelectStage(S, BD)

        return Operation(
            name="l13_alu_shift_select",
            reads={"MARK_AX", "ALU_LO", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI",
                   "OP_SHL", "OP_SHR"},
            writes=set(),
            kind="ffn",
            declarative_bake_fn=bake,
            declarative_authority="structural_model",
            migrated=True,
            # Dim-ownership claims: empty. ``bake`` attaches the
            # shift-select stage to the shared
            # ``_ALUShiftCompositeBuilder``; module assembly, not
            # per-cell writes.
            claims=set(),
            # Module-replacement sentinel: dynamic verifier (Mode B) skips
            # drift detection; static (Mode A) snapshot diffing unaffected.
            produces={'__module_replacement': 'L13.ffn[ALUShiftComposite]'},
        smoke_tests={
            "TestSmoke32Bit::test_shl_8bit",
            "TestSmoke32Bit::test_shr_8bit",
            "TestSmokeShift::test_shl",
            "TestSmokeShift::test_shr",
        },
        spec_section="BLOG_SPEC.md#shifts",
            # See l13_alu_shift_bdtoge gating note: SHL/SHR pipeline stage.
            opcodes={"OP_SHL", "OP_SHR"},
        )

    def make_getobd():
        def bake(ffn, dim_positions, S):
            from ...efficient_alu_neural import ShiftGEToBDStage
            BD = _as_setdim_proxy(dim_positions)
            composite = builder.ensure(S, BD)
            composite.getobd_stage = ShiftGEToBDStage(S, BD)

        return Operation(
            name="l13_alu_shift_getobd",
            reads={"MARK_AX", "ALU_LO", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI",
                   "OP_SHL", "OP_SHR"},
            writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
            kind="ffn",
            declarative_bake_fn=bake,
            declarative_authority="structural_model",
            migrated=True,
            # Wave 5 (docs/PRODUCES_CONSUMES_MIGRATION.md): final GE->BD stage
            # of the SHL/SHR composite writes the shifted result at the
            # AX-marker row into OUTPUT_LO/HI -- same convention as the L14
            # POC ops (AX_byte0 slot). Operand reads (ALU_LO/HI, AX_CARRY_LO/HI)
            # are cross-step durables populated by upstream L7/L8 setup ops,
            # not in-step fresh residuals; consumes_fresh stays empty.
            # Dim-ownership claims: empty. ``bake`` attaches the final
            # GE->BD stage to the shared ``_ALUShiftCompositeBuilder``;
            # module assembly, not per-cell ``(layer, scope, identifier,
            # column)`` writes. The ``writes`` set above documents the
            # OUTPUT residual dims the assembled composite touches at
            # runtime.
            claims=set(),
            # Module-replacement sentinel: dynamic verifier (Mode B) skips
            # drift detection; static (Mode A) snapshot diffing unaffected.
            produces={'__module_replacement': 'L13.ffn[ALUShiftComposite]'},
        smoke_tests={
            "TestSmoke32Bit::test_shl_8bit",
            "TestSmoke32Bit::test_shr_8bit",
            "TestSmokeShift::test_shl",
            "TestSmokeShift::test_shr",
        },
        spec_section="BLOG_SPEC.md#shifts",
            # See l13_alu_shift_bdtoge gating note: SHL/SHR pipeline stage.
            opcodes={"OP_SHL", "OP_SHR"},
        )

    def make_install():
        def bake(block, dim_positions, S):
            if builder.composite is None:
                return  # No stage bakes ran (lookup mode safety).
            block.ffn = builder.composite

        return Operation(
            name="l13_alu_shift_install",
            # Phase 11.A r3: dropped phase=13.5 — target_op_name +
            # requires['after']: l13_alu_shift_getobd already pin
            # placement and intra-target order.
            reads=set(),
            writes=set(),
            kind="block",
            declarative_bake_fn=bake,
            declarative_authority="structural_model",
            # Phase 8.A.4 fix (smoke_bisect_20260602_1339): originally pointed
            # at ``l13_alu_shift_getobd`` (a kind="ffn" composite stage), but
            # that stage is only registered in efficient mode and even then
            # the dep graph did not anchor it at L13 -- block fusion
            # collapsed L12.ffn/L13.ffn into a single dead block, breaking
            # 29 AX-write smoke tests. Bind to ``_layer13_attn_dep_anchor``
            # instead: it is always registered (regardless of alu_mode) and
            # is pinned at L13 via ``requires["after"]:
            # _layer12_ffn_dep_anchor``, mirroring the same pattern used by
            # ``efficient_l10_andorxor_wrap`` against ``layer10_carry_relay``.
            target_op_name="_layer13_attn_dep_anchor",
            migrated=True,
            # Keep ``requires["after"]`` on the final kind="ffn" composite
            # stage so the install runs AFTER the composite is fully
            # assembled (block.ffn = builder.composite must come after
            # ShiftGEToBDStage was attached by the getobd op's bake).
            requires={"after": "l13_alu_shift_getobd"},
            # Dim-ownership claims: empty. ``bake`` swaps
            # ``model.blocks[13].ffn`` for the assembled
            # ``ALUShiftComposite`` (module replacement), not per-cell
            # ``(layer, scope, identifier, column)`` writes. Sentinel
            # below documents the structural effect.
            claims=set(),
            # Module-replacement sentinel: dynamic verifier (Mode B) skips
            # drift detection; static (Mode A) snapshot diffing unaffected.
            produces={'__module_replacement': 'L13.ffn[ALUShiftComposite]'},
        smoke_tests={
            "TestSmoke32Bit::test_shl_8bit",
            "TestSmoke32Bit::test_shr_8bit",
            "TestSmokeShift::test_shl",
            "TestSmokeShift::test_shr",
        },
        spec_section="BLOG_SPEC.md#shifts",
            # Tier A opcode gating: the installed ALUShiftComposite forward
            # is fully SHL/SHR-gated (see ``efficient_alu_neural.py``,
            # ``ALUShiftComposite.forward`` lines 1330-1346: op_shl + op_shr
            # opcode_mask zeros all OUTPUT writes outside those opcodes).
            opcodes={"OP_SHL", "OP_SHR"},
        )

    return [
        make_bdtoge(),
        make_precompute(),
        make_select(),
        make_getobd(),
        make_install(),
    ]


# Keep individual factory shims for callers that want a single op (e.g.,
# unit tests). Each returns a fresh builder so the ops aren't entangled.
def make_l13_alu_shift_bdtoge_op() -> Operation:
    """L13 FFN stage 1: BD → GenericE format conversion (standalone factory)."""
    return make_alu_shift_composite_ops()[0]


def make_l13_alu_shift_precompute_op() -> Operation:
    """L13 FFN stage 2: SHL/SHR sub-chunk precompute (standalone factory)."""
    return make_alu_shift_composite_ops()[1]


def make_l13_alu_shift_select_op() -> Operation:
    """L13 FFN stage 3: shift-select FFN (standalone factory)."""
    return make_alu_shift_composite_ops()[2]


def make_l13_alu_shift_getobd_op() -> Operation:
    """L13 FFN stage 4: GenericE → BD format conversion (standalone factory)."""
    return make_alu_shift_composite_ops()[3]


def make_l13_alu_shift_install_op() -> Operation:
    """L13 block op: swap ``model.blocks[13].ffn`` for the composite (standalone factory)."""
    return make_alu_shift_composite_ops()[4]


# ---------------------------------------------------------------------------
# ALU post-op attach block-ops (migrated from set_vm_weights lines 2377-2382)
# ---------------------------------------------------------------------------
#
# Block-level migration: the lookup-mode ALU layers attach a structural neural
# ALU to ``block.post_ops`` so it runs on top of the lookup-table FFN. Each
# layer/ALU pairing is a separate op so the compiler can place/reorder/expand
# them independently in the future. (Previously these factories built a
# HybridALUBlock wrapper, hence the legacy name; HybridALUBlock has been
# removed and the ALU is now attached directly via ``block.post_ops``.)
#
# Each factory takes alu_mode='lookup' (production default). Efficient mode is
# TODO — its semantics differ (replace ffn vs attach post_op) and need a
# separate migration pass.
#
# The op body (``_make_alu_postop_attach_op``) lives in ``shared.py`` so it can
# be reused; the factories below just thread the per-layer / per-ALU-class
# arguments.


def _annotate_module_replacement(op: Operation, sentinel_value: str) -> Operation:
    """Tag an op as a structural module-replacement (post-construction).

    Used by the ``make_lN_alu_postop_attach_op`` factories below, which
    delegate to ``_make_alu_postop_attach_op`` in ``shared.py`` and so
    cannot pass ``claims`` / ``produces`` via the constructor. Mutating
    after construction is safe: ``claims`` is a mutable set field, and
    ``produces`` has a setter on ``Operation``. The
    ``__module_replacement`` sentinel key is skipped by the validation
    in ``LayerCompiler.add_op`` (see the ``dim_name.startswith("__")``
    guard there).
    """
    op.claims = set()
    op.produces = {'__module_replacement': sentinel_value}
    return op


def make_l8_alu_postop_attach_op(alu_mode: str = 'lookup') -> Operation:
    # Dim-ownership claims: empty. ``bake`` inserts an
    # ``AddSub5StageBlock`` (= ``ALUAddSub``) into
    # ``model.blocks[8].post_ops`` -- module attach, not per-cell
    # ``(layer, scope, identifier, column)`` writes. Sentinel below
    # documents the structural effect.
    return _annotate_module_replacement(
        _mark_structural_declarations(
            _make_alu_postop_attach_op(
                "l8_alu_postop_attach", 8, "ALUAddSub", alu_mode,
                same_layer_as="layer8_alu",
                # layer8_alu binds to ``layer10_byte_passthrough`` (kind=attn);
                # the postop attach co-places via the same anchor.
                target_op_name="layer10_byte_passthrough",
            )
        ),
        'L8.post_ops[AddSub5StageBlock]',
    )


def make_l9_alu_postop_attach_op(alu_mode: str = 'lookup') -> Operation:
    # Dim-ownership claims: empty. ``bake`` inserts an
    # ``AddSub5StageBlock`` (= ``ALUAddSub``) into
    # ``model.blocks[9].post_ops`` -- module attach. Sentinel below
    # documents the structural effect.
    return _annotate_module_replacement(
        _mark_structural_declarations(
            _make_alu_postop_attach_op(
                "l9_alu_postop_attach", 9, "ALUAddSub", alu_mode,
                same_layer_as="layer9_alu",
                # layer9_alu binds to ``layer9_marker_suppress`` (kind=ffn).
                target_op_name="layer9_marker_suppress",
            )
        ),
        'L9.post_ops[AddSub5StageBlock]',
    )


def make_lookup_mode_l10_bitwise_rules_op() -> Operation:
    """Lookup-mode rule-derived replacement for the L10 ``ALUAndOrXor`` post_op.

    V8 follow-up wave (`docs/V8_DELETE_AUDIT_2026_06_04.md`): the legacy
    install path for L10 lookup-mode is ``_make_alu_postop_attach_op``
    instantiating ``efficient_alu_neural.ALUAndOrXor(S, BD)`` and inserting it
    into ``block.post_ops[0]``. ``_expand_wrapper_blocks`` then splits that
    post_op into its own passthrough TransformerBlock, so it runs as the FFN
    of a dedicated block immediately after L10's existing FFN.

    This factory replaces that imperative composite with a rule-derived
    ``PureFFN`` baked from ``wide_alu_dsl.bitwise_rules`` for AND/OR/XOR
    (512 rules per opcode = 1536 hidden units total). Byte-identity at the
    decoded OUTPUT_LO/HI byte is already proven by the POC tests in
    ``tests/test_wide_alu_dsl.py::test_bitwise_rules_byte_identity_*``
    (which compare the same rule-derived PureFFN against ``ALUAndOrXor``
    on randomized AND/OR/XOR inputs).

    Forward semantics:
      - Input: BD-format residual ``[B, seq_len, d_model]``.
      - Output: input + SwiGLU contribution that writes ``2.0 / S`` into
        ``OUTPUT_LO+(a OP b & 0xF)`` and ``OUTPUT_HI+((a OP b) >> 4)`` at
        ``MARK_AX > 0.5`` positions when the matching opcode flag is set.
      - Per-bit semantics differ from ``ALUAndOrXor`` (which routes through
        a BD->GE->FFN->GE->BD pipeline using sigmoid step-pair indicators)
        but the decoded one-hot OUTPUT byte is identical.

    Note: ``ALUAndOrXor`` also writes CARRY flags and AX_FULL_*; those are
    only populated for ADD/SUB/wide ops (MUL/SHL/SHR/DIV/MOD), not for
    AND/OR/XOR, so the rule-derived path is OUTPUT-complete for bitwise.
    """
    def bake(block, dim_positions, S):
        from ...base_layers import PureFFN
        from ..primitives import Primitives
        from ..wide_alu_dsl import bitwise_rules

        # Generate per-opcode rule batches (512 rules each: 256 lo + 256 hi).
        rule_list: list = []
        for op_name, opcode_gate in (
            ("and", "OP_AND"),
            ("or", "OP_OR"),
            ("xor", "OP_XOR"),
        ):
            rule_list.extend(bitwise_rules(
                op=op_name,
                operand_a_lo="ALU_LO",
                operand_a_hi="ALU_HI",
                operand_b_lo="AX_CARRY_LO",
                operand_b_hi="AX_CARRY_HI",
                result_lo="OUTPUT_LO",
                result_hi="OUTPUT_HI",
                opcode_gate=opcode_gate,
                marker_gate="MARK_AX",
                S=S,
            ))
        rules = tuple(rule_list)
        assert len(rules) == 3 * 512, (
            f"bitwise_rules: expected 1536 rules (3 opcodes x 512), got {len(rules)}"
        )

        # Size the new PureFFN to match the parent block's d_model. The
        # post_op runs in its own passthrough block downstream (see
        # ``_expand_wrapper_blocks`` in ``vm_step.py``) where it becomes
        # the block's ``ffn``, so its input/output dim must match the
        # residual width of the surrounding model.
        ffn_in = block.ffn
        if hasattr(ffn_in, "W_up") and ffn_in.W_up is not None:
            d_model = int(ffn_in.W_up.shape[1])
        else:
            d_model = int(getattr(ffn_in, "dim", 512))
        new_ffn = PureFFN(dim=d_model, hidden_dim=len(rules))

        bd_proxy = _as_setdim_proxy(dim_positions)
        names = Primitives.ffn_rule_dim_names(rules)
        dim_pos = Primitives.dim_positions_from_bd(bd_proxy, names)
        end = Primitives.lower_ffn_rules(
            new_ffn, rules, dim_pos, start_unit=0, S=S,
        )
        assert end == len(rules), (
            f"lower_ffn_rules wrote {end} units; expected {len(rules)}"
        )

        # Insert as post_op[0] to match the legacy ``_make_alu_postop_attach_op``
        # behaviour. ``_expand_wrapper_blocks`` will split each post_op into
        # its own passthrough block, preserving execution order. The rebake
        # in ``_expand_wrapper_blocks._rebake_as_pureffn`` is a no-op for
        # vanilla ``PureFFN`` (type check at vm_step.py:2641 short-circuits).
        block.post_ops.insert(0, new_ffn)

    return Operation(
        name="l10_alu_postop_attach",
        reads=set(),
        writes=set(),
        kind="block",
        # Phase 8.G.6 anchor: layer10_alu binds to ``layer10_carry_relay``
        # (kind=attn). Same target as the legacy ``_make_alu_postop_attach_op``
        # path so the dynamic scheduler co-places this op with the L10 ALU.
        target_op_name="layer10_carry_relay",
        bake_fn=bake,
        # Same phase as the legacy factory (1180 + 10 * 0.01 = 1180.10).
        phase=1180 + 10 * 0.01,
        migrated=True,
        requires={"same_layer_as": "layer10_alu"},
        # Module-replacement sentinel: dynamic verifier (Mode B) skips
        # drift detection; static (Mode A) snapshot diffing unaffected.
        claims=set(),
        produces={
            '__module_replacement': 'L10.post_ops[PureFFN/bitwise_rules]',
        },
    )


def make_l10_alu_postop_attach_op(alu_mode: str = 'lookup') -> Operation:
    # V8 follow-up (``docs/LOOKUP_MODE_RULE_DERIVATION_2026_06_04.md``): the
    # legacy ``ALUAndOrXor`` install in lookup mode is replaced by a
    # rule-derived ``PureFFN`` baked from ``wide_alu_dsl.bitwise_rules``.
    # Byte-identity at the decoded OUTPUT byte is proven by
    # ``tests/test_wide_alu_dsl.py::test_bitwise_rules_byte_identity_*``.
    #
    # The replacement preserves the install schedule (phase=1180.10,
    # same target_op_name, same kind="block") so the dynamic scheduler and
    # the ``_expand_wrapper_blocks`` post-op split behave identically.
    if alu_mode == 'lookup':
        return _mark_structural_declarations(
            make_lookup_mode_l10_bitwise_rules_op()
        )
    # Dim-ownership claims: empty. ``bake`` inserts an
    # ``ALUAndOrXor`` into ``model.blocks[10].post_ops`` -- module
    # attach. Sentinel below documents the structural effect.
    return _annotate_module_replacement(
        _mark_structural_declarations(
            _make_alu_postop_attach_op(
                "l10_alu_postop_attach", 10, "ALUAndOrXor", alu_mode,
                same_layer_as="layer10_alu",
                # layer10_alu binds to ``layer10_carry_relay`` (kind=attn).
                target_op_name="layer10_carry_relay",
            )
        ),
        'L10.post_ops[ALUAndOrXor]',
    )


def make_l11_alu_postop_attach_op(alu_mode: str = 'lookup') -> Operation:
    # Dim-ownership claims: empty. ``bake`` inserts a fully-baked
    # ``FlattenedALUMul`` (= ``ALUMul``) into
    # ``model.blocks[11].post_ops`` -- module attach. Sentinel below
    # documents the structural effect.
    return _annotate_module_replacement(
        _mark_structural_declarations(
            _make_alu_postop_attach_op(
                "l11_alu_postop_attach", 11, "ALUMul", alu_mode,
                same_layer_as="layer11_mul_partial",
                # layer11_mul_partial binds to ``_layer11_ffn_dep_anchor``.
                target_op_name="_layer11_ffn_dep_anchor",
            )
        ),
        'L11.post_ops[FlattenedALUMul]',
    )


def make_l12_alu_postop_attach_op(alu_mode: str = 'lookup') -> Operation:
    # Dim-ownership claims: empty. ``bake`` inserts a fully-baked
    # ``FlattenedALUMul`` (= ``ALUMul``) into
    # ``model.blocks[12].post_ops`` -- module attach. Sentinel below
    # documents the structural effect.
    return _annotate_module_replacement(
        _mark_structural_declarations(
            _make_alu_postop_attach_op(
                "l12_alu_postop_attach", 12, "ALUMul", alu_mode,
                same_layer_as="layer12_mul_combine",
                # layer12_mul_combine binds to ``_layer12_ffn_dep_anchor``.
                target_op_name="_layer12_ffn_dep_anchor",
            )
        ),
        'L12.post_ops[FlattenedALUMul]',
    )


def make_l13_alu_postop_attach_op(alu_mode: str = 'lookup') -> Operation:
    # Dim-ownership claims: empty. ``bake`` inserts an
    # ``ALUShiftComposite`` (= ``ALUShift``) into
    # ``model.blocks[13].post_ops`` -- module attach. Sentinel below
    # documents the structural effect.
    return _annotate_module_replacement(
        _mark_structural_declarations(
            _make_alu_postop_attach_op(
                "l13_alu_postop_attach", 13, "ALUShift", alu_mode,
                same_layer_as="layer13_shifts",
                # layer13_shifts binds to ``_layer13_attn_dep_anchor``.
                target_op_name="_layer13_attn_dep_anchor",
            )
        ),
        'L13.post_ops[ALUShiftComposite]',
    )


# ---------------------------------------------------------------------------
# Efficient-mode ALU wrapper installs (migrated from set_vm_weights efficient
# branch, lines ~2208/2223/2240). Each replaces the original ``block.ffn`` with
# a wrapper module (HybridALUBlock for L8 ADD/SUB, ALUAndOrXor for L10 bitwise,
# ALUMul for L11 multiply). Only meaningful in efficient ALU mode; the factories
# return a no-op operation in lookup mode so callers can register them
# unconditionally without affecting lookup-mode bakes.
#
# Ordering notes:
#   - L8: ``_set_layer8_alu`` + ``_set_layer8_multibyte_routing`` still live
#     inline in legacy_bake (efficient branch) and mutate the original PureFFN.
#     They must run BEFORE the HybridALUBlock wrap, so the L8 wrap is a
#     ``kind="model"`` op at phase=1002 (after legacy_bake at phase=999).
#   - L10: nothing else mutates block.ffn after post_op_attach (phase=10.7)
#     reads its d_model. The wrap runs at phase=10.85 — after post_op_attach
#     (10.7) and the DIV/MOD install op (10.8), before L11 ops at 11.0.
#   - L11: ``layer11_mul_partial`` (phase=11) writes to ``block.ffn.W_up`` of
#     the original PureFFN, and the 9 FlattenedALUMul installer ops (phases
#     11.0..12.3) replace ``block.ffn`` with a flattened composite. Our wrap
#     runs at phase=11.05 — after both — and skips the install when
#     ``FlattenedALUMul`` is already present (the normal flow). This matches
#     the previous behavior where set_vm_weights' inline
#     ``model.blocks[11].ffn = ALUMul(...)`` was skipped when FlattenedALUMul
#     was already present.
# ---------------------------------------------------------------------------

def make_efficient_l8_addsub_wrap_op(alu_mode: str = 'lookup') -> Operation:
    """Wrap L8 ``block.ffn`` with ``HybridALUBlock(ffn, AddSub5StageBlock)``.

    Migrates the inline efficient-mode wrap at vm_step.py:
        ``model.blocks[8].ffn = HybridALUBlock(ffn8, AddSub5StageBlock(S, BD))``

    ``kind="model"`` at phase=1002 because the ``_set_layer8_alu`` and
    ``_set_layer8_multibyte_routing`` calls in legacy_bake (phase=999) require
    the original PureFFN with ``.W_up`` etc. The wrap must happen AFTER those
    calls populate the lookup FFN.
    """
    def bake(model, dim_positions, S):
        if alu_mode != 'efficient':
            return
        from ...efficient_alu_addsub_split import AddSub5StageBlock
        BD = _as_setdim_proxy(dim_positions)
        block = model.blocks[8]
        addsub = AddSub5StageBlock(S, BD)
        # Idempotent guard: if already attached, skip.
        if any(isinstance(po, AddSub5StageBlock) for po in block.post_ops):
            return
        # Attach as post_op; _expand_wrapper_blocks splits it downstream.
        block.post_ops.insert(0, addsub)

    return Operation(
        name="efficient_l8_addsub_wrap",
        requires={"after": ("initial_pc_bake",)},
        reads=set(),
        writes=set(),
        kind="model",
        declarative_bake_fn=bake,
        declarative_authority="structural_model",
        migrated=True,
        # Dim-ownership claims: empty. ``bake`` inserts an
        # ``AddSub5StageBlock`` into ``model.blocks[8].post_ops``
        # (module attach -- functionally equivalent to swapping
        # ``model.blocks[8].ffn``), not per-cell ``(layer, scope,
        # identifier, column)`` writes.
        claims=set(),
        # Module-replacement sentinel: dynamic verifier (Mode B) skips
        # drift detection; static (Mode A) snapshot diffing unaffected.
        produces={'__module_replacement': 'L8.post_ops[AddSub5StageBlock]'},
        smoke_tests={
            "TestSmoke32Bit::test_add_16bit",
            "TestSmoke32Bit::test_sub_16bit",
            "TestSmokeAddress::test_lea_basic",
            "TestSmokeBasic::test_add_basic",
            "TestSmokeBasic::test_sub_basic",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
        # Tier A opcode gating: the wrapped AddSub5StageBlock gates every
        # residual write on OP_ADD / OP_SUB (see ``_AddSubStage3`` opcode
        # merge in ``efficient_alu_addsub_split.py``).
        opcodes={"OP_ADD", "OP_SUB"},
    )


def make_efficient_l10_andorxor_wrap_op(alu_mode: str = 'lookup') -> Operation:
    """Replace L10 ``block.ffn`` with a rule-lowered bitwise FFN (= bitwise neural ALU).

    DSL Wave W1 (``docs/IR_DSL_DESIGN.md`` Section 5): the historical
    ``block.ffn = ALUAndOrXor(S, BD)`` install (= a multi-stage BD/GE
    composite from ``efficient_alu_neural.py``) is replaced by a
    rule-derived ``PureFFN`` baked from
    ``wide_alu_dsl.bitwise_rules(op=...)`` for AND/OR/XOR. Each opcode
    emits 512 rules (256 lo + 256 hi nibble cross-product); the three
    batches concatenate to 1,536 hidden units lowered in one
    ``Primitives.lower_ffn_rules`` call.

    Byte-identity contract: ``block.ffn.forward(x)`` decodes to the same
    OUTPUT byte as the legacy ``ALUAndOrXor.forward`` — validated by the
    POC ``tests/test_wide_alu_dsl.py`` (20/20 pass at scaffolding commit
    ``5e9ff9be``) and by ``TestSmokeBitwise`` in the full VM compile.

    ``kind="block"`` at phase=10.85 — runs after ``l10_post_op_attach``
    (phase=10.7, which inspects ``block.ffn.W_up`` to derive d_model) and the
    DIV/MOD install (phase=10.8); before any L11 ops at phase=11.0.
    """
    def bake(block, dim_positions, S):
        if alu_mode != 'efficient':
            return
        from ...base_layers import PureFFN
        from ..primitives import Primitives
        from ..wide_alu_dsl import bitwise_rules
        from ..building_blocks_dsl import step_function_rule

        # d_model comes from the pre-existing block.ffn so we stay
        # shape-stable whether or not the host is the production residual.
        ffn_in = block.ffn
        if hasattr(ffn_in, "W_up") and ffn_in.W_up is not None:
            d_model = int(ffn_in.W_up.shape[1])
        else:
            d_model = int(getattr(ffn_in, "dim", 512))

        bd_proxy = _as_setdim_proxy(dim_positions)

        def _lower(rule_tuple):
            ffn = PureFFN(dim=d_model, hidden_dim=len(rule_tuple))
            names = Primitives.ffn_rule_dim_names(rule_tuple)
            dim_pos = Primitives.dim_positions_from_bd(bd_proxy, names)
            end = Primitives.lower_ffn_rules(
                ffn, rule_tuple, dim_pos, start_unit=0, S=S,
            )
            assert end == len(rule_tuple), (
                f"lower_ffn_rules wrote {end} units; expected {len(rule_tuple)}"
            )
            return ffn

        # ============================================================
        # AND/OR/XOR MARK_AX compute: div-style operand-cleanup pattern
        # (2026-06-11). The IMM-decode keystone (5479fe4c / 383a6eb7)
        # delivered clean immediate VALUES, but the operand BANDS at the
        # MARK_AX row are still NOT clean one-hots -- the block-8 head-0
        # gather leaves constant additive artifacts (spec_k=0 probe
        # ``tools/probe_bitwise_operand_chars.py``):
        #
        #   ALU_LO/HI : real-nibble cell ~5.82, cell 0 ~+5.56,
        #               cell 8 ~+0.45, cell 15 ~+0.47 artifacts.
        #   AX_CARRY  : real-nibble cell ~0.94, cell 0 ~+0.30 artifact.
        #
        # The historical wrap fed these dirty bands straight into a fixed
        # (40,30,30)/thr-80 ``bitwise_rules`` lookup. With a real cell at
        # ~5.82, a single operand term (30*5.82 = 175) clears thr-80
        # alone, so the lookup MASSIVELY over-fires on every artifact
        # cell. For OR that lands on the right answer cell (0 OR b = b),
        # which is why ``or_basic`` passed; for AND the cell-0 artifact
        # firings all write ``a AND 0 = 0`` -> a dominant OUTPUT cell-0
        # spike that buries the real answer (and the L26 tail then
        # amplifies the wrong argmax). EQ/XOR share the path. See
        # ``docs/AND_MUL_MARK_AX_ENDRUN_2026_06_11.md``.
        #
        # FIX = the exact two-stage div pattern (``make_alu_divmod_
        # composite_ops`` GE-format install). The wrap historically did
        # ``block.ffn = bitwise_lookup`` (discarding L10's original FFN).
        # We preserve that "discard + own the AX-row bitwise compute"
        # contract but split into TWO cooperating modules:
        #   1. block.ffn := operand-cleanup FFN: subtracts the constant
        #      operand artifacts so each band becomes a clean one-hot of
        #      its real nibble (gated OP_AND/OP_OR/OP_XOR + MARK_AX).
        #   2. post_op := rescaled ``bitwise_rules`` lookup over the now-
        #      cleaned bands, writing OUTPUT_LO/HI at MARK_AX. The post_op
        #      expands to its own passthrough block AFTER block.ffn, so
        #      the lookup reads the cleaned operands. A clean single answer
        #      cell then dominates at the L14 materialise (block 15) and
        #      survives the L26 tail amplify -- exactly like ``or_basic``
        #      does today.
        #
        # Calibration (probe_bitwise_clean_calib.py, spec_k=0): a
        # MARK_AX-gated ``step_function_rule(write_value=V)`` changes the
        # target residual cell by ~``_RESID_PER_V * V``; measured
        # _RESID_PER_V ~= 2.502 (the same lowering as div's cleanup).
        _RESID_PER_V = 2.502
        _ALU_ARTIFACTS = ((0, 5.56), (8, 0.45), (15, 0.47))
        _CARRY_ARTIFACTS = ((0, 0.30),)
        OPCODES = ("OP_AND", "OP_OR", "OP_XOR")

        cleanup_rules: list = []
        for op_gate in OPCODES:
            for bnd, arts in (
                ("ALU_LO", _ALU_ARTIFACTS), ("ALU_HI", _ALU_ARTIFACTS),
                ("AX_CARRY_LO", _CARRY_ARTIFACTS),
                ("AX_CARRY_HI", _CARRY_ARTIFACTS),
            ):
                for cell, art in arts:
                    cleanup_rules.append(step_function_rule(
                        name=f"bitwise_operand_clean_{op_gate}_{bnd}_{cell}",
                        input_dim="MARK_AX",
                        threshold=0.5,
                        write_dim=f"{bnd}+{cell}",
                        write_value=-art / _RESID_PER_V,
                        gate=op_gate,
                        S=S,
                    ))
        # Wall-4 CMP/EQ engines (2026-06-12, RECONCILED).
        # ------------------------------------------------------------
        # In efficient mode this wrap REPLACES L10's block.ffn, so the
        # lookup-mode ``layer10_alu`` FFN -- which carries the declarative
        # comparison engines -- is discarded. The smoke gate runs efficient
        # mode (trust_neural_alu=True), so the engines must live here to
        # fire. They are MERGED into ``block.ffn`` (the cleanup FFN) rather
        # than appended as post_ops so they do NOT add extra passthrough
        # blocks (which would shift every downstream block index and break
        # absolute-position-dependent ops like lea). The cleanup units only
        # subtract artifacts for OP_AND/OP_OR/OP_XOR, so comparison rows
        # still carry the raw (dirty) bands the engines threshold against,
        # and the comparison units (gated on the cmp opcodes) never touch
        # the bitwise cleanup.
        #
        # THREE pieces, reconciled so the CMP flags are written EXACTLY ONCE:
        #
        # 1. ORDERING engine (``_layer10_alu_ordering_engine_rules``, 272
        #    units): the SOLE writer of CMP+0..3 for ALL six comparison
        #    opcodes (EQ/NE/LT/GT/LE/GE). Recomputes hi_lt/hi_eq/lo_eq/lo_lt
        #    from the raw AX-row operands (A in ALU_HI/LO, B in
        #    AX_CARRY_HI/LO), gated on the OR of the cmp opcode flags,
        #    feeding the live ComparisonCombine. Each flag lands ~1.1 at the
        #    decode row -- a single flag must NOT trip the 3-way override,
        #    two together must. This is the if_* ordering cluster fix.
        # 2. EQ engine (``_layer10_alu_eq_engine_rules``, 256 units): its
        #    CMP+1/CMP+2 flag writes were REMOVED (the ordering engine's
        #    hi_eq/lo_eq subsume them; double-writing would over-trip the
        #    EQ override). It now contributes ONLY the decode-margin push --
        #    the decisive 0x01 OUTPUT byte at the AX decode row for EQUAL
        #    operands -- DOWNSTREAM of the flags.
        # 3. EQ default (``_layer10_alu_eq_default_rules``, 1 unit): the
        #    L25-band fix -- one unconditional OP_EQ default-0 unit writes
        #    0x00 (smaller magnitude) so UNEQUAL operands (which fire no
        #    equal unit) decode 0x00 and survive the +238 L25 tail band that
        #    otherwise corrupts eq_false to 0x11=17.
        #
        # All three are gated on the cmp opcodes (ordering: all six; eq +
        # default: OP_EQ) so non-cmp opcodes are structurally untouched.
        # The decode-margin + default live ONLY in this efficient-mode merge
        # (the smoke path); lookup-mode keeps the same layout via
        # ``_layer10_alu_rules``.
        from .l10_ops import (
            _layer10_alu_eq_engine_rules,
            _layer10_alu_eq_default_rules,
            _layer10_alu_ordering_engine_rules,
        )
        eq_rules = _layer10_alu_eq_engine_rules(S)
        eq_default_rules = _layer10_alu_eq_default_rules(S)
        ordering_rules = _layer10_alu_ordering_engine_rules(S)
        cleanup_ffn = _lower(
            tuple(cleanup_rules)
            + tuple(eq_rules)
            + tuple(eq_default_rules)
            + tuple(ordering_rules)
        )

        # ---- Stage 2: rescaled bitwise lookup over the cleaned bands ----
        # Cleaned operand-A cells ~5.82, operand-B (AX_CARRY) cells ~0.94.
        # Rescale cond weights so a matched cell contributes ~30, keeping
        # the default-equivalent 40 + 30 + 30 = 100 > 80 threshold math.
        operand_a_cw = 30.0 / 5.82
        operand_b_cw = 30.0 / 0.94
        rule_list: list = []
        for op_name, opcode_gate in (
            ("and", "OP_AND"),
            ("or", "OP_OR"),
            ("xor", "OP_XOR"),
        ):
            rule_list.extend(bitwise_rules(
                op=op_name,
                operand_a_lo="ALU_LO",
                operand_a_hi="ALU_HI",
                operand_b_lo="AX_CARRY_LO",
                operand_b_hi="AX_CARRY_HI",
                result_lo="OUTPUT_LO",
                result_hi="OUTPUT_HI",
                opcode_gate=opcode_gate,
                marker_gate="MARK_AX",
                S=S,
                operand_a_cond_weight=operand_a_cw,
                operand_b_cond_weight=operand_b_cw,
                marker_cond_weight=40.0,
                threshold=80.0,
            ))
        rules = tuple(rule_list)
        assert len(rules) == 3 * 512, (
            f"bitwise_rules: expected 1536 rules (3 opcodes x 512), got {len(rules)}"
        )
        lookup_ffn = _lower(rules)

        # Stage 1 owns block.ffn (replacing L10's original FFN, exactly
        # as the historical wrap did); Stage 2 is a post_op that
        # ``_expand_wrapper_blocks`` splits into its own passthrough block
        # AFTER block.ffn -- so the lookup reads the cleaned operands.
        # Stage 1 (cleanup + EQ engine) owns block.ffn; Stage 2 (lookup)
        # is a post_op expanded into its own passthrough block AFTER
        # block.ffn so the bitwise lookup reads the cleaned operands.
        block.ffn = cleanup_ffn
        block.post_ops.append(lookup_ffn)

    return Operation(
        name="efficient_l10_andorxor_wrap",
        reads=set(),
        writes=set(),
        kind="block",
        declarative_bake_fn=bake,
        declarative_authority="structural_model",
        # Phase 8.G.6: drop ``layer_idx=10`` literal; bind to the L10
        # attn anchor ``layer10_carry_relay`` so the block op resolves
        # to whichever layer the compiler places the anchor at.
        target_op_name="layer10_carry_relay",
        requires={"after": "l10_alu_divmod_install"},
        migrated=True,
        # Dim-ownership claims: empty. ``bake`` replaces
        # ``model.blocks[10].ffn`` with a rule-derived ``PureFFN``
        # baked from ``bitwise_rules`` (AND/OR/XOR) -- module
        # replacement, not per-cell ``(layer, scope, identifier,
        # column)`` writes.
        claims=set(),
        # Module-replacement sentinel: dynamic verifier (Mode B) skips
        # drift detection; static (Mode A) snapshot diffing unaffected.
        produces={'__module_replacement': 'L10.ffn[PureFFN/bitwise_rules]'},
        smoke_tests={
            "TestSmoke32Bit::test_and_16bit",
            "TestSmoke32Bit::test_or_16bit",
            "TestSmoke32Bit::test_xor_16bit",
            "TestSmokeBitwise::test_and_basic",
            "TestSmokeBitwise::test_or_basic",
            "TestSmokeBitwise::test_xor_basic",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
        # Tier A opcode gating: ALUAndOrXor wraps ``PureNeuralALU(operations=
        # 'bitwise')``, which gates RESULT on OP_OR / OP_XOR / OP_AND (the
        # three bitwise opcodes built via ``build_or_layers(opcode=28)`` /
        # ``build_xor_layers(opcode=29)`` / ``build_and_layers(opcode=30)``).
        opcodes={"OP_AND", "OP_OR", "OP_XOR"},
    )


def make_efficient_l11_alumul_wrap_op(alu_mode: str = 'lookup') -> Operation:
    """Replace L11 ``block.ffn`` with a rule-lowered 8-bit MUL FFN.

    DSL Wave W5 (``docs/IR_DSL_DESIGN.md`` Section 5): the historical
    ``block.ffn = ALUMul(S, BD)`` install (and its 9-stage flattened
    sibling ``FlattenedALUMul``) is replaced by a rule-derived
    ``PureFFN`` baked from ``wide_alu_dsl.wide_mul_rules(width_bytes=1,
    ...)``. The POC emits 256 rules (one per (a_nib, b_nib) in
    ``0..15 × 0..15``); each rule writes the low nibble of the product
    to ``OUTPUT_LO+lo_nib`` and the high nibble to ``OUTPUT_LO+16+hi_nib``
    (= ``OUTPUT_HI+hi_nib`` since OUTPUT_HI = OUTPUT_LO + 16 in the
    ``_SetDim`` layout — see ``vm_step.py:2320``).

    Byte-identity contract (8-bit slice): for the nibble × nibble lookup
    over the byte 0 operands (``ALU_LO`` × ``AX_CARRY_LO``), the lowered
    FFN reproduces ``(a_nib * b_nib) & 0xFF`` byte-for-byte — validated
    by ``tests/test_wide_alu_dsl.py::test_wide_mul_rules_byte_identity_8bit``
    (256/256 pass). The ``TestSmokeBasic::test_mul_basic`` smoke
    (= 6 * 7 = 42) exercises exactly this lo-nibble path (both operands
    have ``ALU_HI = AX_CARRY_HI = 0``) and continues to pass with the
    rule-derived install.

    **Multi-byte deferral**: ``wide_mul_rules`` POC currently supports
    only ``width_bytes=1`` (= one nibble × one nibble); it raises
    ``NotImplementedError`` for ``width_bytes>1``. The full
    ``FlattenedALUMul`` 9-stage pipeline (BD->GE, schoolbook partial
    products, 3 carry-extraction passes, gen/prop, binary carry
    lookahead, final correction, GE->BD) handles arbitrary 16/32-bit
    operands with inter-byte carry propagation. Re-expressing that
    pipeline as ``FFNRule`` lists requires a follow-up wave per
    ``docs/IR_DSL_DESIGN.md`` Section 5. Until then the 16-bit and
    32-bit MUL smoke tests (``TestSmoke32Bit::test_mul_overflow``)
    remain in their pre-migration state.

    ``kind="block"`` — runs AFTER the full 9-stage assembly chain
    (``l12_alu_mul_getobd`` at phase=12.3) so any partially-constructed
    ``FlattenedALUMul`` is discarded cleanly. This mirrors W1, which
    similarly replaced the imperative composite at the end of the L10
    dep chain.
    """
    def bake(block, dim_positions, S):
        if alu_mode != 'efficient':
            return
        from ...base_layers import PureFFN
        from ..primitives import Primitives
        from ..wide_alu_dsl import wide_mul_rules

        # W5 POC: 256 rules covering nibble × nibble = 0..15 × 0..15.
        # Operand A's low nibble lives in ALU_LO band, operand B's in
        # AX_CARRY_LO band -- matching FlattenedALUMul's BD->GE
        # converter source dims (see efficient_alu_neural.py:1133-1138
        # and dim_layout.py:52-58). Result base = OUTPUT_LO; the high
        # nibble write at OUTPUT_LO+16+k lands in OUTPUT_HI+k since the
        # two bands are contiguous in _SetDim (174..189, 190..205).
        # Operand-magnitude-matched AND thresholds (2026-06-13 MUL root fix).
        # The L8 operand-gather emits a ~6-magnitude one-hot on ALU_LO
        # (operand A) and a ~0.9-magnitude one-hot on AX_CARRY_LO (operand B)
        # at the MARK_AX MUL row — NOT the 1.0/1.0 the default 30/30/40+thr80
        # AND-pattern assumes. With the defaults the ALU_LO term alone
        # (30 * 6 = 180) blows past threshold 80, so every (a=k, b=*) rule
        # fires and the OUTPUT product band fills with noise (decodes to 1).
        # Rescale: operand_a weight 5 (5 * 6 = 30), operand_b weight 30
        # (30 * 0.9 = 27), marker 40, threshold 80. All-on = 97 > 80; drop
        # b = 70 < 80; drop a = 67 < 80; drop marker = 57 < 80 — a clean
        # 3-way AND for the real operand magnitudes. (a=0 / b=0 nibble rules
        # write to OUTPUT_LO+0 only when the *zero* one-hot is hot, which the
        # gather does emit, so a*0 / 0*b still resolve to 0.) Verified
        # byte-identical lo-nibble product for 6*7=42 and 100*5=500 via
        # tools/probe_mul_full_fix_validate.py.
        rules = wide_mul_rules(
            operand_a_base="ALU_LO",
            operand_b_base="AX_CARRY_LO",
            result_base="OUTPUT_LO",
            width_bytes=1,
            opcode_gate="OP_MUL",
            marker_gate="MARK_AX",
            S=S,
            operand_a_cond_weight=5.0,
            operand_b_cond_weight=30.0,
            marker_cond_weight=40.0,
            threshold=80.0,
        )
        assert len(rules) == 256, (
            f"wide_mul_rules(width_bytes=1): expected 256 rules, "
            f"got {len(rules)}"
        )

        # Size the PureFFN to the rule count and lower in one pass.
        # d_model comes from whatever the prior bake left on block.ffn
        # (PureFFN, FlattenedALUMul, or a partial composite) so we stay
        # shape-stable. ``FlattenedALUMul`` has no top-level ``W_up`` (its
        # W_up lives inside the 7 sub-FFN stages), so fall back to the
        # block's attention ``dim`` which is the canonical residual width
        # (= 800 in the compiled VM, vs the legacy 512 default).
        ffn_in = block.ffn
        if hasattr(ffn_in, "W_up") and ffn_in.W_up is not None:
            d_model = int(ffn_in.W_up.shape[1])
        elif hasattr(block, "attn") and hasattr(block.attn, "dim"):
            d_model = int(block.attn.dim)
        else:
            d_model = int(getattr(ffn_in, "dim", 512))
        new_ffn = PureFFN(dim=d_model, hidden_dim=len(rules))

        bd_proxy = _as_setdim_proxy(dim_positions)
        names = Primitives.ffn_rule_dim_names(rules)
        dim_pos = Primitives.dim_positions_from_bd(bd_proxy, names)
        end = Primitives.lower_ffn_rules(
            new_ffn, rules, dim_pos, start_unit=0, S=S,
        )
        assert end == len(rules), (
            f"lower_ffn_rules wrote {end} units; expected {len(rules)}"
        )
        block.ffn = new_ffn

    return Operation(
        name="efficient_l11_alumul_wrap",
        # Phase 1 (memory cluster fix plan): the 10 L11/L12 mul stages and
        # this wrapper all co-bake ``block.ffn[FlattenedALUMul]`` on the
        # same L11 block via :func:`_ensure_l11_mul_module`. Module
        # assembly, not silent overwrites. See
        # docs/SLOT_REGISTRY_AUDIT_2026_06_05.md.
        slot_share=("ffn",),
        reads=set(),
        writes=set(),
        kind="block",
        declarative_bake_fn=bake,
        declarative_authority="structural_model",
        # 2026-06-13 MUL placement fix: RESTORE the explicit ``layer_idx=11``
        # pin (pre-expansion block 11 -> physical block 12 -> logical L11).
        # The Phase 8.G.6 ``target_op_name="_layer11_ffn_dep_anchor"`` binding
        # was MIS-RESOLVING: the dep-scheduler stacks the L10 op family
        # (passthrough / ax_broadcast / stack0_relay / carry_relay) across
        # pre-exp layers 9..14, which pushed ``_layer11_ffn_dep_anchor``
        # (``requires after: layer10_carry_relay``) to pre-exp layer 15
        # (= physical block 26 = logical L15). The 256-rule wide_mul FFN was
        # therefore baked onto L15, NOT L11 — so the MUL operands (present and
        # clean at the MARK_AX row through L11) were never multiplied, and the
        # L15-resident wide_mul wrote a noisy OUTPUT product band that the
        # L20 tail spike then amplified into a wrong decode (mul_basic -> 1).
        # This is the "L15 OUTPUT materialiser" the AND_MUL_MARK_AX_ENDRUN doc
        # described: it IS this misplaced wrap. Pinning ``layer_idx=11``
        # restores the intended L11 placement (validated:
        # tools/probe_mul_full_fix_validate.py -> 6*7=42, 100*5=500).
        # ``requires['after']`` is KEPT so the rule-derived install still
        # fires AFTER the 9-stage FlattenedALUMul installer chain
        # (``l12_alu_mul_getobd``, phase=12.3) and discards any
        # partially-assembled composite cleanly.
        layer_idx=11,
        requires={"after": "l12_alu_mul_getobd"},
        migrated=True,
        # Dim-ownership claims: empty. ``bake`` replaces
        # ``model.blocks[11].ffn`` with a rule-derived ``PureFFN``
        # baked from ``wide_mul_rules`` -- module replacement, not
        # per-cell ``(layer, scope, identifier, column)`` writes.
        claims=set(),
        # Module-replacement sentinel: dynamic verifier (Mode B) skips
        # drift detection; static (Mode A) snapshot diffing unaffected.
        produces={'__module_replacement': 'L11.ffn[PureFFN/wide_mul_rules]'},
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#binary-ALU",
        # Tier A opcode gating: ALUMul / FlattenedALUMul gate RESULT on
        # OP_MUL (see ``_MulCombineStage.forward`` in efficient_alu_neural.py
        # — ``op_mul = (x_ge_flat[:, 0, ge.OP_START + 27] > 0.1)``).
        opcodes={"OP_MUL"},
    )


# ---------------------------------------------------------------------------
# L11/L12 MUL ALU flattening (2026-05-10)
#
# The previous `set_vm_weights` line
#   model.blocks[11].ffn = ALUMul(S, BD)
# wrapped 9 logical sub-stages (BD→GE convert, schoolbook partial products,
# 3 carry-extraction passes, gen/prop, binary carry-lookahead, final
# correction, GE→BD convert) inside a single `PureNeuralALU(operations='mul')`
# runtime class. The 9 ops below split that wrapper into discrete compiler
# operations:
#
#   phase=11.0  install BD → GE converter         (FlattenedALUMul.bd_to_ge)
#   phase=11.1  append SchoolbookFFN              (mul_layers[0])
#   phase=11.2  append CarryPassFFN(pass_idx=0)   (mul_layers[1])
#   phase=11.3  append CarryPassFFN(pass_idx=1)   (mul_layers[2])
#   phase=11.4  append CarryPassFFN(pass_idx=2)   (mul_layers[3])
#   phase=12.0  append MulGenPropFFN              (mul_layers[4])
#   phase=12.1  append MulBinaryLookaheadFFN      (mul_layers[5])
#   phase=12.2  append MulFinalCorrectionFFN      (mul_layers[6])
#   phase=12.3  install GE → BD converter         (FlattenedALUMul.ge_to_bd)
#
# All nine are kind="block", layer_idx=11. Block ops are dispatched after
# per-layer ops and BEFORE legacy_bake (model op, phase=999), so the flattened
# module is fully assembled by the time legacy_bake's `set_vm_weights` runs
# (which no longer touches `model.blocks[11].ffn` for MUL).
#
# Forward is byte-identical to the previous ALUMul wrapper — see
# `FlattenedALUMul.forward` in efficient_alu_neural.py.
#
# fp32 only: NIBBLE config keeps dtype = fp32 throughout. CarryPassFFN's
# fp64-upcast guard (`S * max_value > 2**23`) does not trigger for NIBBLE
# (max S * max_value ≈ 100 * 1807 = 180700 << 8388608), so all 7 sub-FFNs
# build fp32 weights and run fp32 forward.
# ---------------------------------------------------------------------------


def make_l11_alu_mul_bdtoge_op() -> Operation:
    """phase=11.0: install BD → GE converter on the L11 flattened MUL FFN.

    First op in the chain — instantiates the FlattenedALUMul wrapper on
    ``block.ffn`` and bakes its `bd_to_ge` sub-FFN (which one-hot → scalar
    converts ALU_LO/HI and AX_CARRY_LO/HI into the GenericE NIB_A/NIB_B
    slots used by the schoolbook + carry pipeline).

    Equivalent to the construction of ``self.bd_to_ge`` inside
    ``PureNeuralALU.__init__(operations='mul')``.
    """
    def bake(block, dim_positions, S):
        module = _ensure_l11_mul_module(block, S, dim_positions=dim_positions)
        module.install_bdtoge()

    return Operation(
        name="l11_alu_mul_bdtoge",
        # Phase 1 (memory cluster fix plan): co-bakes L11 block.ffn[FlattenedALUMul]
        # with sibling mul stages. See docs/SLOT_REGISTRY_AUDIT_2026_06_05.md.
        slot_share=("ffn",),
        reads={"ALU_LO", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI", "OP_MUL"},
        writes=set(),
        kind="block",
        declarative_bake_fn=bake,
        declarative_authority="structural_model",
        # Phase 8.G.6: drop ``layer_idx=11`` literal; bind to the L11
        # ffn dep anchor so the block op resolves to whichever layer
        # the compiler places the anchor at.
        target_op_name="_layer11_ffn_dep_anchor",
        requires={"after": "_layer11_ffn_dep_anchor"},
        migrated=True,
        # Dim-ownership claims: empty. ``bake`` installs the
        # ``FlattenedALUMul`` BD->GE sub-FFN on ``model.blocks[11].ffn``
        # (module replacement), not per-cell ``(layer, scope, identifier,
        # column)`` writes. Sentinel below documents the structural
        # effect.
        claims=set(),
        # Module-replacement sentinel: dynamic verifier (Mode B) skips
        # drift detection; static (Mode A) snapshot diffing unaffected.
        produces={'__module_replacement': 'L11.ffn[FlattenedALUMul]'},
        smoke_tests={
            "TestSmoke32Bit::test_mul_overflow",
            "TestSmokeBasic::test_mul_basic",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
        # Tier A opcode gating: stage of FlattenedALUMul whose forward gates
        # all RESULT writes on OP_MUL (see ``_MulCombineStage`` in
        # ``efficient_alu_neural.py``).
        opcodes={"OP_MUL"},
    )


def make_l11_alu_mul_schoolbook_op() -> Operation:
    """phase=11.1: append the schoolbook partial-product FFN.

    Computes all N*(N+1)/2 partial products a[i]*b[j] for output position
    k=i+j, sums them into RESULT[k]. Equivalent to ``layers[0]`` from
    ``build_mul_layers(NIBBLE, opcode=27)``.
    """
    def bake(block, dim_positions, S):
        module = _ensure_l11_mul_module(block, S, dim_positions=dim_positions)
        module.install_schoolbook()

    return Operation(
        name="l11_alu_mul_schoolbook",
        # Phase 1 (memory cluster fix plan): co-bakes L11 block.ffn[FlattenedALUMul].
        slot_share=("ffn",),
        reads={"OP_MUL"},
        writes=set(),
        kind="block",
        declarative_bake_fn=bake,
        declarative_authority="structural_model",
        # Phase 8.G.6: drop ``layer_idx=11`` literal; bind to the L11
        # ffn dep anchor so the block op resolves to whichever layer
        # the compiler places the anchor at.
        target_op_name="_layer11_ffn_dep_anchor",
        requires={"after": "l11_alu_mul_bdtoge"},
        migrated=True,
        # Dim-ownership claims: empty. ``bake`` appends the schoolbook
        # partial-product sub-FFN to the ``FlattenedALUMul`` composite
        # on ``model.blocks[11].ffn`` (module assembly), not per-cell
        # ``(layer, scope, identifier, column)`` writes.
        claims=set(),
        # Module-replacement sentinel: dynamic verifier (Mode B) skips
        # drift detection; static (Mode A) snapshot diffing unaffected.
        produces={'__module_replacement': 'L11.ffn[FlattenedALUMul]'},
        smoke_tests={
            "TestSmoke32Bit::test_mul_overflow",
            "TestSmokeBasic::test_mul_basic",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
        # See l11_alu_mul_bdtoge: FlattenedALUMul stage.
        opcodes={"OP_MUL"},
    )


def make_l11_alu_mul_carrypass1_op() -> Operation:
    """phase=11.2: append CarryPassFFN(pass_idx=0).

    First carry-extraction pass; max_carry = 112 for NIBBLE schoolbook
    output (8 * 15 * 15 // 16 = 112). No incoming carry to add (pass_idx==0).
    Equivalent to ``layers[1]`` from ``build_mul_layers(NIBBLE, opcode=27)``.
    """
    def bake(block, dim_positions, S):
        module = _ensure_l11_mul_module(block, S, dim_positions=dim_positions)
        module.install_carrypass(pass_idx=0)

    return Operation(
        name="l11_alu_mul_carrypass1",
        # Phase 1 (memory cluster fix plan): co-bakes L11 block.ffn[FlattenedALUMul].
        slot_share=("ffn",),
        reads={"OP_MUL"},
        writes=set(),
        kind="block",
        declarative_bake_fn=bake,
        declarative_authority="structural_model",
        # Phase 8.G.6: drop ``layer_idx=11`` literal; bind to the L11
        # ffn dep anchor so the block op resolves to whichever layer
        # the compiler places the anchor at.
        target_op_name="_layer11_ffn_dep_anchor",
        requires={"after": "l11_alu_mul_schoolbook"},
        migrated=True,
        # Dim-ownership claims: empty. ``bake`` appends the carry-pass
        # sub-FFN to the ``FlattenedALUMul`` composite on
        # ``model.blocks[11].ffn`` (module assembly), not per-cell
        # writes.
        claims=set(),
        # Module-replacement sentinel: dynamic verifier (Mode B) skips
        # drift detection; static (Mode A) snapshot diffing unaffected.
        produces={'__module_replacement': 'L11.ffn[FlattenedALUMul]'},
        smoke_tests={
            "TestSmoke32Bit::test_mul_overflow",
            "TestSmokeBasic::test_mul_basic",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
        # See l11_alu_mul_bdtoge: FlattenedALUMul stage.
        opcodes={"OP_MUL"},
    )


def make_l11_alu_mul_carrypass2_op() -> Operation:
    """phase=11.3: append CarryPassFFN(pass_idx=1).

    Second carry-extraction pass; max_carry = 7 (= ((base-1) + 112) // base
    = (15 + 112) // 16). Adds incoming CARRY_OUT from pass 0 to RESULT
    before extracting new carry. Equivalent to ``layers[2]`` from
    ``build_mul_layers(NIBBLE, opcode=27)``.
    """
    def bake(block, dim_positions, S):
        module = _ensure_l11_mul_module(block, S, dim_positions=dim_positions)
        module.install_carrypass(pass_idx=1)

    return Operation(
        name="l11_alu_mul_carrypass2",
        # Phase 1 (memory cluster fix plan): co-bakes L11 block.ffn[FlattenedALUMul].
        slot_share=("ffn",),
        reads={"OP_MUL"},
        writes=set(),
        kind="block",
        declarative_bake_fn=bake,
        declarative_authority="structural_model",
        # Phase 8.G.6: drop ``layer_idx=11`` literal; bind to the L11
        # ffn dep anchor so the block op resolves to whichever layer
        # the compiler places the anchor at.
        target_op_name="_layer11_ffn_dep_anchor",
        requires={"after": "l11_alu_mul_carrypass1"},
        migrated=True,
        # Dim-ownership claims: empty. ``bake`` appends the carry-pass
        # sub-FFN to the ``FlattenedALUMul`` composite on
        # ``model.blocks[11].ffn`` (module assembly), not per-cell
        # writes.
        claims=set(),
        # Module-replacement sentinel: dynamic verifier (Mode B) skips
        # drift detection; static (Mode A) snapshot diffing unaffected.
        produces={'__module_replacement': 'L11.ffn[FlattenedALUMul]'},
        smoke_tests={
            "TestSmoke32Bit::test_mul_overflow",
            "TestSmokeBasic::test_mul_basic",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
        # See l11_alu_mul_bdtoge: FlattenedALUMul stage.
        opcodes={"OP_MUL"},
    )


def make_l11_alu_mul_carrypass3_op() -> Operation:
    """phase=11.4: append CarryPassFFN(pass_idx=2).

    Third carry-extraction pass; max_carry = 1 (= ((base-1) + 7) // base
    = (15 + 7) // 16). Final carry-extraction pass for NIBBLE; ensures
    incoming carry to GenProp is <= 1 so the binary lookahead correctness
    invariant holds. Equivalent to ``layers[3]`` from
    ``build_mul_layers(NIBBLE, opcode=27)``.
    """
    def bake(block, dim_positions, S):
        module = _ensure_l11_mul_module(block, S, dim_positions=dim_positions)
        module.install_carrypass(pass_idx=2)

    return Operation(
        name="l11_alu_mul_carrypass3",
        # Phase 1 (memory cluster fix plan): co-bakes L11 block.ffn[FlattenedALUMul].
        slot_share=("ffn",),
        reads={"OP_MUL"},
        writes=set(),
        kind="block",
        declarative_bake_fn=bake,
        declarative_authority="structural_model",
        # Phase 8.G.6: drop ``layer_idx=11`` literal; bind to the L11
        # ffn dep anchor so the block op resolves to whichever layer
        # the compiler places the anchor at.
        target_op_name="_layer11_ffn_dep_anchor",
        requires={"after": "l11_alu_mul_carrypass2"},
        migrated=True,
        # Dim-ownership claims: empty. ``bake`` appends the final
        # carry-pass sub-FFN to the ``FlattenedALUMul`` composite on
        # ``model.blocks[11].ffn`` (module assembly), not per-cell
        # writes.
        claims=set(),
        # Module-replacement sentinel: dynamic verifier (Mode B) skips
        # drift detection; static (Mode A) snapshot diffing unaffected.
        produces={'__module_replacement': 'L11.ffn[FlattenedALUMul]'},
        smoke_tests={
            "TestSmoke32Bit::test_mul_overflow",
            "TestSmokeBasic::test_mul_basic",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
        # See l11_alu_mul_bdtoge: FlattenedALUMul stage.
        opcodes={"OP_MUL"},
    )


def make_l12_alu_mul_genprop_op() -> Operation:
    """phase=12.0: append the gen/prop FFN.

    Adds incoming carry from the last carry pass, computes G[i] (RESULT
    + carry >= base → CARRY_OUT) and P[i] (RESULT + carry == base-1 →
    TEMP) for the binary carry chain, applies mod-base correction. Equivalent
    to ``layers[4]`` from ``build_mul_layers(NIBBLE, opcode=27)``.
    """
    def bake(block, dim_positions, S):
        module = _ensure_l11_mul_module(block, S, dim_positions=dim_positions)
        module.install_genprop()

    return Operation(
        name="l12_alu_mul_genprop",
        # Phase 1 (memory cluster fix plan): co-bakes L11 block.ffn[FlattenedALUMul]
        # (the "L12" label is historical — these stages target the same L11 block).
        slot_share=("ffn",),
        reads={"OP_MUL"},
        writes=set(),
        kind="block",
        declarative_bake_fn=bake,
        declarative_authority="structural_model",
        # Phase 8.G.6: drop ``layer_idx=11`` literal; bind to the L11
        # ffn dep anchor so the block op resolves to whichever layer
        # the compiler places the anchor at.
        target_op_name="_layer11_ffn_dep_anchor",
        requires={"after": "l11_alu_mul_carrypass3"},
        migrated=True,
        # Dim-ownership claims: empty. ``bake`` appends the gen/prop
        # sub-FFN to the ``FlattenedALUMul`` composite on
        # ``model.blocks[11].ffn`` (module assembly), not per-cell
        # writes.
        claims=set(),
        # Module-replacement sentinel: dynamic verifier (Mode B) skips
        # drift detection; static (Mode A) snapshot diffing unaffected.
        produces={'__module_replacement': 'L11.ffn[FlattenedALUMul]'},
        smoke_tests={
            "TestSmoke32Bit::test_mul_overflow",
            "TestSmokeBasic::test_mul_basic",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
        # See l11_alu_mul_bdtoge: FlattenedALUMul stage.
        opcodes={"OP_MUL"},
    )


def make_l12_alu_mul_binarylookahead_op() -> Operation:
    """phase=12.1: append the binary carry-lookahead FFN.

    Computes carries C[i] for i=1..N-1 from G/P pairs via N*(N-1)/2
    AND-gate hidden units, writes them into CARRY_IN, clears G (CARRY_OUT)
    and P (TEMP). Equivalent to ``layers[5]`` from
    ``build_mul_layers(NIBBLE, opcode=27)``.
    """
    def bake(block, dim_positions, S):
        module = _ensure_l11_mul_module(block, S, dim_positions=dim_positions)
        module.install_binarylookahead()

    return Operation(
        name="l12_alu_mul_binarylookahead",
        # Phase 1 (memory cluster fix plan): co-bakes L11 block.ffn[FlattenedALUMul].
        slot_share=("ffn",),
        reads={"OP_MUL"},
        writes=set(),
        kind="block",
        declarative_bake_fn=bake,
        declarative_authority="structural_model",
        # Phase 8.G.6: drop ``layer_idx=11`` literal; bind to the L11
        # ffn dep anchor so the block op resolves to whichever layer
        # the compiler places the anchor at.
        target_op_name="_layer11_ffn_dep_anchor",
        requires={"after": "l12_alu_mul_genprop"},
        migrated=True,
        # Dim-ownership claims: empty. ``bake`` appends the binary
        # carry-lookahead sub-FFN to the ``FlattenedALUMul`` composite
        # on ``model.blocks[11].ffn`` (module assembly), not per-cell
        # writes.
        claims=set(),
        # Module-replacement sentinel: dynamic verifier (Mode B) skips
        # drift detection; static (Mode A) snapshot diffing unaffected.
        produces={'__module_replacement': 'L11.ffn[FlattenedALUMul]'},
        smoke_tests={
            "TestSmoke32Bit::test_mul_overflow",
            "TestSmokeBasic::test_mul_basic",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
        # See l11_alu_mul_bdtoge: FlattenedALUMul stage.
        opcodes={"OP_MUL"},
    )


def make_l12_alu_mul_finalcorrection_op() -> Operation:
    """phase=12.2: append the final-correction FFN.

    Adds CARRY_IN from the lookahead to RESULT, applies mod-base
    correction (subtract base when sum >= base), clears CARRY_IN.
    Equivalent to ``layers[6]`` from ``build_mul_layers(NIBBLE, opcode=27)``.
    """
    def bake(block, dim_positions, S):
        module = _ensure_l11_mul_module(block, S, dim_positions=dim_positions)
        module.install_finalcorrection()

    return Operation(
        name="l12_alu_mul_finalcorrection",
        # Phase 1 (memory cluster fix plan): co-bakes L11 block.ffn[FlattenedALUMul].
        slot_share=("ffn",),
        reads={"OP_MUL"},
        writes=set(),
        kind="block",
        declarative_bake_fn=bake,
        declarative_authority="structural_model",
        # Phase 8.G.6: drop ``layer_idx=11`` literal; bind to the L11
        # ffn dep anchor so the block op resolves to whichever layer
        # the compiler places the anchor at.
        target_op_name="_layer11_ffn_dep_anchor",
        requires={"after": "l12_alu_mul_binarylookahead"},
        migrated=True,
        # Dim-ownership claims: empty. ``bake`` appends the final
        # correction sub-FFN to the ``FlattenedALUMul`` composite on
        # ``model.blocks[11].ffn`` (module assembly), not per-cell
        # writes.
        claims=set(),
        # Module-replacement sentinel: dynamic verifier (Mode B) skips
        # drift detection; static (Mode A) snapshot diffing unaffected.
        produces={'__module_replacement': 'L11.ffn[FlattenedALUMul]'},
        smoke_tests={
            "TestSmoke32Bit::test_mul_overflow",
            "TestSmokeBasic::test_mul_basic",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
        # See l11_alu_mul_bdtoge: FlattenedALUMul stage.
        opcodes={"OP_MUL"},
    )


def make_l12_alu_mul_getobd_op() -> Operation:
    """phase=12.3: install GE → BD converter on the L11 flattened MUL FFN.

    Final stage: convert RESULT scalar back to one-hot OUTPUT_LO/HI in BD
    format (using step-pair detection per nibble value), gated on OP_MUL
    AND on MARK_AX (so non-AX positions are untouched). Equivalent to the
    construction of ``self.ge_to_bd`` inside
    ``PureNeuralALU.__init__(operations='mul')``.
    """
    def bake(block, dim_positions, S):
        module = _ensure_l11_mul_module(block, S, dim_positions=dim_positions)
        module.install_getobd()

    return Operation(
        name="l12_alu_mul_getobd",
        # Phase 1 (memory cluster fix plan): co-bakes L11 block.ffn[FlattenedALUMul].
        slot_share=("ffn",),
        reads=set(),
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="block",
        declarative_bake_fn=bake,
        declarative_authority="structural_model",
        # Wave 5 (docs/PRODUCES_CONSUMES_MIGRATION.md): final GE->BD stage of
        # the L11/L12 MUL flattened composite writes the multiplication result
        # at the AX-marker row into OUTPUT_LO/HI -- AX_byte0 slot per the L14
        # POC convention. Stage takes no fresh residual reads (operates on
        # module-internal GE buffer); consumes_fresh stays empty.
        # Phase 11.A r3: dropped phase=12.3 — target_op_name +
        # requires['after']: l12_alu_mul_finalcorrection already pin order.
        # Phase 8.G.6: drop ``layer_idx=11`` literal; bind to the L11
        # ffn dep anchor so the block op resolves to whichever layer
        # the compiler places the anchor at.
        target_op_name="_layer11_ffn_dep_anchor",
        migrated=True,
        # Phase 7.A.5 default-flip: this MUL ALU stage runs after the
        # ``l12_alu_mul_finalcorrection`` stage (phase=12.2) on the same
        # ``_ensure_l11_mul_module`` builder. Without an explicit
        # ``requires["after"]`` the strict admission gate flags this op
        # as ``phase_required_but_undeclared`` because ``reads=set()``
        # leaves no in-edges. Anchoring it to its immediate phase
        # predecessor makes the chain dep-derived.
        requires={"after": [
            "l12_alu_mul_finalcorrection",
        ]},
        # Dim-ownership claims: empty. ``bake`` installs the final
        # GE->BD sub-FFN on the ``FlattenedALUMul`` composite that
        # owns ``model.blocks[11].ffn`` (module assembly), not per-cell
        # ``(layer, scope, identifier, column)`` writes. The ``writes``
        # set above documents the OUTPUT residual dims the assembled
        # composite touches at runtime.
        claims=set(),
        # Module-replacement sentinel: dynamic verifier (Mode B) skips
        # drift detection; static (Mode A) snapshot diffing unaffected.
        produces={'__module_replacement': 'L11.ffn[FlattenedALUMul]'},
        smoke_tests={
            "TestSmoke32Bit::test_mul_overflow",
            "TestSmokeBasic::test_mul_basic",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
        # See l11_alu_mul_bdtoge: FlattenedALUMul stage (the GE→BD writeback
        # whose ``opcode_mask`` zeros all OUTPUT writes outside OP_MUL).
        opcodes={"OP_MUL"},
    )


# L10 DIV/MOD ALU flattening (2026-05-10)
#
# The previous lookup-mode override
#   model.blocks[10].post_ops[-1] = EfficientDivMod_Neural(S, BD)
# in ``set_vm_weights`` and the efficient-mode append
#   block.post_ops.append(EfficientDivMod_Neural(S, _SetDim))
# in ``make_l10_post_op_attach_op`` both wrapped 3 logical sub-stages
# (BD→GE convert, long-division pipeline, GE→BD convert) inside a single
# ``PureNeuralALU(operations='div_mod')`` runtime class (alias
# ``ALUDivMod`` / ``EfficientDivMod_Neural``). The 4 ops below split that
# wrapper into discrete compiler operations:
#
#   phase=10.0  install BD → GE converter         (FlattenedDivMod.bd_to_ge)
#   phase=10.1  install long-division pipeline    (FlattenedDivMod.div_layers + mod_layers)
#                                                  = ClearDivSlotsFFN +
#                                                    LongDivisionModule +
#                                                    EmitDivResultModule per opcode
#   phase=10.2  install GE → BD converter         (FlattenedDivMod.ge_to_bd)
#   phase=10.8  install composite onto post_ops   (model.blocks[10].post_ops.append)
#
# The first 3 stage ops are kind="block", layer_idx=10. They run after
# `make_l10_post_op_attach_op` (phase=10.7) since 10.0/10.1/10.2 are < 10.7
# only in numeric-phase comparison — but since BLOCK ops sort by
# (layer_idx, phase), the smaller phases run FIRST. That's fine: the
# first 3 ops only construct sub-stages on a builder; nothing depends on
# `block.post_ops` until the install op (phase=10.8) actually inserts
# the composite.
#
# The install op (phase=10.8, kind="block", layer_idx=10) appends the
# fully-constructed FlattenedDivMod composite to ``block.post_ops``.
# It runs AFTER `make_l10_post_op_attach_op` (phase=10.7) which appends
# the standard L10 post_ops (BinaryOpByteZeroingPostOp etc.) but no longer
# appends EfficientDivMod_Neural / DivModModule.
#
# The legacy lookup-mode override in set_vm_weights
# (`model.blocks[10].post_ops[-1] = EfficientDivMod_Neural(S, BD)`) is
# also removed so the composite isn't clobbered.
#
# Forward is byte-identical to the previous EfficientDivMod_Neural — see
# ``FlattenedDivMod.forward`` in efficient_alu_divmod_split.py.
# ---------------------------------------------------------------------------


from .shared import _FlattenedDivModBuilder


def make_alu_divmod_composite_ops(alu_mode: str = 'lookup'):
    """Build the 4 cooperating ops (3 stage + 1 install) for FlattenedDivMod.

    Returns ``[bdtoge, longdiv, getobd, install]`` — all sharing the same
    ``_FlattenedDivModBuilder`` so the install op can append the
    fully-constructed composite to ``model.blocks[10].post_ops``.

    Stage ops are kind="block", layer_idx=10 (so they have access to the
    block when needed; their bake_fns operate on the shared builder
    rather than the block). Install op is kind="block", layer_idx=10.

    DSL Wave W6 (2026-06-11): when ``alu_mode == 'efficient'``, the 3
    stage ops (bdtoge/longdiv/getobd) become no-ops and the install op
    installs the BYTE-ACCURATE ``wide_alu_dsl.wide_div_rules_ge_format``
    lookup (DIV + MOD batches) as two cooperating ``PureFFN`` post_ops,
    in place of both the hand-written ``FlattenedDivMod`` composite and
    the earlier per-nibble ``wide_div_rules`` POC.

    Why the GE-format is now wired (the W4 blocker is resolved): the
    per-nibble ``wide_div_rules`` POC was mathematically wrong for
    cross-nibble dividends (84 = 0x54 / 2 -> 42 must divide 0x54 as a
    single integer, not nibble-by-nibble — see
    ``docs/DSL_W5_MULDIV_LIMIT.md``). ``wide_div_rules_ge_format`` is a
    flat 256x256 cross-product lookup (65,536 rules per opcode batch)
    that is byte-accurate by construction. It was blocked at the L10
    install point on operand cleanliness: the block-8 operand-gather
    leaves a constant ~5.56 magnitude artifact on cell 0 of ALU_LO/HI
    at the AX row (Wall-1's block-8 head-0 slope=0.1 fix cleaned the SE
    row, NOT the AX row this MARK_AX-gated install reads — see
    ``docs/DIV_GE_FORMAT_INSTALL_BLOCKER_2026_06_10.md`` and
    ``tools/probe_div_operand_clean.py``).

    The install resolves this with two staged post_ops (each
    ``post_ops`` entry expands to its own passthrough block):

      1. operand-cleanup FFN: subtracts the constant ~5.56 cell-0
         artifact from ALU_LO/HI (gated OP_DIV/OP_MOD + MARK_AX), making
         the dividend bands clean per-nibble one-hots. AX_CARRY (divisor)
         is already clean.
      2. GE-format lookup FFN: reads the cleaned operands, writes the
         byte-accurate quotient/remainder to OUTPUT_LO/HI at MARK_AX.
         The cleaned dividend cells sit at ~5.82 residual magnitude, so
         the per-cell dividend condition weight is rescaled to 30/5.82
         to preserve the 5-way-AND threshold math.

    DIV decodes from the MARK_AX row (the pre-Wave-B AX-row decode path,
    verified via ``tools/probe_div_decode_row.py``) — it is NOT
    SE-row-decode-blocked like the migrated CMP path (Wall-4).

    For the legacy ``alu_mode == 'lookup'`` path nothing changes —
    ``FlattenedDivMod`` is still appended as the post_op.
    """
    builder = _FlattenedDivModBuilder()

    def make_bdtoge():
        def bake(block, dim_positions, S):
            if alu_mode == 'efficient':
                # DSL W4: stage ops are no-ops; install op builds the
                # rule-derived PureFFN directly.
                return
            BD = _as_setdim_proxy(dim_positions)
            composite = builder.ensure(S, BD)
            composite.install_bdtoge()

        return Operation(
            name="l10_alu_divmod_bdtoge",
            # Phase 1 (memory cluster fix plan): co-bakes L10
            # block.post_ops[FlattenedDivMod] with sibling divmod stages.
            # See docs/SLOT_REGISTRY_AUDIT_2026_06_05.md.
            slot_share=("post_ops_append",),
            reads={"ALU_LO", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI",
                   "OP_DIV", "OP_MOD"},
            writes=set(),
            kind="block",
            declarative_bake_fn=bake,
            # Phase 8.G.6: drop ``layer_idx=10`` literal; bind to the L10
            # attn anchor ``layer10_carry_relay`` so the block op resolves
            # to whichever layer the compiler places the anchor at.
            target_op_name="layer10_carry_relay",
            migrated=True,
            declarative_authority="structural_model",
            # Dim-ownership claims: empty. ``bake`` attaches the BD->GE
            # sub-FFN to the shared ``_FlattenedDivModBuilder``; the
            # install op below ultimately appends the assembled
            # ``FlattenedDivMod`` to ``model.blocks[10].post_ops``
            # (module attach), not per-cell ``(layer, scope, identifier,
            # column)`` writes.
            claims=set(),
            # Module-replacement sentinel: dynamic verifier (Mode B) skips
            # drift detection; static (Mode A) snapshot diffing unaffected.
            produces={'__module_replacement': 'L10.post_ops[FlattenedDivMod]'},
        smoke_tests={
            "TestSmokeBasic::test_div_basic",
            "TestSmokeBasic::test_mod_basic",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
            # Tier A opcode gating: FlattenedDivMod stage. The composite's
            # forward gates RESULT on OP_DIV / OP_MOD (see
            # ``_DivModGEToBDStage`` in ``efficient_alu_divmod_split.py``:
            # ``op_div = (x_ge_flat[:, 0, ge.OP_START + 31] > 0.1)``,
            # ``op_mod = (x_ge_flat[:, 0, ge.OP_START + 32] > 0.1)``).
            opcodes={"OP_DIV", "OP_MOD"},
        )

    def make_longdiv():
        def bake(block, dim_positions, S):
            if alu_mode == 'efficient':
                return
            BD = _as_setdim_proxy(dim_positions)
            composite = builder.ensure(S, BD)
            composite.install_longdiv()

        return Operation(
            name="l10_alu_divmod_longdiv",
            # Phase 1 (memory cluster fix plan): co-bakes L10
            # block.post_ops[FlattenedDivMod].
            slot_share=("post_ops_append",),
            reads={"OP_DIV", "OP_MOD"},
            writes=set(),
            kind="block",
            declarative_bake_fn=bake,
            # Phase 8.G.6: drop ``layer_idx=10`` literal; bind to the L10
            # attn anchor ``layer10_carry_relay`` so the block op resolves
            # to whichever layer the compiler places the anchor at.
            target_op_name="layer10_carry_relay",
            migrated=True,
            declarative_authority="structural_model",
            # Dim-ownership claims: empty. ``bake`` attaches the
            # long-division pipeline to the shared
            # ``_FlattenedDivModBuilder``; module assembly, not
            # per-cell writes.
            claims=set(),
            # Module-replacement sentinel: dynamic verifier (Mode B) skips
            # drift detection; static (Mode A) snapshot diffing unaffected.
            produces={'__module_replacement': 'L10.post_ops[FlattenedDivMod]'},
        smoke_tests={
            "TestSmokeBasic::test_div_basic",
            "TestSmokeBasic::test_mod_basic",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
            # See l10_alu_divmod_bdtoge: FlattenedDivMod stage.
            opcodes={"OP_DIV", "OP_MOD"},
        )

    def make_getobd():
        def bake(block, dim_positions, S):
            if alu_mode == 'efficient':
                return
            BD = _as_setdim_proxy(dim_positions)
            composite = builder.ensure(S, BD)
            composite.install_getobd()

        return Operation(
            name="l10_alu_divmod_getobd",
            # Phase 1 (memory cluster fix plan): co-bakes L10
            # block.post_ops[FlattenedDivMod].
            slot_share=("post_ops_append",),
            reads={"OP_DIV", "OP_MOD", "MARK_AX"},
            writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
            kind="block",
            declarative_bake_fn=bake,
            # Phase 8.G.6: drop ``layer_idx=10`` literal; bind to the L10
            # attn anchor ``layer10_carry_relay`` so the block op resolves
            # to whichever layer the compiler places the anchor at.
            target_op_name="layer10_carry_relay",
            migrated=True,
            declarative_authority="structural_model",
            # Wave 5 (docs/PRODUCES_CONSUMES_MIGRATION.md): final GE->BD stage
            # of the L10 DIV/MOD flattened composite writes the quotient/
            # remainder at the AX-marker row into OUTPUT_LO/HI -- AX_byte0
            # slot per the L14 POC convention. Reads OP_DIV/OP_MOD/MARK_AX
            # are all cross-step durables (opcode/marker dims); consumes_fresh
            # stays empty.
            # Dim-ownership claims: empty. ``bake`` attaches the final
            # GE->BD sub-FFN to the shared
            # ``_FlattenedDivModBuilder``; module assembly, not
            # per-cell ``(layer, scope, identifier, column)`` writes.
            # The ``writes`` set above documents the OUTPUT residual
            # dims the assembled composite touches at runtime.
            claims=set(),
            # Module-replacement sentinel: dynamic verifier (Mode B) skips
            # drift detection; static (Mode A) snapshot diffing unaffected.
            produces={'__module_replacement': 'L10.post_ops[FlattenedDivMod]'},
        smoke_tests={
            "TestSmokeBasic::test_div_basic",
            "TestSmokeBasic::test_mod_basic",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
            # See l10_alu_divmod_bdtoge: FlattenedDivMod GE→BD writeback
            # stage; ``opcode_mask`` zeros OUTPUT writes outside OP_DIV/OP_MOD.
            opcodes={"OP_DIV", "OP_MOD"},
        )

    def make_install():
        def bake(block, dim_positions, S):
            if alu_mode == 'efficient':
                # DSL Wave W6 (2026-06-11): install the BYTE-ACCURATE
                # GE-format DIV/MOD lookup (``wide_div_rules_ge_format``)
                # in place of the per-nibble ``wide_div_rules`` POC. The
                # per-nibble lookup was mathematically wrong for
                # cross-nibble dividends (84 = 0x54 / 2 -> 42 needs 0x54
                # divided as a single integer, not nibble-by-nibble — see
                # ``docs/DSL_W5_MULDIV_LIMIT.md``).
                #
                # Two-post_op staging (each ``post_ops`` entry expands to
                # its own passthrough block via ``_expand_wrapper_blocks``):
                #
                #   1. operand-cleanup FFN: the block-8 operand-gather
                #      leaves a constant ~5.56 magnitude artifact on cell 0
                #      of ALU_LO/ALU_HI at the AX row (the dividend bands).
                #      Wall-1 (block-8 head-0 slope=0.1) cleaned the SE row
                #      but NOT the AX row, which is where this MARK_AX-gated
                #      install reads. The cleanup subtracts the constant
                #      artifact from cell 0 (gated OP_DIV/OP_MOD + MARK_AX)
                #      so the dividend bands become clean per-nibble
                #      one-hots. AX_CARRY (the divisor) is already clean.
                #      See ``docs/DIV_GE_FORMAT_INSTALL_BLOCKER_2026_06_10.md``
                #      and ``tools/probe_div_operand_clean.py``.
                #
                #   2. GE-format lookup FFN: 256x256 byte-accurate
                #      cross-product, reading the now-clean operands and
                #      writing quotient (DIV) / remainder (MOD) to
                #      OUTPUT_LO/HI at the MARK_AX row (the pre-Wave-B
                #      AX-row decode path — NOT the SE-row CMP/ALU Wall-4
                #      decode site, verified via
                #      ``tools/probe_div_decode_row.py``).
                #
                # Operand-magnitude rescale: the cleaned dividend cells are
                # ~5.82 in residual units (not the 1.0 binary one-hot the
                # default cond weights assume). We rescale the dividend
                # condition weight to 30/5.82 so a matched dividend cell
                # contributes ~30, restoring the 5-way-AND threshold math
                # (marker 40 + 4*30 = 160 > 150 fires; missing one -> 130).
                # The divisor cells (AX_CARRY) are already ~1.0, so keep
                # the default 30.0.
                from ...base_layers import PureFFN
                from ..primitives import Primitives
                from ..wide_alu_dsl import wide_div_rules_ge_format
                from ..building_blocks_dsl import step_function_rule

                # --- d_model from the existing block.ffn ---
                ffn_in = block.ffn
                if hasattr(ffn_in, "W_up") and ffn_in.W_up is not None:
                    d_model = int(ffn_in.W_up.shape[1])
                else:
                    d_model = int(getattr(ffn_in, "dim", 512))

                bd_proxy = _as_setdim_proxy(dim_positions)

                def _lower(rule_tuple):
                    ffn = PureFFN(dim=d_model, hidden_dim=len(rule_tuple))
                    names = Primitives.ffn_rule_dim_names(rule_tuple)
                    dim_pos = Primitives.dim_positions_from_bd(bd_proxy, names)
                    end = Primitives.lower_ffn_rules(
                        ffn, rule_tuple, dim_pos, start_unit=0, S=S,
                    )
                    assert end == len(rule_tuple), (
                        f"lower_ffn_rules wrote {end} units; "
                        f"expected {len(rule_tuple)}"
                    )
                    return ffn

                # ---------- Stage 1: dividend operand cleanup ----------
                # The artifact magnitude (~5.56 residual) and the residual-
                # per-write-weight factor (~2.502 for a MARK_AX-gated
                # threshold-0.5 write at this scale) were measured at the
                # install block (probe_div_operand_clean.py). V is the raw
                # write numerator that subtracts exactly the artifact:
                # 2.502 * V == 5.56  ->  V == 5.56 / 2.502.
                _ARTIFACT = 5.56
                _RESID_PER_V = 2.502
                cleanup_V = _ARTIFACT / _RESID_PER_V
                # Each rule is a MARK_AX-keyed, opcode-gated constant
                # write of ``-cleanup_V`` into cell 0 (step_function_rule
                # with the marker as the >=0.5 input). Two opcodes x two
                # dividend bands = 4 rules.
                cleanup_rules: list = []
                for op_gate in ("OP_DIV", "OP_MOD"):
                    for cell in ("ALU_LO+0", "ALU_HI+0"):
                        cleanup_rules.append(step_function_rule(
                            name=f"div_operand_clean_{op_gate}_{cell}",
                            input_dim="MARK_AX",
                            threshold=0.5,
                            write_dim=cell,
                            write_value=-cleanup_V,
                            gate=op_gate,
                            S=S,
                        ))
                cleanup_ffn = _lower(tuple(cleanup_rules))

                # ---------- Stage 2: GE-format byte-accurate lookup -------
                # Dividend cells ~5.82 -> rescale cond weight so a match
                # contributes ~30 (matching the helper's threshold math).
                dividend_cw = 30.0 / 5.82
                rule_list: list = []
                for op_gate, op_name in (("OP_DIV", "div"), ("OP_MOD", "mod")):
                    rule_list.extend(wide_div_rules_ge_format(
                        dividend_lo_base="ALU_LO",
                        dividend_hi_base="ALU_HI",
                        divisor_lo_base="AX_CARRY_LO",
                        divisor_hi_base="AX_CARRY_HI",
                        result_lo_base="OUTPUT_LO",
                        result_hi_base="OUTPUT_HI",
                        width_bytes=1,
                        opcode_gate=op_gate,
                        marker_gate="MARK_AX",
                        S=S,
                        op=op_name,
                        dividend_cond_weight=dividend_cw,
                        divisor_cond_weight=30.0,
                        marker_cond_weight=40.0,
                        threshold=150.0,
                    ))
                ge_rules = tuple(rule_list)
                # 2 opcodes x 256 x 256 = 131,072 rules.
                assert len(ge_rules) == 2 * 256 * 256, (
                    f"wide_div_rules_ge_format: expected 131072 rules "
                    f"(2 x 256 x 256), got {len(ge_rules)}"
                )
                ge_ffn = _lower(ge_rules)

                # Cleanup runs first (its expanded passthrough block lands
                # before the lookup block), so the lookup reads the
                # already-cleaned dividend bands.
                block.post_ops.append(cleanup_ffn)
                block.post_ops.append(ge_ffn)
                return

            if builder.composite is None:
                # No stage bakes ran (defensive). Skip cleanly.
                return
            block.post_ops.append(builder.composite)

        return Operation(
            name="l10_alu_divmod_install",
            # Phase 1 (memory cluster fix plan): co-bakes L10
            # block.post_ops[FlattenedDivMod] (the install step that appends
            # the assembled composite). See
            # docs/SLOT_REGISTRY_AUDIT_2026_06_05.md.
            slot_share=("post_ops_append",),
            # Phase 11.A r3: dropped phase=10.8 — target_op_name +
            # requires['after']: l10_alu_divmod_getobd already pin order.
            reads=set(),
            writes=set(),
            kind="block",
            declarative_bake_fn=bake,
            # Phase 8.G.6: drop ``layer_idx=10`` literal; bind to the L10
            # attn anchor ``layer10_carry_relay`` so the block op resolves
            # to whichever layer the compiler places the anchor at.
            target_op_name="layer10_carry_relay",
            requires={"after": "l10_alu_divmod_getobd"},
            migrated=True,
            declarative_authority="structural_model",
            # Dim-ownership claims: empty. ``bake`` appends the
            # assembled ``FlattenedDivMod`` composite to
            # ``model.blocks[10].post_ops`` (or, in efficient mode,
            # installs a rule-derived ``PureFFN`` post_op from
            # ``wide_div_rules``) -- module attach, not per-cell
            # ``(layer, scope, identifier, column)`` writes.
            claims=set(),
            # Module-replacement sentinel: dynamic verifier (Mode B) skips
            # drift detection; static (Mode A) snapshot diffing unaffected.
            produces={'__module_replacement': 'L10.post_ops[FlattenedDivMod]'},
        smoke_tests={
            "TestSmokeBasic::test_div_basic",
            "TestSmokeBasic::test_mod_basic",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
            # See l10_alu_divmod_bdtoge: FlattenedDivMod's forward gates
            # all OUTPUT writes on OP_DIV / OP_MOD via opcode_mask.
            opcodes={"OP_DIV", "OP_MOD"},
        )

    return [make_bdtoge(), make_longdiv(), make_getobd(), make_install()]


# Single-op factory shims for callers that want one op (e.g. unit tests).
# Each returns a fresh builder so the ops aren't entangled across factories.
def make_l10_alu_divmod_bdtoge_op(alu_mode: str = 'lookup') -> Operation:
    """L10 stage 1: BD → GE format conversion (standalone factory)."""
    return make_alu_divmod_composite_ops(alu_mode=alu_mode)[0]


def make_l10_alu_divmod_longdiv_op(alu_mode: str = 'lookup') -> Operation:
    """L10 stage 2: long-division pipeline (standalone factory)."""
    return make_alu_divmod_composite_ops(alu_mode=alu_mode)[1]


def make_l10_alu_divmod_getobd_op(alu_mode: str = 'lookup') -> Operation:
    """L10 stage 3: GE → BD format conversion (standalone factory)."""
    return make_alu_divmod_composite_ops(alu_mode=alu_mode)[2]


def make_l10_alu_divmod_install_op(alu_mode: str = 'lookup') -> Operation:
    """L10 install op: append composite to block.post_ops (standalone factory)."""
    return make_alu_divmod_composite_ops(alu_mode=alu_mode)[3]


def make_layer10_divmod_op() -> Operation:
    """Declarative wrapper op for the L10 FlattenedDivMod composite (Bug #36).

    Bug #36 / ``LONG_DIVISION_BUG36_2026_06_09.md`` flagged that DIV/MOD
    semantics live entirely in an imperative ``nn.Module``
    (``FlattenedDivMod``) appended at L10 phase=10.0-10.2, with
    multi-byte DIV having no rule-set path (see
    ``wide_alu_dsl.wide_div_rules`` -- ``width_bytes>1`` raises
    ``NotImplementedError`` per ``DSL_W5_MULDIV_LIMIT.md``).

    Full declarative migration is structurally infeasible: long
    division requires conditional subtraction per nibble (8 outer x 15
    inner trial-multiply / compare / subtract steps), and the partial
    dividend lives in a 9-nibble GE workspace accumulator that does
    not project onto the BD residual stream without 24+ extra layers
    (see ``LongDivisionModule`` docstring in
    ``alu/ops/divmod_longdiv.py``).

    This wrapper is the alternative deliverable from the Bug #36
    brief: a no-op declarative ``Operation`` whose ``reads`` /
    ``writes`` enumerate every BD-format dim the imperative composite
    actually touches at runtime, so:

      * ``dim_contracts_audit`` and
        ``decl_verifier.verify_rule_scopes`` see the divmod composite
        as a single declarative producer/consumer rather than only as
        4 stage ops carrying ``produces={'__module_replacement': ...}``
        sentinels with individually-sliced reads/writes (the existing
        ``l10_alu_divmod_{bdtoge,longdiv,getobd,install}`` ops).
      * Future producer/consumer ``DimContract`` registrations for
        ``AX_FULL_LO`` / ``AX_FULL_HI`` (the byte-1 stack staging
        identified as the leak source for the ``div_5`` failure
        shape) can name a single op rather than tracking the 4-stage
        composite boundary.

    The ``bake_fn`` is a deliberate no-op -- the actual composite
    install happens via ``make_alu_divmod_composite_ops`` (the 4-stage
    builder). This op MUST run alongside that composite (it does not
    replace it) and is tagged with
    ``produces={'__declarations_only': ...}`` so dynamic verifiers
    (Mode B) skip drift detection and static (Mode A) snapshot diffing
    is unaffected.

    Read enumeration (cross-referenced with
    ``efficient_alu_neural.BDToGEConverter.forward``):

      * Operand bytes:       ``ALU_LO``, ``ALU_HI``, ``AX_CARRY_LO``, ``AX_CARRY_HI``
      * Wide-operand byte-1: ``AX_FULL_LO``, ``AX_FULL_HI``
      * Stack0 fallback:     ``STACK0_BYTE1``, ``CLEAN_EMBED_LO``, ``CLEAN_EMBED_HI``
      * Opcode gates:        ``OP_DIV``, ``OP_MOD``, ``OP_MUL``, ``OP_SHL``, ``OP_SHR``
      * Position gate:       ``MARK_AX``

    Writes (from ``GEToBDConverter`` in the GE->BD writeback stage):

      * Quotient/remainder:  ``OUTPUT_LO``, ``OUTPUT_HI_THIS_STEP``
    """
    def _bake(target, dim_positions, S):
        # Deliberate no-op: the actual install is performed by the
        # 4-stage builder in ``make_alu_divmod_composite_ops``. This
        # op exists solely to declare the consolidated reads/writes
        # set for ``dim_contracts_audit`` / ``verify_rule_scopes``
        # coverage of the FlattenedDivMod composite.
        pass

    return Operation(
        name="layer10_divmod",
        # Co-bake alongside the 4-stage composite ops; no ordering
        # constraint since this op is declarations-only and has no
        # bake side effect.
        slot_share=("post_ops_append",),
        reads={
            # Operand bytes (low/high) read by BDToGEConverter.
            "ALU_LO", "ALU_HI",
            "AX_CARRY_LO", "AX_CARRY_HI",
            # Wide-operand byte-1 staging for 16/32-bit DIV/MOD.
            "AX_FULL_LO", "AX_FULL_HI",
            # STACK0 byte-1 fallback path (cummax over prefix).
            "STACK0_BYTE1",
            "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
            # Opcode gates (DIV/MOD plus the wide_op union used by
            # the AX_FULL branch in BDToGEConverter).
            "OP_DIV", "OP_MOD",
            "OP_MUL", "OP_SHL", "OP_SHR",
            # AX marker (position gate for the long-division step).
            "MARK_AX",
        },
        writes={
            # Quotient (DIV) or remainder (MOD) written into OUTPUT
            # bands at the MARK_AX position by GEToBDConverter.
            "OUTPUT_LO", "OUTPUT_HI_THIS_STEP",
        },
        kind="block",
        declarative_bake_fn=_bake,
        target_op_name="layer10_carry_relay",
        migrated=True,
        declarative_authority="structural_model",
        # No per-cell claims -- the 4 stage ops own the module-attach
        # claim; this wrapper only declares the I/O interface.
        claims=set(),
        # Declarations-only sentinel: dynamic verifier (Mode B) skips
        # drift detection (no weights are written); static (Mode A)
        # snapshot diffing is unaffected.
        produces={'__declarations_only': 'L10.post_ops[FlattenedDivMod]'},
        smoke_tests={
            "TestSmokeBasic::test_div_basic",
            "TestSmokeBasic::test_mod_basic",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
        opcodes={"OP_DIV", "OP_MOD"},
    )


def make_layer10_residual_alibi_slopes_op(alu_mode: str = 'lookup') -> Operation:
    """Bake the residual L10 ALiBi-slope mutations previously inline in set_vm_weights.

    Mode-conditional: in lookup mode, head 0..4 slopes are written
    (carry relay + 3 byte-passthrough heads + STACK0 byte relay for
    bitwise). In efficient mode, head 0..3 slopes are written (no
    STACK0 byte relay — that was lookup-only).

    Phase=999.1 places this after ``residual_alibi_slopes`` (phase 999)
    so it runs in the same "post-block-ops, pre-post-passes" window the
    legacy bake occupied.
    """
    def _bake(model, dim_positions, S):
        if len(model.blocks) <= 10:
            return
        attn10 = model.blocks[10].attn
        if not (hasattr(attn10, 'alibi_slopes') and attn10.alibi_slopes is not None):
            return
        attn10.alibi_slopes[0] = 5.0  # head 0: steep slope for carry relay
        attn10.alibi_slopes[1] = 1.0  # head 1: AX byte passthrough
        attn10.alibi_slopes[2] = 1.0  # head 2: SP byte passthrough
        attn10.alibi_slopes[3] = 0.5  # head 3: PSH STACK0 passthrough
        # Cluster C (2026-06-03): head 4 spec bakes unconditionally (the
        # bitwise stack-byte relay also feeds SUB byte 1+ in efficient mode
        # via the shared head). Gating the slope on alu_mode='lookup' left
        # efficient mode with slope[4]=0, so the head's attention was
        # muted and SUB stack byte 1 read as zero -> UINT32 wrap. Ungate
        # so the slope matches the (already unconditional) weight bake.
        attn10.alibi_slopes[4] = 1.0  # head 4: STACK0 byte relay for bitwise

    return Operation(
        name="layer10_residual_alibi_slopes",
        requires={"after": ("residual_alibi_slopes",)},
        reads=set(),
        writes=set(),
        kind="model",
        declarative_bake_fn=_bake,
        migrated=True,
        declarative_authority="structural_model",
        smoke_tests=set(),
        spec_section="BLOG_SPEC.md#the-attention-layer",
    )


# ---------------------------------------------------------------------------
# L8 ALU ADD/SUB flatten: 5 spec-stage anchors replacing the monolithic
# ALUAddSub wrapper
# ---------------------------------------------------------------------------
#
# These ops correspond 1:1 to the 5 stages of the flattened ADD/SUB pipeline
# (see efficient_alu_addsub_split.py). Their bake_fns are no-ops because the
# stage modules are installed by `set_vm_weights` via `AddSub5StageBlock` and
# split into 5 successive blocks by `_expand_wrapper_blocks`. The ops exist
# to:
#
#   1. Document the data flow / dependency graph for the compiler
#   2. Reserve their phase slots (8.0..8.4) so the compiler can place
#      downstream ops correctly
#   3. Provide hooks for future migration to true bake-FFN-weights ops
#
# The runtime model after these ops contains NO `ALUAddSub` instance. The
# compiler-owned L8 ALU attach op installs `AddSub5StageBlock`, whose
# `nn.Sequential` pipeline is generated from these five stages and validated
# byte-for-byte against `PureNeuralALU(operations="add_sub")`.

_L8_ADDSUB_SMOKE_TESTS = {
    "TestSmoke32Bit::test_add_16bit",
    "TestSmoke32Bit::test_sub_16bit",
    "TestSmokeAddress::test_lea_basic",
    "TestSmokeBasic::test_add_basic",
    "TestSmokeBasic::test_sub_basic",
}

def make_l8_alu_addsub_bdtoge_op() -> Operation:
    """Stage 0: BD -> GE format projection (BDToGEConverter equivalent).

    Reads BD-format ALU operand nibbles (ALU_LO/HI, AX_CARRY_LO/HI) and
    opcode flags. Writes the GE-format intermediate state (consumed by
    stage 1). Phase=8.0.
    """
    def bake(target, dim_positions, S):
        # Stage modules are installed by set_vm_weights via AddSub5StageBlock,
        # then split by _expand_wrapper_blocks. No bake-time work here.
        pass

    return Operation(
        name="l8_alu_addsub_bdtoge",
        reads=set(),
        writes=set(),
        kind="model",  # no-op model op; documentation only
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        # Phase 11.A: empty CompilerIR (bake_fn is a no-op; AddSub5StageBlock
        # installed imperatively by set_vm_weights, validated against PureNeuralALU).
        compiler_ir=CompilerIR(),
        # Dim-ownership claims: empty. ``bake`` is a documentation-only
        # no-op; the actual structural effect (swapping
        # ``model.blocks[8].ffn`` for ``AddSub5StageBlock``) is performed
        # by ``set_vm_weights`` / ``_expand_wrapper_blocks`` -- module
        # replacement, not per-cell ``(layer, scope, identifier, column)``
        # writes. Sentinel below documents the structural effect.
        claims=set(),
        # Module-replacement sentinel: dynamic verifier (Mode B) skips
        # drift detection; static (Mode A) snapshot diffing unaffected.
        produces={'__module_replacement': 'L8.ffn[AddSub5StageBlock]'},
        smoke_tests=_L8_ADDSUB_SMOKE_TESTS,
        spec_section="BLOG_SPEC.md#binary-ALU",
    )


def make_l8_alu_addsub_stage1_op() -> Operation:
    """Stage 1: AddRawAndGenFFN + SubRawAndGenFFN. Phase=8.1.

    Computes RAW_SUM, CARRY_OUT, TEMP for both ADD and SUB pipelines on
    the GE-format buffer. No BD reads/writes (operates on side-channel).
    """
    def bake(target, dim_positions, S):
        pass

    return Operation(
        name="l8_alu_addsub_stage1",
        requires={"after": ("l8_alu_addsub_bdtoge",)},
        reads=set(),
        writes=set(),
        kind="model",
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        # Phase 11.A: empty CompilerIR (bake_fn is a no-op; AddSub5StageBlock
        # installed imperatively by set_vm_weights, validated against PureNeuralALU).
        compiler_ir=CompilerIR(),
        # Dim-ownership claims: empty. ``bake`` is a documentation-only
        # no-op; module replacement performed by ``set_vm_weights`` /
        # ``_expand_wrapper_blocks``.
        claims=set(),
        # Module-replacement sentinel: dynamic verifier (Mode B) skips
        # drift detection; static (Mode A) snapshot diffing unaffected.
        produces={'__module_replacement': 'L8.ffn[AddSub5StageBlock]'},
        smoke_tests=_L8_ADDSUB_SMOKE_TESTS,
        spec_section="BLOG_SPEC.md#binary-ALU",
    )


def make_l8_alu_addsub_stage2_op() -> Operation:
    """Stage 2: AddCarryLookaheadFFN + SubBorrowLookaheadFFN. Phase=8.2.

    Cross-position carry/borrow propagation. For NIBBLE config (N=8 not 1),
    this clears CARRY_OUT and TEMP; carry-lookahead structure runs but is
    a no-op when N positions are independent (each position is its own
    chunk in the byte-level pipeline).
    """
    def bake(target, dim_positions, S):
        pass

    return Operation(
        name="l8_alu_addsub_stage2",
        requires={"after": ("l8_alu_addsub_stage1",)},
        reads=set(),
        writes=set(),
        kind="model",
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        # Phase 11.A: empty CompilerIR (bake_fn is a no-op; AddSub5StageBlock
        # installed imperatively by set_vm_weights, validated against PureNeuralALU).
        compiler_ir=CompilerIR(),
        # Dim-ownership claims: empty. ``bake`` is a documentation-only
        # no-op; module replacement performed by ``set_vm_weights`` /
        # ``_expand_wrapper_blocks``.
        claims=set(),
        # Module-replacement sentinel: dynamic verifier (Mode B) skips
        # drift detection; static (Mode A) snapshot diffing unaffected.
        produces={'__module_replacement': 'L8.ffn[AddSub5StageBlock]'},
        smoke_tests=_L8_ADDSUB_SMOKE_TESTS,
        spec_section="BLOG_SPEC.md#binary-ALU",
    )


def make_l8_alu_addsub_stage3_op() -> Operation:
    """Stage 3: AddFinalResultFFN + SubFinalResultFFN + opcode merge. Phase=8.3.

    Produces RESULT = (RAW_SUM +/- CARRY) mod base for both pipelines, then
    merges them via opcode mask: RESULT = ADD_RESULT*op_add + SUB_RESULT*op_sub.
    """
    def bake(target, dim_positions, S):
        pass

    return Operation(
        name="l8_alu_addsub_stage3",
        requires={"after": ("l8_alu_addsub_stage2",)},
        reads=set(),
        writes=set(),
        kind="model",
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        # Phase 11.A: empty CompilerIR (bake_fn is a no-op; AddSub5StageBlock
        # installed imperatively by set_vm_weights, validated against PureNeuralALU).
        compiler_ir=CompilerIR(),
        # Dim-ownership claims: empty. ``bake`` is a documentation-only
        # no-op; module replacement performed by ``set_vm_weights`` /
        # ``_expand_wrapper_blocks``.
        claims=set(),
        # Module-replacement sentinel: dynamic verifier (Mode B) skips
        # drift detection; static (Mode A) snapshot diffing unaffected.
        produces={'__module_replacement': 'L8.ffn[AddSub5StageBlock]'},
        smoke_tests=_L8_ADDSUB_SMOKE_TESTS,
        spec_section="BLOG_SPEC.md#binary-ALU",
    )


def make_l8_alu_addsub_getobd_op() -> Operation:
    """Stage 4: GE -> BD writeback (GEToBDConverter equivalent). Phase=8.4.

    Reads RESULT from the merged GE buffer, applies AX-marker + opcode
    masking, and writes OUTPUT_LO/HI + CARRY[1]/CARRY[2] to BD.
    """
    def bake(target, dim_positions, S):
        pass

    return Operation(
        name="l8_alu_addsub_getobd",
        requires={"after": ("l8_alu_addsub_stage3",)},
        reads=set(),
        writes=set(),
        kind="model",
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        # Phase 11.A: empty CompilerIR (bake_fn is a no-op; AddSub5StageBlock
        # installed imperatively by set_vm_weights, validated against PureNeuralALU).
        compiler_ir=CompilerIR(),
        # Dim-ownership claims: empty. ``bake`` is a documentation-only
        # no-op; module replacement performed by ``set_vm_weights`` /
        # ``_expand_wrapper_blocks``.
        claims=set(),
        # Module-replacement sentinel: dynamic verifier (Mode B) skips
        # drift detection; static (Mode A) snapshot diffing unaffected.
        produces={'__module_replacement': 'L8.ffn[AddSub5StageBlock]'},
        smoke_tests=_L8_ADDSUB_SMOKE_TESTS,
        spec_section="BLOG_SPEC.md#binary-ALU",
    )
