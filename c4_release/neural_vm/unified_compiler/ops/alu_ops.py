"""ALU composite/wrapping op factories. See ../migrated_ops.py for history."""

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
            phase=13,
            reads={"MARK_AX", "ALU_LO", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI",
                   "OP_SHL", "OP_SHR"},
            writes=set(),
            kind="ffn",
            bake_fn=bake,
            migrated=True,
        smoke_tests={
            "TestSmoke32Bit::test_shl_8bit",
            "TestSmoke32Bit::test_shr_8bit",
            "TestSmokeShift::test_shl",
            "TestSmokeShift::test_shr",
        },
        spec_section="BLOG_SPEC.md#shifts",
        )

    def make_precompute():
        def bake(ffn, dim_positions, S):
            from ...efficient_alu_neural import ShiftPrecomputeStage
            BD = _as_setdim_proxy(dim_positions)
            composite = builder.ensure(S, BD)
            composite.precompute_stage = ShiftPrecomputeStage(S, BD)

        return Operation(
            name="l13_alu_shift_precompute",
            phase=13,
            reads={"MARK_AX", "ALU_LO", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI",
                   "OP_SHL", "OP_SHR"},
            writes=set(),
            kind="ffn",
            bake_fn=bake,
            migrated=True,
        smoke_tests={
            "TestSmoke32Bit::test_shl_8bit",
            "TestSmoke32Bit::test_shr_8bit",
            "TestSmokeShift::test_shl",
            "TestSmokeShift::test_shr",
        },
        spec_section="BLOG_SPEC.md#shifts",
        )

    def make_select():
        def bake(ffn, dim_positions, S):
            from ...efficient_alu_neural import ShiftSelectStage
            BD = _as_setdim_proxy(dim_positions)
            composite = builder.ensure(S, BD)
            composite.select_stage = ShiftSelectStage(S, BD)

        return Operation(
            name="l13_alu_shift_select",
            phase=13,
            reads={"MARK_AX", "ALU_LO", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI",
                   "OP_SHL", "OP_SHR"},
            writes=set(),
            kind="ffn",
            bake_fn=bake,
            migrated=True,
        smoke_tests={
            "TestSmoke32Bit::test_shl_8bit",
            "TestSmoke32Bit::test_shr_8bit",
            "TestSmokeShift::test_shl",
            "TestSmokeShift::test_shr",
        },
        spec_section="BLOG_SPEC.md#shifts",
        )

    def make_getobd():
        def bake(ffn, dim_positions, S):
            from ...efficient_alu_neural import ShiftGEToBDStage
            BD = _as_setdim_proxy(dim_positions)
            composite = builder.ensure(S, BD)
            composite.getobd_stage = ShiftGEToBDStage(S, BD)

        return Operation(
            name="l13_alu_shift_getobd",
            phase=13,
            reads={"MARK_AX", "ALU_LO", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI",
                   "OP_SHL", "OP_SHR"},
            writes={"OUTPUT_LO", "OUTPUT_HI"},
            kind="ffn",
            bake_fn=bake,
            migrated=True,
        smoke_tests={
            "TestSmoke32Bit::test_shl_8bit",
            "TestSmoke32Bit::test_shr_8bit",
            "TestSmokeShift::test_shl",
            "TestSmokeShift::test_shr",
        },
        spec_section="BLOG_SPEC.md#shifts",
        )

    def make_install():
        def bake(block, dim_positions, S):
            if builder.composite is None:
                return  # No stage bakes ran (lookup mode safety).
            block.ffn = builder.composite

        return Operation(
            name="l13_alu_shift_install",
            phase=13.5,
            reads=set(),
            writes=set(),
            kind="block",
            bake_fn=bake,
            layer_idx=13,
            migrated=True,
        smoke_tests={
            "TestSmoke32Bit::test_shl_8bit",
            "TestSmoke32Bit::test_shr_8bit",
            "TestSmokeShift::test_shl",
            "TestSmokeShift::test_shr",
        },
        spec_section="BLOG_SPEC.md#shifts",
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


def make_l8_alu_postop_attach_op(alu_mode: str = 'lookup') -> Operation:
    return _mark_structural_declarations(
        _make_alu_postop_attach_op("l8_alu_postop_attach", 8, "ALUAddSub", alu_mode)
    )


def make_l9_alu_postop_attach_op(alu_mode: str = 'lookup') -> Operation:
    return _mark_structural_declarations(
        _make_alu_postop_attach_op("l9_alu_postop_attach", 9, "ALUAddSub", alu_mode)
    )


def make_l10_alu_postop_attach_op(alu_mode: str = 'lookup') -> Operation:
    return _mark_structural_declarations(
        _make_alu_postop_attach_op("l10_alu_postop_attach", 10, "ALUAndOrXor", alu_mode)
    )


def make_l11_alu_postop_attach_op(alu_mode: str = 'lookup') -> Operation:
    return _mark_structural_declarations(
        _make_alu_postop_attach_op("l11_alu_postop_attach", 11, "ALUMul", alu_mode)
    )


def make_l12_alu_postop_attach_op(alu_mode: str = 'lookup') -> Operation:
    return _mark_structural_declarations(
        _make_alu_postop_attach_op("l12_alu_postop_attach", 12, "ALUMul", alu_mode)
    )


def make_l13_alu_postop_attach_op(alu_mode: str = 'lookup') -> Operation:
    return _mark_structural_declarations(
        _make_alu_postop_attach_op("l13_alu_postop_attach", 13, "ALUShift", alu_mode)
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
        reads=set(),
        writes=set(),
        kind="model",
        bake_fn=bake,
        phase=1002,
        migrated=True,
        smoke_tests={
            "TestSmoke32Bit::test_add_16bit",
            "TestSmoke32Bit::test_sub_16bit",
            "TestSmokeAddress::test_lea_basic",
            "TestSmokeBasic::test_add_basic",
            "TestSmokeBasic::test_sub_basic",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
    )


def make_efficient_l10_andorxor_wrap_op(alu_mode: str = 'lookup') -> Operation:
    """Replace L10 ``block.ffn`` with ``ALUAndOrXor(S, BD)`` (= bitwise neural ALU).

    Migrates the inline efficient-mode assignment at vm_step.py:
        ``model.blocks[10].ffn = EfficientALU_L10_Neural(S, BD)``
    (``EfficientALU_L10_Neural`` is an alias for ``ALUAndOrXor``.)

    ``kind="block"`` at phase=10.85 — runs after ``l10_post_op_attach``
    (phase=10.7, which inspects ``block.ffn.W_up`` to derive d_model) and the
    DIV/MOD install (phase=10.8); before any L11 ops at phase=11.0.
    """
    def bake(block, dim_positions, S):
        if alu_mode != 'efficient':
            return
        from ...efficient_alu_neural import ALUAndOrXor
        BD = _as_setdim_proxy(dim_positions)
        block.ffn = ALUAndOrXor(S, BD)

    return Operation(
        name="efficient_l10_andorxor_wrap",
        reads=set(),
        writes=set(),
        kind="block",
        bake_fn=bake,
        phase=10.85,
        layer_idx=10,
        migrated=True,
        smoke_tests={
            "TestSmoke32Bit::test_and_16bit",
            "TestSmoke32Bit::test_or_16bit",
            "TestSmoke32Bit::test_xor_16bit",
            "TestSmokeBitwise::test_and_basic",
            "TestSmokeBitwise::test_or_basic",
            "TestSmokeBitwise::test_xor_basic",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
    )


def make_efficient_l11_alumul_wrap_op(alu_mode: str = 'lookup') -> Operation:
    """Replace L11 ``block.ffn`` with ``ALUMul(S, BD)`` (= MUL neural ALU).

    Migrates the inline efficient-mode fallback at vm_step.py:
        ``model.blocks[11].ffn = EfficientALU_L11_L12_Neural(S, BD)``
    (``EfficientALU_L11_L12_Neural`` is an alias for ``ALUMul``.)

    ``kind="block"`` at phase=11.05 — runs AFTER ``layer11_mul_partial``
    (phase=11) which writes to ``block.ffn.W_up`` of the original PureFFN, and
    AFTER ``l11_alu_mul_bdtoge`` (phase=11.0) which installs
    ``FlattenedALUMul``. Our isinstance check below makes the wrap a no-op
    when ``FlattenedALUMul`` is already installed (the normal compile_full_vm
    flow). The install path remains a fallback for direct ``set_vm_weights``
    callers that don't run the 9 flattening ops.
    """
    def bake(block, dim_positions, S):
        if alu_mode != 'efficient':
            return
        from ...efficient_alu_neural import ALUMul, FlattenedALUMul
        # Don't clobber FlattenedALUMul if a sibling op already installed it
        # (the normal compile_full_vm flow). The 9 ``FlattenedALUMul`` installer
        # ops at phases 11.0..12.3 run alongside us; the bdtoge op (phase=11.0)
        # runs first and we skip the ALUMul install when it already did so.
        if isinstance(block.ffn, FlattenedALUMul):
            return
        BD = _as_setdim_proxy(dim_positions)
        block.ffn = ALUMul(S, BD)

    return Operation(
        name="efficient_l11_alumul_wrap",
        reads=set(),
        writes=set(),
        kind="block",
        bake_fn=bake,
        phase=11.05,
        layer_idx=11,
        migrated=True,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#binary-ALU",
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
        reads={"ALU_LO", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI", "OP_MUL"},
        writes=set(),
        kind="block",
        bake_fn=bake,
        phase=11.0,
        layer_idx=11,
        migrated=True,
        smoke_tests={
            "TestSmoke32Bit::test_mul_overflow",
            "TestSmokeBasic::test_mul_basic",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
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
        reads={"OP_MUL"},
        writes=set(),
        kind="block",
        bake_fn=bake,
        phase=11.1,
        layer_idx=11,
        migrated=True,
        smoke_tests={
            "TestSmoke32Bit::test_mul_overflow",
            "TestSmokeBasic::test_mul_basic",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
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
        reads={"OP_MUL"},
        writes=set(),
        kind="block",
        bake_fn=bake,
        phase=11.2,
        layer_idx=11,
        migrated=True,
        smoke_tests={
            "TestSmoke32Bit::test_mul_overflow",
            "TestSmokeBasic::test_mul_basic",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
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
        reads={"OP_MUL"},
        writes=set(),
        kind="block",
        bake_fn=bake,
        phase=11.3,
        layer_idx=11,
        migrated=True,
        smoke_tests={
            "TestSmoke32Bit::test_mul_overflow",
            "TestSmokeBasic::test_mul_basic",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
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
        reads={"OP_MUL"},
        writes=set(),
        kind="block",
        bake_fn=bake,
        phase=11.4,
        layer_idx=11,
        migrated=True,
        smoke_tests={
            "TestSmoke32Bit::test_mul_overflow",
            "TestSmokeBasic::test_mul_basic",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
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
        reads={"OP_MUL"},
        writes=set(),
        kind="block",
        bake_fn=bake,
        phase=12.0,
        layer_idx=11,
        migrated=True,
        smoke_tests={
            "TestSmoke32Bit::test_mul_overflow",
            "TestSmokeBasic::test_mul_basic",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
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
        reads={"OP_MUL"},
        writes=set(),
        kind="block",
        bake_fn=bake,
        phase=12.1,
        layer_idx=11,
        migrated=True,
        smoke_tests={
            "TestSmoke32Bit::test_mul_overflow",
            "TestSmokeBasic::test_mul_basic",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
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
        reads={"OP_MUL"},
        writes=set(),
        kind="block",
        bake_fn=bake,
        phase=12.2,
        layer_idx=11,
        migrated=True,
        smoke_tests={
            "TestSmoke32Bit::test_mul_overflow",
            "TestSmokeBasic::test_mul_basic",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
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
        reads=set(),
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        bake_fn=bake,
        phase=12.3,
        layer_idx=11,
        migrated=True,
        smoke_tests={
            "TestSmoke32Bit::test_mul_overflow",
            "TestSmokeBasic::test_mul_basic",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
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


def make_alu_divmod_composite_ops():
    """Build the 4 cooperating ops (3 stage + 1 install) for FlattenedDivMod.

    Returns ``[bdtoge, longdiv, getobd, install]`` — all sharing the same
    ``_FlattenedDivModBuilder`` so the install op can append the
    fully-constructed composite to ``model.blocks[10].post_ops``.

    Stage ops are kind="block", layer_idx=10 (so they have access to the
    block when needed; their bake_fns operate on the shared builder
    rather than the block). Install op is kind="block", layer_idx=10.
    """
    builder = _FlattenedDivModBuilder()

    def make_bdtoge():
        def bake(block, dim_positions, S):
            BD = _as_setdim_proxy(dim_positions)
            composite = builder.ensure(S, BD)
            composite.install_bdtoge()

        return Operation(
            name="l10_alu_divmod_bdtoge",
            phase=10.0,
            reads={"ALU_LO", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI",
                   "OP_DIV", "OP_MOD"},
            writes=set(),
            kind="block",
            bake_fn=bake,
            declarative_bake_fn=bake,
            layer_idx=10,
            migrated=True,
            declarative_authority="structural_model",
        smoke_tests={
            "TestSmokeBasic::test_div_basic",
            "TestSmokeBasic::test_mod_basic",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
        )

    def make_longdiv():
        def bake(block, dim_positions, S):
            BD = _as_setdim_proxy(dim_positions)
            composite = builder.ensure(S, BD)
            composite.install_longdiv()

        return Operation(
            name="l10_alu_divmod_longdiv",
            phase=10.1,
            reads={"OP_DIV", "OP_MOD"},
            writes=set(),
            kind="block",
            bake_fn=bake,
            declarative_bake_fn=bake,
            layer_idx=10,
            migrated=True,
            declarative_authority="structural_model",
        smoke_tests={
            "TestSmokeBasic::test_div_basic",
            "TestSmokeBasic::test_mod_basic",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
        )

    def make_getobd():
        def bake(block, dim_positions, S):
            BD = _as_setdim_proxy(dim_positions)
            composite = builder.ensure(S, BD)
            composite.install_getobd()

        return Operation(
            name="l10_alu_divmod_getobd",
            phase=10.2,
            reads={"OP_DIV", "OP_MOD", "MARK_AX"},
            writes={"OUTPUT_LO", "OUTPUT_HI"},
            kind="block",
            bake_fn=bake,
            declarative_bake_fn=bake,
            layer_idx=10,
            migrated=True,
            declarative_authority="structural_model",
        smoke_tests={
            "TestSmokeBasic::test_div_basic",
            "TestSmokeBasic::test_mod_basic",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
        )

    def make_install():
        def bake(block, dim_positions, S):
            if builder.composite is None:
                # No stage bakes ran (defensive). Skip cleanly.
                return
            block.post_ops.append(builder.composite)

        return Operation(
            name="l10_alu_divmod_install",
            phase=10.8,
            reads=set(),
            writes=set(),
            kind="block",
            bake_fn=bake,
            declarative_bake_fn=bake,
            layer_idx=10,
            migrated=True,
            declarative_authority="structural_model",
        smoke_tests={
            "TestSmokeBasic::test_div_basic",
            "TestSmokeBasic::test_mod_basic",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
        )

    return [make_bdtoge(), make_longdiv(), make_getobd(), make_install()]


# Single-op factory shims for callers that want one op (e.g. unit tests).
# Each returns a fresh builder so the ops aren't entangled across factories.
def make_l10_alu_divmod_bdtoge_op() -> Operation:
    """L10 stage 1: BD → GE format conversion (standalone factory)."""
    return make_alu_divmod_composite_ops()[0]


def make_l10_alu_divmod_longdiv_op() -> Operation:
    """L10 stage 2: long-division pipeline (standalone factory)."""
    return make_alu_divmod_composite_ops()[1]


def make_l10_alu_divmod_getobd_op() -> Operation:
    """L10 stage 3: GE → BD format conversion (standalone factory)."""
    return make_alu_divmod_composite_ops()[2]


def make_l10_alu_divmod_install_op() -> Operation:
    """L10 install op: append composite to block.post_ops (standalone factory)."""
    return make_alu_divmod_composite_ops()[3]


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
        if alu_mode == 'lookup':
            attn10.alibi_slopes[4] = 1.0  # head 4: STACK0 byte relay for bitwise

    return Operation(
        name="layer10_residual_alibi_slopes",
        reads=set(),
        writes=set(),
        kind="model",
        bake_fn=_bake,
        declarative_bake_fn=_bake,
        phase=999.1,
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
        phase=8.0,
        reads=set(),
        writes=set(),
        kind="model",  # no-op model op; documentation only
        bake_fn=bake,
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
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
        phase=8.1,
        reads=set(),
        writes=set(),
        kind="model",
        bake_fn=bake,
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
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
        phase=8.2,
        reads=set(),
        writes=set(),
        kind="model",
        bake_fn=bake,
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
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
        phase=8.3,
        reads=set(),
        writes=set(),
        kind="model",
        bake_fn=bake,
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
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
        phase=8.4,
        reads=set(),
        writes=set(),
        kind="model",
        bake_fn=bake,
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        smoke_tests=_L8_ADDSUB_SMOKE_TESTS,
        spec_section="BLOG_SPEC.md#binary-ALU",
    )
