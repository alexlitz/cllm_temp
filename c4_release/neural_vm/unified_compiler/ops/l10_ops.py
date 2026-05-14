"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ..layer_compiler import Operation
from ..primitives import AO, AP, DeclarativeAttentionHeadSpec, Primitives
from .shared import _as_setdim_proxy
from .shared import _bake_post_op_into


def _bake_layer10_carry_relay_head(attn, BD, S, HD) -> None:
    """Declarative L10 head 0 carry relay spec."""
    AX_IDX = 1
    L = S
    Primitives.generate_attention_head(
        attn,
        DeclarativeAttentionHeadSpec(
            head_idx=0,
            q=(
                AP(0, BD.IS_BYTE, L),
                AP(0, BD.CONST, -L / 2),
                AP(33, BD.H1 + AX_IDX, L),
                AP(33, BD.CONST, -L / 2),
            ),
            k=(AP(0, BD.MARK_AX, L), AP(33, BD.CONST, L)),
            v=(AP(1, BD.CARRY + 1, 1.0), AP(2, BD.CARRY + 2, 1.0)),
            o=(AO(BD.CARRY + 1, 1, 1.0), AO(BD.CARRY + 2, 2, 1.0)),
        ),
        HD,
    )


def _bake_layer10_byte_passthrough_head(attn, BD, S, HD) -> None:
    """Declarative L10 head 1 AX byte passthrough spec."""
    AX_IDX = 1
    Primitives.byte_passthrough_chain(
        attn,
        head_idx=1,
        source_marker_dim=BD.H1 + AX_IDX,
        target_marker_dim=BD.H1 + AX_IDX,
        value_lo_dim=BD.CLEAN_EMBED_LO,
        value_hi_dim=BD.CLEAN_EMBED_HI,
        suppress_op_dims=[BD.OP_IMM, BD.TEMP + 3],
        S=S,
        HD=HD,
        alibi_slope=1.0,
    )


def _bake_layer10_sp_byte_passthrough_head(attn, BD, S, HD) -> None:
    """Declarative L10 head 2 SP byte passthrough spec."""
    SP_IDX = 2
    Primitives.byte_passthrough_chain(
        attn,
        head_idx=2,
        source_marker_dim=BD.H1 + SP_IDX,
        target_marker_dim=BD.H1 + SP_IDX,
        value_lo_dim=BD.CLEAN_EMBED_LO,
        value_hi_dim=BD.CLEAN_EMBED_HI,
        suppress_op_dims=[BD.PSH_AT_SP],
        S=S,
        HD=HD,
        alibi_slope=1.0,
        is_byte_strength=1.0,
        has_se_strength=2.0,
        suppress_strength=2.0,
        q0_threshold=1.5,
        gate_const=-30000.0,
        gate_extras=[
            (BD.IS_BYTE, 10000.0),
            (BD.PSH_AT_SP, -10000.0),
            (BD.CMP + 3, -10000.0),
        ],
    )


def _bake_layer10_psh_stack0_passthrough_head(attn, BD, S, HD) -> None:
    """Declarative L10 head 3 PSH STACK0 passthrough spec."""
    AX_IDX = 1
    BP_IDX = 3
    L = S
    q = [
        AP(0, BD.IS_BYTE, L),
        AP(1, BD.H4 + BP_IDX, L),
        AP(1, BD.H1 + BP_IDX, -L),
        AP(1, BD.CONST, -L / 2),
        AP(2, BD.BYTE_INDEX_3, -L),
        AP(2, BD.CONST, L / 2),
        AP(3, BD.PSH_AT_SP, L),
        AP(3, BD.CONST, -L / 2),
        AP(4, BD.BYTE_INDEX_0, L),
        AP(5, BD.BYTE_INDEX_1, L),
        AP(6, BD.BYTE_INDEX_2, L),
        AP(33, BD.CONST, -30000.0),
        AP(33, BD.IS_BYTE, 10000.0),
        AP(33, BD.H4 + BP_IDX, 10000.0),
        AP(33, BD.H1 + BP_IDX, -10000.0),
        AP(33, BD.PSH_AT_SP, 10000.0),
        AP(33, BD.MARK_STACK0, -10000.0),
    ]
    k = [
        AP(0, BD.IS_BYTE, L),
        AP(1, BD.H1 + AX_IDX, L),
        AP(2, BD.BYTE_INDEX_0, -L),
        AP(2, BD.CONST, L / 2),
        AP(4, BD.BYTE_INDEX_1, L),
        AP(5, BD.BYTE_INDEX_2, L),
        AP(6, BD.BYTE_INDEX_3, L),
        AP(33, BD.CONST, 5.0),
    ]
    v = []
    o = []
    for k_idx in range(16):
        v.append(AP(k_idx, BD.CLEAN_EMBED_LO + k_idx, 1.0))
        v.append(AP(16 + k_idx, BD.CLEAN_EMBED_HI + k_idx, 1.0))
        o.append(AO(BD.OUTPUT_LO + k_idx, k_idx, 3.0))
        o.append(AO(BD.OUTPUT_HI + k_idx, 16 + k_idx, 3.0))
    Primitives.generate_attention_head(
        attn,
        DeclarativeAttentionHeadSpec(
            head_idx=3,
            q=tuple(q),
            k=tuple(k),
            v=tuple(v),
            o=tuple(o),
        ),
        HD,
    )


def _bake_layer10_stack0_byte_relay_head(attn, BD, S, HD) -> None:
    """Declarative L10 head 4 STACK0 byte relay spec."""
    AX_IDX = 1
    SP_IDX = 2
    BP_IDX = 3
    L = S
    q = [
        AP(0, BD.IS_BYTE, L),
        AP(0, BD.H1 + AX_IDX, L),
        AP(0, BD.CONST, -L * 1.5),
        AP(1, BD.BYTE_INDEX_0, L),
        AP(2, BD.BYTE_INDEX_1, L),
        AP(3, BD.BYTE_INDEX_2, L),
        AP(4, BD.BYTE_INDEX_3, -L),
        AP(4, BD.CONST, L / 2),
        AP(33, BD.CONST, -20000.0),
        AP(33, BD.IS_BYTE, 10000.0),
        AP(33, BD.H1 + AX_IDX, 10000.0),
        AP(33, BD.HAS_SE, 10000.0),
    ]
    k = [
        AP(0, BD.IS_BYTE, L),
        AP(1, BD.H4 + BP_IDX, L),
        AP(1, BD.H1 + BP_IDX, -2 * L),
        AP(1, BD.H1 + SP_IDX, -2 * L),
        AP(1, BD.H1 + AX_IDX, -L),
        AP(2, BD.BYTE_INDEX_1, L),
        AP(3, BD.BYTE_INDEX_2, L),
        AP(33, BD.CONST, 5.0),
    ]
    v = []
    o = []
    for k_idx in range(16):
        v.append(AP(k_idx, BD.CLEAN_EMBED_LO + k_idx, 1.0))
        v.append(AP(16 + k_idx, BD.CLEAN_EMBED_HI + k_idx, 1.0))
        o.append(AO(BD.ALU_LO + k_idx, k_idx, 2.0))
        o.append(AO(BD.ALU_HI + k_idx, 16 + k_idx, 2.0))
    Primitives.generate_attention_head(
        attn,
        DeclarativeAttentionHeadSpec(
            head_idx=4,
            q=tuple(q),
            k=tuple(k),
            v=tuple(v),
            o=tuple(o),
        ),
        HD,
    )


def make_layer10_carry_relay_op() -> Operation:
    """Topology anchor for L10 head 0 carry relay.

    The actual weight bake is owned by ``layer10_carry_relay_bake`` below,
    pinned to ``model.blocks[10].attn``.
    """
    def bake(attn, dim_positions, S):
        return None

    return Operation(
        name="layer10_carry_relay",
        phase=10,
        reads={"MARK_AX", "IS_BYTE", "H1", "CARRY"},
        writes={"CARRY"},  # broadcast
        kind="attn",
        bake_fn=bake,
        migrated=True,
        declarative_authority="topology_anchor",
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def make_layer10_byte_passthrough_op() -> Operation:
    """Topology anchor for L10 head 1 AX byte passthrough.

    The spec-generated weight bake is owned by
    ``layer10_byte_passthrough_bake`` below, pinned to
    ``model.blocks[10].attn``.
    """
    def bake(attn, dim_positions, S):
        return None

    return Operation(
        name="layer10_byte_passthrough",
        phase=10,
        reads={"IS_BYTE", "HAS_SE", "OP_IMM", "TEMP",
               "H1", "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="attn",
        bake_fn=bake,
        migrated=True,
        declarative_authority="topology_anchor",
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def make_layer10_sp_byte_passthrough_op() -> Operation:
    """Topology anchor for L10 head 2 SP byte passthrough.

    The spec-generated weight bake is owned by
    ``layer10_sp_byte_passthrough_bake`` below, pinned to
    ``model.blocks[10].attn``.
    """
    def bake(attn, dim_positions, S):
        return None

    return Operation(
        name="layer10_sp_byte_passthrough",
        phase=10,
        reads={"IS_BYTE", "HAS_SE", "H1",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="attn",
        bake_fn=bake,
        migrated=True,
        declarative_authority="topology_anchor",
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def make_layer10_psh_stack0_passthrough_op() -> Operation:
    """Topology anchor for L10 head 3 PSH STACK0 passthrough.

    The actual weight bake is owned by
    ``layer10_psh_stack0_passthrough_bake`` below, pinned to
    ``model.blocks[10].attn``.
    """
    def bake(attn, dim_positions, S):
        return None

    return Operation(
        name="layer10_psh_stack0_passthrough",
        phase=10,
        reads={"MARK_STACK0", "OP_PSH", "AX_CARRY_LO", "AX_CARRY_HI",
               "OP_LI", "OP_LC", "OP_SI", "OP_SC"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="attn",
        bake_fn=bake,
        migrated=True,
        declarative_authority="topology_anchor",
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


# -- L10 attention bake ops (migrated 2026-05-10) -----------------------------
#
# These five ``kind="block", layer_idx=10, migrated=True`` ops bake the five
# inline ``_set_layer10_*`` attention calls that used to live in
# ``set_vm_weights`` (both the ``alu_mode == 'lookup'`` and
# ``alu_mode == 'efficient'`` branches). The inline calls have been removed
# from both branches; these ops now own the bake. Phases 10.0-10.4 preserve
# the original ordering. The five ``layer10_*`` kind="attn" placeholders
# above are retained as migrated no-op dep-graph anchors so the LayerCompiler
# topology does not shift downstream block assignments.
#
# All five target ``model.blocks[10].attn`` and run BEFORE legacy_bake (999),
# so the alibi_slopes mutations and the L10 FFN bake inside set_vm_weights
# still execute in their original order. The attn weight slots they write
# are NOT touched by legacy_bake after the inline removals.


def make_layer10_carry_relay_bake_op() -> Operation:
    """Bake ``_set_layer10_carry_relay`` into ``model.blocks[10].attn``.

    Was an inline call in ``set_vm_weights`` (both lookup and efficient
    branches): ``_set_layer10_carry_relay(attn10, S, BD, HD)``. Inline call
    removed; this op now owns the bake. Phase=10.0 preserves the original
    ordering relative to the four sibling L10 attn bake ops below.
    """
    def bake(block, dim_positions, S):
        proxy = _as_setdim_proxy(dim_positions)
        attn = block.attn
        HD = attn.W_q.shape[0] // attn.num_heads
        _bake_layer10_carry_relay_head(attn, proxy, S, HD)

    # Dim-ownership claims: L10 attn head 0 CARRY relay (AX marker → AX bytes).
    #   W_v[0*HD + 1, CARRY + 1]  (CARRY[1] = ADD byte carry)
    #   W_v[0*HD + 2, CARRY + 2]  (CARRY[2] = SUB byte borrow)
    #   W_o[CARRY + 1, 0*HD + 1]
    #   W_o[CARRY + 2, 0*HD + 2]
    _claims = {
        (10, "attn_W_v", "0_1", "CARRY+1"),
        (10, "attn_W_v", "0_2", "CARRY+2"),
    }

    return Operation(
        name="layer10_carry_relay_bake",
        phase=10.0,
        reads={"MARK_AX", "IS_BYTE", "H1", "CARRY", "CONST"},
        writes={"CARRY"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        layer_idx=10,
        migrated=True,
        declarative_authority="spec_generated",
        claims=_claims,
        smoke_tests={
            "TestSmoke32Bit::test_add_16bit",
            "TestSmoke32Bit::test_sub_16bit",
            "TestSmokeAddress::test_lea_basic",
            "TestSmokeBasic::test_add_basic",
            "TestSmokeBasic::test_sub_basic",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
    )


def make_layer10_byte_passthrough_bake_op() -> Operation:
    """Bake ``_set_layer10_byte_passthrough`` into ``model.blocks[10].attn``.

    Was an inline call in ``set_vm_weights`` (both branches):
    ``_set_layer10_byte_passthrough(attn10, S, BD, HD)``. Inline call
    removed; this op now owns the bake. Phase=10.1.
    """
    def bake(block, dim_positions, S):
        proxy = _as_setdim_proxy(dim_positions)
        attn = block.attn
        HD = attn.W_q.shape[0] // attn.num_heads
        _bake_layer10_byte_passthrough_head(attn, proxy, S, HD)

    # Dim-ownership claims: L10 attn head 1 AX byte passthrough.
    # ``byte_passthrough_chain`` writes V slots 0..31 + O writes OUTPUT_LO/HI:
    #   W_v[1*HD + k, CLEAN_EMBED_LO + k]    for k=0..15
    #   W_v[1*HD + 16 + k, CLEAN_EMBED_HI + k]  for k=0..15
    #   W_o[OUTPUT_LO + k, 1*HD + k]         for k=0..15
    #   W_o[OUTPUT_HI + k, 1*HD + 16 + k]    for k=0..15
    _claims = set()
    for k in range(16):
        _claims.add((10, "attn_W_v", f"1_{k}", f"CLEAN_EMBED_LO+{k}"))
        _claims.add((10, "attn_W_v", f"1_{16 + k}", f"CLEAN_EMBED_HI+{k}"))

    return Operation(
        name="layer10_byte_passthrough_bake",
        phase=10.1,
        reads={"IS_BYTE", "HAS_SE", "OP_IMM", "TEMP",
               "H1", "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
               "BYTE_INDEX_3", "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        layer_idx=10,
        migrated=True,
        declarative_authority="spec_generated",
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def make_layer10_sp_byte_passthrough_bake_op() -> Operation:
    """Bake ``_set_layer10_sp_byte_passthrough`` into ``model.blocks[10].attn``.

    Was an inline call in ``set_vm_weights`` (both branches):
    ``_set_layer10_sp_byte_passthrough(attn10, S, BD, HD)``. Inline call
    removed; this op now owns the bake. Phase=10.2.
    """
    def bake(block, dim_positions, S):
        proxy = _as_setdim_proxy(dim_positions)
        attn = block.attn
        HD = attn.W_q.shape[0] // attn.num_heads
        _bake_layer10_sp_byte_passthrough_head(attn, proxy, S, HD)

    # Dim-ownership claims: L10 attn head 2 SP byte passthrough.
    #   W_v[2*HD + k, CLEAN_EMBED_LO + k]      for k=0..15
    #   W_v[2*HD + 16 + k, CLEAN_EMBED_HI + k] for k=0..15
    _claims = set()
    for k in range(16):
        _claims.add((10, "attn_W_v", f"2_{k}", f"CLEAN_EMBED_LO+{k}"))
        _claims.add((10, "attn_W_v", f"2_{16 + k}", f"CLEAN_EMBED_HI+{k}"))

    return Operation(
        name="layer10_sp_byte_passthrough_bake",
        phase=10.2,
        reads={"IS_BYTE", "HAS_SE", "H1", "PSH_AT_SP", "CMP",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        layer_idx=10,
        migrated=True,
        declarative_authority="spec_generated",
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def make_layer10_psh_stack0_passthrough_bake_op() -> Operation:
    """Bake ``_set_layer10_psh_stack0_passthrough`` into ``model.blocks[10].attn``.

    Was an inline call in ``set_vm_weights`` (both branches):
    ``_set_layer10_psh_stack0_passthrough(attn10, S, BD, HD)``. Inline call
    removed; this op now owns the bake. Phase=10.3.
    """
    def bake(block, dim_positions, S):
        proxy = _as_setdim_proxy(dim_positions)
        attn = block.attn
        HD = attn.W_q.shape[0] // attn.num_heads
        _bake_layer10_psh_stack0_passthrough_head(attn, proxy, S, HD)

    # Dim-ownership claims: L10 attn head 3 PSH STACK0 passthrough.
    #   W_v[3*HD + k, CLEAN_EMBED_LO + k]      for k=0..15
    #   W_v[3*HD + 16 + k, CLEAN_EMBED_HI + k] for k=0..15
    _claims = set()
    for k in range(16):
        _claims.add((10, "attn_W_v", f"3_{k}", f"CLEAN_EMBED_LO+{k}"))
        _claims.add((10, "attn_W_v", f"3_{16 + k}", f"CLEAN_EMBED_HI+{k}"))

    return Operation(
        name="layer10_psh_stack0_passthrough_bake",
        phase=10.3,
        reads={"MARK_STACK0", "IS_BYTE", "PSH_AT_SP", "H1", "H4",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        layer_idx=10,
        migrated=True,
        declarative_authority="spec_generated",
        claims=_claims,
        smoke_tests={"TestSmokeBasic::test_add_basic"},
        spec_section="BLOG_SPEC.md#registers",
    )


def make_layer10_stack0_byte_relay_bake_op() -> Operation:
    """Bake ``_set_layer10_stack0_byte_relay`` into ``model.blocks[10].attn``.

    Was an inline call in ``set_vm_weights`` (lookup branch only):
    ``_set_layer10_stack0_byte_relay(attn10, S, BD, HD)``. Inline call
    removed; this op now owns the bake. Phase=10.4.
    """
    def bake(block, dim_positions, S):
        proxy = _as_setdim_proxy(dim_positions)
        attn = block.attn
        HD = attn.W_q.shape[0] // attn.num_heads
        _bake_layer10_stack0_byte_relay_head(attn, proxy, S, HD)

    # Dim-ownership claims: L10 attn head 4 STACK0-byte-relay (→ ALU at AX byte).
    #   W_v[4*HD + k, CLEAN_EMBED_LO + k]      for k=0..15
    #   W_v[4*HD + 16 + k, CLEAN_EMBED_HI + k] for k=0..15
    _claims = set()
    for k in range(16):
        _claims.add((10, "attn_W_v", f"4_{k}", f"CLEAN_EMBED_LO+{k}"))
        _claims.add((10, "attn_W_v", f"4_{16 + k}", f"CLEAN_EMBED_HI+{k}"))

    return Operation(
        name="layer10_stack0_byte_relay_bake",
        phase=10.4,
        reads={"IS_BYTE", "HAS_SE", "H1", "H4",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"ALU_LO", "ALU_HI"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        layer_idx=10,
        migrated=True,
        declarative_authority="spec_generated",
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def make_layer10_alu_op() -> Operation:
    """L10 FFN: AND/OR/XOR + DIV/MOD setup.

    Pinned to ``layer_idx=10`` via ``kind="block"``: the legacy
    ``set_vm_weights`` lookup branch targeted ``model.blocks[10].ffn``.
    Without pinning, dep-graph layer assignment could place this op on the
    wrong block. ``phase=10.2`` is before
    ``make_l10_post_op_attach_op`` (phase=10.7) and
    ``make_l10_alu_divmod_install_op`` (phase=10.8) so they don't conflict.

    Migrated 2026-05-10: the inline ``_set_layer10_alu(ffn10, S, BD)`` call
    in the lookup branch of ``set_vm_weights`` has been removed; this op
    now owns the bake. (Per Unit 9 diagnosis, this migration is SAFE so
    long as ``make_l10_post_op_attach_op`` is NOT modified.)

    Declarations-only note: intentionally unsupported until the
    ``_set_layer10_alu`` lookup-table helper is lowered into explicit FFN
    units or a structural stage module. Tagging the helper itself as
    declarations-only would hide the remaining imperative weight generator.
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer10_alu
        _set_layer10_alu(block.ffn, S, _as_setdim_proxy(dim_positions))

    return Operation(
        name="layer10_alu",
        phase=10.2,
        reads={"MARK_AX", "ALU_LO", "AX_CARRY_LO", "ALU_HI", "AX_CARRY_HI",
               "OP_OR", "OP_XOR", "OP_AND", "OP_DIV", "OP_MOD"},
        writes={"OUTPUT_LO", "OUTPUT_HI", "DIV_STAGING"},
        kind="block",
        bake_fn=bake,
        layer_idx=10,
        migrated=True,
        # Staleness invariants (Phase 3 / Agent G): L10 ALU consumes
        # ALU_LO/HI (operand A) and AX_CARRY_LO/HI (operand B) at the AX
        # marker for bitwise OR/XOR/AND + DIV/MOD setup. Both must be
        # current-step fresh values.
        consumes_fresh={
            "ALU_LO": "AX_byte0",
            "ALU_HI": "AX_byte0",
            "AX_CARRY_LO": "AX_byte0",
            "AX_CARRY_HI": "AX_byte0",
        },
        # ``_set_layer10_alu`` writes the comparison-combine (18 units) +
        # bitwise-cross-product (~1536) + AX passthrough (~32) + DIV/MOD
        # setup units, reaching unit 1845. No other op writes to L10 FFN
        # so this op holds the per-layer width annotation.
        ffn_units_used=1846,
        smoke_tests={
            "TestSmoke32Bit::test_and_16bit",
            "TestSmoke32Bit::test_or_16bit",
            "TestSmoke32Bit::test_xor_16bit",
            "TestSmokeBasic::test_div_basic",
            "TestSmokeBasic::test_mod_basic",
            "TestSmokeBitwise::test_and_basic",
            "TestSmokeBitwise::test_or_basic",
            "TestSmokeBitwise::test_xor_basic",
            "TestSmokeComparison::test_eq_false",
            "TestSmokeComparison::test_eq_true",
            "TestSmokeComparison::test_ge_true",
            "TestSmokeComparison::test_gt_true",
            "TestSmokeComparison::test_le_true",
            "TestSmokeComparison::test_lt_true",
            "TestSmokeComparison::test_ne_true",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
    )


def make_layer10_stack0_byte_relay_op() -> Operation:
    """Topology anchor for L10 head 4 STACK0 byte relay.

    The actual weight bake is owned by ``layer10_stack0_byte_relay_bake``
    above, pinned to ``model.blocks[10].attn``.
    """
    def bake(attn, dim_positions, S):
        return None

    return Operation(
        name="layer10_stack0_byte_relay",
        phase=10,
        reads={"MARK_AX", "IS_BYTE", "H1", "MARK_STACK0", "STACK0_BYTE0",
               "TEMP", "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"ALU_LO", "ALU_HI"},
        kind="attn",
        bake_fn=bake,
        migrated=True,
        declarative_authority="topology_anchor",
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def make_l10_post_ops_combined() -> Operation:
    """Combined L10 post_ops: BinaryOpByteZeroing + 3x CarryPropagation +
    BitwiseBytePropagation + ComparisonCombine, baked sequentially into
    one FFN.

    Originally these were 6 separate post_ops on L10 in vm_step.py. Per Phase 0
    policy they belong in their own blocks, but for the migration we combine
    them additively into a single ffn at phase=10.5 so the compiler places
    them right after layer10_alu.
    """
    def bake(ffn, dim_positions, S):
        from ...vm_step import (
            BinaryOpByteZeroingPostOp,
            CarryPropagationPostOp,
            BitwiseBytePropagationPostOp,
            ComparisonCombine,
        )
        d_model = ffn.W_up.shape[1]
        offset = 0
        # Thread dim_positions so each fresh post-op instance bakes against
        # the compact layout, matching the per-block post_op attach path.
        offset = _bake_post_op_into(
            ffn, BinaryOpByteZeroingPostOp(d_model, S, dim_positions=dim_positions), offset)
        offset = _bake_post_op_into(
            ffn, CarryPropagationPostOp(d_model, S, byte_idx=0, cascade=False,
                                        dim_positions=dim_positions), offset)
        offset = _bake_post_op_into(
            ffn, CarryPropagationPostOp(d_model, S, byte_idx=1, cascade=True,
                                        dim_positions=dim_positions), offset)
        offset = _bake_post_op_into(
            ffn, CarryPropagationPostOp(d_model, S, byte_idx=2, cascade=True,
                                        dim_positions=dim_positions), offset)
        offset = _bake_post_op_into(
            ffn, BitwiseBytePropagationPostOp(d_model, S, dim_positions=dim_positions), offset)
        offset = _bake_post_op_into(
            ffn, ComparisonCombine(d_model, S, dim_positions=dim_positions), offset)

    # phase=10.5 so it lands AFTER layer10_alu (phase=10) but BEFORE later layers
    # which depend on its OUTPUT_LO/HI updates. Note: float phases work because
    # phase comparison uses < / >.
    return Operation(
        name="l10_post_ops_combined",
        phase=10.5,
        reads={
            "MARK_AX", "MARK_PC", "IS_BYTE", "H1",
            "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
            "OP_SHL", "OP_SHR",
            "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
            "OP_OR", "OP_XOR", "OP_AND",
            "OP_LEA", "OP_IMM", "OP_JMP", "OP_JSR", "OP_BZ", "OP_BNZ",
            "OP_ENT", "OP_ADJ", "OP_LEV", "OP_LI", "OP_LC",
            "OP_SI", "OP_SC", "OP_PSH", "OP_EXIT", "OP_NOP",
            "OP_PUTCHAR", "OP_GETCHAR",
            "OUTPUT_LO", "OUTPUT_HI", "ALU_LO", "ALU_HI",
            "CARRY", "CMP", "TEMP",
            "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
        },
        writes={"OUTPUT_LO", "OUTPUT_HI", "CARRY"},
        kind="ffn",
        bake_fn=bake,
        declarative_bake_fn=bake,
        migrated=True,
        declarative_authority="declarative",
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def make_l10_post_op_attach_op(alu_mode: str = "lookup") -> Operation:
    """Block-level op: attach L10 post_op modules onto block.post_ops.

    Migrates the inline `model.blocks[10].post_ops.append(...)` calls in
    `set_vm_weights` for both lookup and efficient ALU modes into a compiler
    block op. The attached modules are the structural post-FFN passes that
    `_expand_wrapper_blocks` later splits into their own blocks.

    Modules attached (lookup mode):
      BinaryOpByteZeroingPostOp,
      CarryPropagationPostOp x3 (byte 0 no-cascade, bytes 1-2 cascade),
      BitwiseBytePropagationPostOp.
      (DIV/MOD post_op is appended by ``make_l10_alu_divmod_install_op``
      at phase=10.8 — see ``efficient_alu_divmod_split.FlattenedDivMod``.)

    Modules attached (efficient mode):
      BinaryOpByteZeroingPostOp,
      CarryPropagationPostOp x3,
      BitwiseBytePropagationPostOp,
      ComparisonCombine.
      (DIV/MOD post_op is appended by ``make_l10_alu_divmod_install_op``
      at phase=10.8 — see ``efficient_alu_divmod_split.FlattenedDivMod``.)

    The existing `make_l10_post_ops_combined` is unrelated: it bakes the
    LOGIC of the FFN-style post_ops into a single phase-10.5 FFN (a parallel
    representation), not the attached module list. Both can coexist.

    phase=10.7: runs after L10 FFN bake (phase=10) and the combined FFN
    (phase=10.5), but well before structural post-passes (1100+).
    """
    if alu_mode not in ("lookup", "efficient"):
        raise ValueError(
            f"alu_mode must be 'lookup' or 'efficient'; got {alu_mode!r}"
        )

    def bake(block, dim_positions, S):
        from ...vm_step import (
            BinaryOpByteZeroingPostOp,
            CarryPropagationPostOp,
            BitwiseBytePropagationPostOp,
            ComparisonCombine,
        )
        # Use the block's d_model when available; fall back to 512 to mirror
        # the previous inline behavior.
        d_model = 512
        if hasattr(block, "ffn") and hasattr(block.ffn, "W_up"):
            try:
                d_model = block.ffn.W_up.shape[1]
            except (AttributeError, IndexError):
                d_model = 512

        # Pass dim_positions so each post-op bakes against the compact layout
        # rather than legacy `_SetDim` positions. Without this, the post-ops
        # write to / read from `_SetDim.OUTPUT_LO/HI/CARRY/H1/OP_*` etc.,
        # which alias unrelated compact dims (e.g. `_SetDim.H1+1=68` aliases
        # compact `EMBED_HI[15]`; `_SetDim.CARRY=392` aliases a different
        # compact slot, etc.), corrupting OUTPUT/CARRY/CMP flags and silently
        # zeroing or scrambling the binary-op result. Threading dim_positions
        # to all 4 post-op classes is the L10 counterpart of the L1 fix in
        # commit 5fc519d (BinaryOpByteZeroingPostOp).
        block.post_ops.append(
            BinaryOpByteZeroingPostOp(d_model=d_model, S=S, dim_positions=dim_positions)
        )
        block.post_ops.append(
            CarryPropagationPostOp(d_model=d_model, S=S, byte_idx=0, cascade=False,
                                   dim_positions=dim_positions)
        )
        block.post_ops.append(
            CarryPropagationPostOp(d_model=d_model, S=S, byte_idx=1, cascade=True,
                                   dim_positions=dim_positions)
        )
        block.post_ops.append(
            CarryPropagationPostOp(d_model=d_model, S=S, byte_idx=2, cascade=True,
                                   dim_positions=dim_positions)
        )
        block.post_ops.append(
            BitwiseBytePropagationPostOp(d_model=d_model, S=S, dim_positions=dim_positions)
        )
        if alu_mode == "efficient":
            # Pass the model's actual d_model so the underlying PureFFN's
            # Linear input dim matches the residual stream width. Without
            # this, ComparisonCombine builds a Linear(512, 18) which fails
            # forward when d_model != 512 (e.g., pin_io_only=True paths).
            block.post_ops.append(
                ComparisonCombine(d_model=d_model, S=S, dim_positions=dim_positions)
            )
        # DIV/MOD post_op (FlattenedDivMod) appended by
        # ``make_l10_alu_divmod_install_op`` (phase=10.8). Both modes use the
        # same flattened composite — its forward is byte-identical to the
        # previous EfficientDivMod_Neural.

    return Operation(
        name="l10_post_op_attach",
        reads=set(),
        writes=set(),
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        phase=10.7,
        layer_idx=10,
        migrated=True,
        declarative_authority="structural_model",
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )
