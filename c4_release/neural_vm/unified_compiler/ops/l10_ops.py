"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ..ir import CompilerIR, FFNRule
from ..layer_compiler import Operation
from ..primitives import AO, AP, DeclarativeAttentionHeadSpec, Primitives
from .shared import _as_setdim_proxy
from .shared import _bake_post_op_into


def _bake_layer10_carry_relay_head(attn, BD, S, HD) -> None:
    """Declarative L10 head 0 carry relay spec."""
    Primitives.generate_attention_head(
        attn,
        _layer10_carry_relay_head_spec(BD, S),
        HD,
    )


def _layer10_carry_relay_head_spec(BD, S) -> DeclarativeAttentionHeadSpec:
    AX_IDX = 1
    L = S
    return DeclarativeAttentionHeadSpec(
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
    )


def _byte_passthrough_chain_spec(
    BD,
    *,
    head_idx: int,
    source_marker_dim: int,
    target_marker_dim: int,
    value_lo_dim: int,
    value_hi_dim: int,
    suppress_op_dims,
    S: float,
    is_byte_strength: float = 3.0,
    has_se_strength: float = 1.0,
    suppress_strength: float = 3.0,
    q0_threshold: float = 3.5,
    gate_const: float = -20000.0,
    gate_target_marker: float = 10000.0,
    gate_has_se: float = 10000.0,
    gate_extras=None,
) -> DeclarativeAttentionHeadSpec:
    L = S
    q = [
        AP(0, BD.IS_BYTE, L * is_byte_strength),
        AP(0, BD.HAS_SE, L * has_se_strength),
        AP(0, BD.CONST, -L * q0_threshold),
        AP(1, target_marker_dim, L),
        AP(1, BD.CONST, -L / 2),
        AP(2, BD.BYTE_INDEX_3, -L),
        AP(2, BD.CONST, L / 2),
        AP(3, BD.BYTE_INDEX_0, L),
        AP(4, BD.BYTE_INDEX_1, L),
        AP(5, BD.BYTE_INDEX_2, L),
        AP(33, BD.CONST, gate_const),
        AP(33, target_marker_dim, gate_target_marker),
        AP(33, BD.HAS_SE, gate_has_se),
    ]
    for dim in suppress_op_dims:
        q.append(AP(0, dim, -L * suppress_strength))
    if gate_extras:
        for dim, weight in gate_extras:
            q.append(AP(33, dim, weight))

    k = [
        AP(0, BD.IS_BYTE, L),
        AP(1, source_marker_dim, L),
        AP(2, BD.BYTE_INDEX_0, -L),
        AP(2, BD.CONST, L / 2),
        AP(3, BD.BYTE_INDEX_1, L),
        AP(4, BD.BYTE_INDEX_2, L),
        AP(5, BD.BYTE_INDEX_3, L),
        AP(33, BD.CONST, 5.0),
    ]
    v = []
    o = []
    for idx in range(16):
        v.append(AP(idx, value_lo_dim + idx, 1.0))
        v.append(AP(16 + idx, value_hi_dim + idx, 1.0))
        o.append(AO(BD.OUTPUT_LO + idx, idx, 2.0))
        o.append(AO(BD.OUTPUT_HI + idx, 16 + idx, 2.0))
    return DeclarativeAttentionHeadSpec(
        head_idx=head_idx,
        q=tuple(q),
        k=tuple(k),
        v=tuple(v),
        o=tuple(o),
    )


def _bake_layer10_byte_passthrough_head(attn, BD, S, HD) -> None:
    """Declarative L10 head 1 AX byte passthrough spec."""
    AX_IDX = 1
    Primitives.generate_attention_head(
        attn,
        _byte_passthrough_chain_spec(
            BD,
            head_idx=1,
            source_marker_dim=BD.H1 + AX_IDX,
            target_marker_dim=BD.H1 + AX_IDX,
            value_lo_dim=BD.CLEAN_EMBED_LO,
            value_hi_dim=BD.CLEAN_EMBED_HI,
            suppress_op_dims=[BD.OP_IMM, BD.TEMP + 3],
            S=S,
        ),
        HD,
    )
    if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
        attn.alibi_slopes.data[1] = 1.0


def _bake_layer10_sp_byte_passthrough_head(attn, BD, S, HD) -> None:
    """Declarative L10 head 2 SP byte passthrough spec."""
    SP_IDX = 2
    Primitives.generate_attention_head(
        attn,
        _layer10_sp_byte_passthrough_head_spec(BD, S),
        HD,
    )
    if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
        attn.alibi_slopes.data[2] = 1.0


def _bake_layer10_bp_byte_passthrough_head(attn, BD, S, HD) -> None:
    """Declarative L10 head 7 BP byte passthrough spec."""
    Primitives.generate_attention_head(
        attn,
        _layer10_bp_byte_passthrough_head_spec(BD, S),
        HD,
    )
    if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
        attn.alibi_slopes.data[7] = 1.0


def _layer10_sp_byte_passthrough_head_spec(BD, S) -> DeclarativeAttentionHeadSpec:
    SP_IDX = 2
    spec = _byte_passthrough_chain_spec(
        BD,
        head_idx=2,
        source_marker_dim=BD.H1 + SP_IDX,
        target_marker_dim=BD.H1 + SP_IDX,
        value_lo_dim=BD.CLEAN_EMBED_LO,
        value_hi_dim=BD.CLEAN_EMBED_HI,
        suppress_op_dims=[BD.PSH_AT_SP],
        S=S,
        is_byte_strength=1.0,
        has_se_strength=2.0,
        suppress_strength=2.0,
        q0_threshold=1.5,
        gate_const=-30000.0,
        gate_extras=[
            (BD.IS_BYTE, 10000.0),
            (BD.MARK_SP, 10000.0),
            (BD.PSH_AT_SP, -10000.0),
        ],
    )
    L = S
    marker_s = 300.0
    # Marker carry-forward for SP byte 0 on step 1+. The byte-chain above
    # handles SP byte positions; the marker itself needs to copy the previous
    # step's SP byte 0 unless the current op is actively rewriting SP (PSH).
    return DeclarativeAttentionHeadSpec(
        head_idx=spec.head_idx,
        q=spec.q + (
            AP(34, BD.MARK_SP, marker_s),
            AP(34, BD.HAS_SE, marker_s),
            AP(34, BD.PSH_AT_SP, -2.0 * marker_s),
            AP(34, BD.CMP + 3, -2.0 * marker_s),
            AP(34, BD.CONST, -marker_s),
            AP(35, BD.MARK_SP, marker_s),
            AP(35, BD.HAS_SE, marker_s),
            AP(35, BD.PSH_AT_SP, -2.0 * marker_s),
            AP(35, BD.CMP + 3, -2.0 * marker_s),
            AP(35, BD.CONST, -marker_s),
        ),
        k=spec.k + (
            AP(34, BD.H1 + SP_IDX, marker_s),
            AP(35, BD.BYTE_INDEX_0, marker_s),
        ),
        v=spec.v,
        o=spec.o,
    )


def _layer10_bp_byte_passthrough_head_spec(BD, S) -> DeclarativeAttentionHeadSpec:
    BP_IDX = 3
    return _byte_passthrough_chain_spec(
        BD,
        head_idx=7,
        source_marker_dim=BD.H1 + BP_IDX,
        target_marker_dim=BD.H1 + BP_IDX,
        value_lo_dim=BD.CLEAN_EMBED_LO,
        value_hi_dim=BD.CLEAN_EMBED_HI,
        suppress_op_dims=[BD.OP_ENT, BD.OP_LEV],
        S=S,
        is_byte_strength=1.0,
        has_se_strength=2.0,
        suppress_strength=2.0,
        q0_threshold=1.5,
        gate_const=-30000.0,
        gate_extras=[
            (BD.IS_BYTE, 10000.0),
            (BD.OP_ENT, -10000.0),
            (BD.OP_LEV, -10000.0),
        ],
    )


def _bake_layer10_psh_stack0_passthrough_head(attn, BD, S, HD) -> None:
    """Declarative L10 head 3 PSH STACK0 passthrough spec."""
    Primitives.generate_attention_head(
        attn,
        _layer10_psh_stack0_passthrough_head_spec(BD, S),
        HD,
    )


def _layer10_psh_stack0_passthrough_head_spec(BD, S) -> DeclarativeAttentionHeadSpec:
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
    return DeclarativeAttentionHeadSpec(
        head_idx=3,
        q=tuple(q),
        k=tuple(k),
        v=tuple(v),
        o=tuple(o),
    )


def _bake_layer10_stack0_byte_relay_head(attn, BD, S, HD) -> None:
    """Declarative L10 STACK0 byte relay specs."""
    Primitives.generate_attention_head(
        attn,
        _layer10_stack0_byte_relay_head_spec(BD, S),
        HD,
    )
    Primitives.generate_attention_head(
        attn,
        _layer10_nonbitwise_stack0_byte_relay_head_spec(BD, S),
        HD,
    )
    Primitives.generate_attention_head(
        attn,
        _layer10_stack0_persistence_head_spec(BD, S),
        HD,
    )
    if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
        attn.alibi_slopes.data[6] = 1.0


def _layer10_stack0_byte_relay_head_spec(BD, S) -> DeclarativeAttentionHeadSpec:
    AX_IDX = 1
    L = S
    q = [
        AP(0, BD.CONST, -3000.0),
        AP(0, BD.IS_BYTE, 1000.0),
        AP(0, BD.H1 + AX_IDX, 1000.0),
        AP(0, BD.TEMP + 3, 1000.0),
        AP(0, BD.BYTE_INDEX_3, -3000.0),
        AP(1, BD.TEMP + 3, 50.0),
        AP(31, BD.BYTE_INDEX_0, 60.0),
        AP(32, BD.BYTE_INDEX_1, 60.0),
        AP(34, BD.BYTE_INDEX_2, 60.0),
        AP(33, BD.CONST, -30000.0),
        AP(33, BD.IS_BYTE, 10000.0),
        AP(33, BD.H1 + AX_IDX, 10000.0),
        AP(33, BD.TEMP + 3, 10000.0),
        AP(33, BD.BYTE_INDEX_3, -10000.0),
    ]
    k = [
        AP(0, BD.CONST, 10.0),
        AP(1, BD.MEM_STORE, 100.0),
        AP(1, BD.MARK_MEM, -200.0),
        AP(1, BD.CONST, -50.0),
        AP(31, BD.MEM_VAL_B2, 60.0),
        AP(32, BD.MEM_VAL_B3, 60.0),
        AP(34, BD.H3 + 4, 60.0),
        AP(34, BD.H2 + 4, -60.0),
        AP(33, BD.CONST, 5.0),
    ]
    v = []
    o = []
    for k_idx in range(16):
        v.append(AP(1 + k_idx, BD.CLEAN_EMBED_LO + k_idx, 1.0))
        v.append(AP(17 + k_idx, BD.CLEAN_EMBED_HI + k_idx, 1.0))
        o.append(AO(BD.ALU_LO + k_idx, 1 + k_idx, 11.0))
        o.append(AO(BD.ALU_HI + k_idx, 17 + k_idx, 11.0))
    v.append(AP(0, BD.CONST, 1.0))
    for k_idx in range(16):
        o.append(AO(BD.ALU_LO + k_idx, 0, -8.0))
        o.append(AO(BD.ALU_HI + k_idx, 0, -8.0))
    return DeclarativeAttentionHeadSpec(
        head_idx=4,
        q=tuple(q),
        k=tuple(k),
        v=tuple(v),
        o=tuple(o),
    )


def _layer10_nonbitwise_stack0_byte_relay_head_spec(BD, S) -> DeclarativeAttentionHeadSpec:
    """Relay stored STACK0 bytes for non-bitwise pop ALU byte post-ops.

    Head 4 is shared with the existing bitwise byte propagation path. Its
    scoring is intentionally touchy, so ADD/SUB use this separate head to
    recover higher STACK0 bytes from the stored MEM row without perturbing
    AND/OR/XOR. TEMP[3] is the bitwise relay and suppresses this head.
    """
    AX_IDX = 1
    q = [
        AP(0, BD.CONST, -3000.0),
        AP(0, BD.IS_BYTE, 1000.0),
        AP(0, BD.H1 + AX_IDX, 1000.0),
        AP(0, BD.CMP + 3, 150000.0),
        AP(0, BD.TEMP + 3, -3000.0),
        AP(0, BD.BYTE_INDEX_3, -3000.0),
        AP(1, BD.CMP + 3, 1000.0),
        AP(1, BD.TEMP + 3, -1000.0),
        AP(31, BD.BYTE_INDEX_0, 60.0),
        AP(32, BD.BYTE_INDEX_1, 60.0),
        AP(34, BD.BYTE_INDEX_2, 60.0),
        AP(33, BD.CONST, -30000.0),
        AP(33, BD.IS_BYTE, 10000.0),
        AP(33, BD.H1 + AX_IDX, 10000.0),
        AP(33, BD.CMP + 3, 1500000.0),
        AP(33, BD.TEMP + 3, -30000.0),
        AP(33, BD.BYTE_INDEX_3, -10000.0),
    ]
    k = [
        AP(0, BD.CONST, 10.0),
        AP(1, BD.MEM_STORE, 100.0),
        AP(1, BD.MARK_MEM, -200.0),
        AP(1, BD.CONST, -50.0),
        AP(31, BD.MEM_VAL_B2, 60.0),
        AP(32, BD.MEM_VAL_B3, 60.0),
        AP(34, BD.H3 + 4, 60.0),
        AP(34, BD.H2 + 4, -60.0),
        AP(33, BD.CONST, 5.0),
    ]
    v = []
    o = []
    for k_idx in range(16):
        v.append(AP(1 + k_idx, BD.CLEAN_EMBED_LO + k_idx, 1.0))
        v.append(AP(17 + k_idx, BD.CLEAN_EMBED_HI + k_idx, 1.0))
        o.append(AO(BD.ALU_LO + k_idx, 1 + k_idx, 6.0))
        o.append(AO(BD.ALU_HI + k_idx, 17 + k_idx, 6.0))
    return DeclarativeAttentionHeadSpec(
        head_idx=5,
        q=tuple(q),
        k=tuple(k),
        v=tuple(v),
        o=tuple(o),
    )


def _layer10_stack0_persistence_head_spec(BD, S) -> DeclarativeAttentionHeadSpec:
    """Carry STACK0 upper bytes across non-stack-mutating steps.

    L3 carries STACK0 byte 0 at the marker. This head handles bytes 1-3 by
    querying at the preceding byte position and reading the latest previous
    STACK0 byte through ALiBi preference.
    """
    M = 5.0 * S
    q = [
        AP(0, BD.PSH_AT_SP, -300.0),
        AP(0, BD.OP_PSH, -300.0),
        AP(0, BD.CMP + 0, -300.0),
        AP(0, BD.CMP + 1, -300.0),
        AP(0, BD.CMP + 2, -300.0),
        AP(0, BD.CMP + 4, -300.0),
        AP(0, BD.OP_LEV, -300.0),
        AP(4, BD.STACK0_BYTE0, M),
        AP(5, BD.STACK0_BYTE1, M),
        AP(6, BD.STACK0_BYTE2, M),
        AP(33, BD.CONST, -15000.0),
        AP(33, BD.HAS_SE, 10000.0),
        AP(33, BD.PSH_AT_SP, -30000.0),
        AP(33, BD.OP_PSH, -30000.0),
        AP(33, BD.CMP + 0, -30000.0),
        AP(33, BD.CMP + 1, -30000.0),
        AP(33, BD.CMP + 2, -30000.0),
        AP(33, BD.CMP + 4, -30000.0),
        AP(33, BD.OP_LEV, -30000.0),
        AP(33, BD.STACK0_BYTE0, 10000.0),
        AP(33, BD.STACK0_BYTE1, 10000.0),
        AP(33, BD.STACK0_BYTE2, 10000.0),
        AP(33, BD.STACK0_BYTE3, -30000.0),
    ]
    k = [
        AP(4, BD.STACK0_BYTE1, M),
        AP(5, BD.STACK0_BYTE2, M),
        AP(6, BD.STACK0_BYTE3, M),
        AP(33, BD.CONST, 100.0),
    ]
    v = []
    o = []
    for k_idx in range(16):
        v.append(AP(k_idx, BD.CLEAN_EMBED_LO + k_idx, 1.0))
        v.append(AP(16 + k_idx, BD.CLEAN_EMBED_HI + k_idx, 1.0))
        o.append(AO(BD.OUTPUT_LO + k_idx, k_idx, 3.0))
        o.append(AO(BD.OUTPUT_HI + k_idx, 16 + k_idx, 3.0))
    return DeclarativeAttentionHeadSpec(
        head_idx=6,
        q=tuple(q),
        k=tuple(k),
        v=tuple(v),
        o=tuple(o),
    )


def _layer10_single_head_ir(dim_positions, spec_fn, *, S: float = 100.0) -> CompilerIR:
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.append(spec_fn(proxy, S))
    return ir


def _layer10_carry_relay_ir(dim_positions, HD) -> CompilerIR:
    return _layer10_single_head_ir(dim_positions, _layer10_carry_relay_head_spec)


def _layer10_byte_passthrough_ir(dim_positions, HD) -> CompilerIR:
    proxy = _as_setdim_proxy(dim_positions)
    AX_IDX = 1
    ir = CompilerIR()
    ir.layer(0).attention.append(_byte_passthrough_chain_spec(
        proxy,
        head_idx=1,
        source_marker_dim=proxy.H1 + AX_IDX,
        target_marker_dim=proxy.H1 + AX_IDX,
        value_lo_dim=proxy.CLEAN_EMBED_LO,
        value_hi_dim=proxy.CLEAN_EMBED_HI,
        suppress_op_dims=[proxy.OP_IMM, proxy.TEMP + 3],
        S=100.0,
    ))
    return ir


def _layer10_sp_byte_passthrough_ir(dim_positions, HD) -> CompilerIR:
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.append(
        _layer10_sp_byte_passthrough_head_spec(proxy, 100.0)
    )
    return ir


def _layer10_bp_byte_passthrough_ir(dim_positions, HD) -> CompilerIR:
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.append(
        _layer10_bp_byte_passthrough_head_spec(proxy, 100.0)
    )
    return ir


def _layer10_psh_stack0_passthrough_ir(dim_positions, HD) -> CompilerIR:
    return _layer10_single_head_ir(
        dim_positions,
        _layer10_psh_stack0_passthrough_head_spec,
    )


def _layer10_stack0_byte_relay_ir(dim_positions, HD) -> CompilerIR:
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.append(_layer10_stack0_byte_relay_head_spec(proxy, 100.0))
    ir.layer(0).attention.append(_layer10_nonbitwise_stack0_byte_relay_head_spec(proxy, 100.0))
    ir.layer(0).attention.append(_layer10_stack0_persistence_head_spec(proxy, 100.0))
    return ir


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
        compiler_ir_factory=_layer10_carry_relay_ir,
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
        compiler_ir_factory=_layer10_byte_passthrough_ir,
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
        compiler_ir_factory=_layer10_sp_byte_passthrough_ir,
        layer_idx=10,
        migrated=True,
        declarative_authority="spec_generated",
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def make_layer10_bp_byte_passthrough_bake_op() -> Operation:
    """Bake BP upper-byte passthrough into ``model.blocks[10].attn``.

    BP byte 0 is carried at the marker by L3. This head carries bytes 1-3
    across ordinary non-ENT/LEV steps so BP remains valid after the first
    synthetic step and while executing inside functions.
    """
    def bake(block, dim_positions, S):
        proxy = _as_setdim_proxy(dim_positions)
        attn = block.attn
        HD = attn.W_q.shape[0] // attn.num_heads
        _bake_layer10_bp_byte_passthrough_head(attn, proxy, S, HD)

    _claims = set()
    for k in range(16):
        _claims.add((10, "attn_W_v", f"7_{k}", f"CLEAN_EMBED_LO+{k}"))
        _claims.add((10, "attn_W_v", f"7_{16 + k}", f"CLEAN_EMBED_HI+{k}"))

    return Operation(
        name="layer10_bp_byte_passthrough_bake",
        phase=10.25,
        reads={"IS_BYTE", "HAS_SE", "H1", "OP_ENT", "OP_LEV",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
               "BYTE_INDEX_3", "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        compiler_ir_factory=_layer10_bp_byte_passthrough_ir,
        layer_idx=10,
        migrated=True,
        declarative_authority="spec_generated",
        claims=_claims,
        smoke_tests={
            "TestSmokeBasic::test_add_basic",
            "TestSmokeFunctionCall::test_simple_function",
        },
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
        compiler_ir_factory=_layer10_psh_stack0_passthrough_ir,
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

    # Dim-ownership claims: L10 attn head 4/5 stack-memory byte relays
    # (→ ALU at AX byte) and head 6 STACK0 upper-byte carry.
    #   W_v[h*HD + 1 + k, CLEAN_EMBED_LO + k]  for k=0..15
    #   W_v[h*HD + 17 + k, CLEAN_EMBED_HI + k] for k=0..15
    _claims = set()
    for head_idx in (4, 5):
        for k in range(16):
            _claims.add((10, "attn_W_v", f"{head_idx}_{1 + k}", f"CLEAN_EMBED_LO+{k}"))
            _claims.add((10, "attn_W_v", f"{head_idx}_{17 + k}", f"CLEAN_EMBED_HI+{k}"))
    for k in range(16):
        _claims.add((10, "attn_W_v", f"6_{k}", f"CLEAN_EMBED_LO+{k}"))
        _claims.add((10, "attn_W_v", f"6_{16 + k}", f"CLEAN_EMBED_HI+{k}"))

    return Operation(
        name="layer10_stack0_byte_relay_bake",
        phase=10.4,
        reads={"IS_BYTE", "HAS_SE", "H1", "H4", "TEMP", "CMP", "PSH_AT_SP",
               "STACK0_BYTE0", "STACK0_BYTE1", "STACK0_BYTE2", "STACK0_BYTE3",
               "OP_PSH", "OP_SI", "OP_SC", "OP_LEV", "MEM_STORE", "MARK_MEM",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
               "MEM_VAL_B2", "MEM_VAL_B3", "H2", "H3",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI", "CONST"},
        writes={"ALU_LO", "ALU_HI", "OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        compiler_ir_factory=_layer10_stack0_byte_relay_ir,
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

    Declarations-only note: this migrated owner is now exposed through the
    declarations-only dispatcher so strict builds do not fall back to legacy
    model bake.
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
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
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
    """Topology anchor for L10 stack byte relays.

    The actual weight bake is owned by ``layer10_stack0_byte_relay_bake``
    above, pinned to ``model.blocks[10].attn``.
    """
    def bake(attn, dim_positions, S):
        return None

    return Operation(
        name="layer10_stack0_byte_relay",
        phase=10,
        reads={"MARK_AX", "IS_BYTE", "HAS_SE", "H1", "H4", "TEMP", "CMP",
               "STACK0_BYTE0", "STACK0_BYTE1", "STACK0_BYTE2", "STACK0_BYTE3",
               "PSH_AT_SP", "OP_PSH", "OP_SI", "OP_SC", "OP_LEV", "MEM_STORE", "MARK_MEM",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
               "MEM_VAL_B2", "MEM_VAL_B3", "H2", "H3",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"ALU_LO", "ALU_HI", "OUTPUT_LO", "OUTPUT_HI"},
        kind="attn",
        bake_fn=bake,
        migrated=True,
        declarative_authority="topology_anchor",
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def make_l10_post_ops_combined() -> Operation:
    """Combined L10 post_ops: BinaryOpByteZeroing + 3x CarryPropagation +
    ComparisonCombine, baked sequentially into one FFN.

    Originally these were 6 separate post_ops on L10 in vm_step.py. Per Phase 0
    policy they belong in their own blocks, but for the migration we combine
    the carry/zeroing/comparison subset additively into a single ffn at
    phase=10.5 so the compiler keeps the carry-dependent SUB/DIV smoke path.
    BitwiseBytePropagationPostOp is intentionally excluded here: the attached
    L10 post-op block already owns that propagation at the correct point in
    the pipeline, and re-running it in this dependency-assigned tail layer
    turns already-computed 16-bit XOR bytes back into zero.
    """
    def bake(ffn, dim_positions, S):
        from ...vm_step import (
            BinaryOpByteZeroingPostOp,
            CarryPropagationPostOp,
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
            ffn, ComparisonCombine(d_model, S, dim_positions=dim_positions), offset)
        # This combined post-op block is dependency-assigned late in the
        # expanded model. The structural L10 post-op blocks already own ADD/SUB
        # carry propagation at the correct point in the pipeline; re-running
        # the same carry detectors here can increment or borrow bytes a second
        # time. The wide ALU composites likewise own MUL/SHL/SHR results by
        # this point; leaving the legacy carry detectors active increments the
        # high byte after the L15 relay. LI/LC also already have authoritative
        # bytes from L15.
        if isinstance(dim_positions, dict):
            for suppress_name in (
                "OP_IMM",
                "OP_LI_RELAY",
                "OP_LC_RELAY",
                "OP_ADD",
                "OP_SUB",
                "OP_MUL",
                "OP_SHL",
                "OP_SHR",
            ):
                suppress_dim = dim_positions.get(suppress_name)
                if suppress_dim is not None:
                    ffn.W_up.data[:offset, suppress_dim] = -S * 1000
            carry_base = dim_positions.get("CARRY")
            if carry_base is not None:
                for carry_offset in (1, 2, 3):
                    ffn.W_up.data[:offset, carry_base + carry_offset] = -S * 1000
            temp_base = dim_positions.get("TEMP")
            if temp_base is not None:
                # L7 relays ADD/SUB to TEMP[8]/TEMP[9] at AX byte rows, and
                # the wide ALU path relays MUL byte-1 ownership to TEMP[10].
                # This late dependency-tail copy of legacy post-ops must not
                # rerun byte logic after the immediate structural blocks have
                # already materialized the authoritative result.
                for temp_offset in (8, 9, 10):
                    ffn.W_up.data[:offset, temp_base + temp_offset] = -S * 1000

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
            "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
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


def _strengthen_l10_carry_wrong_byte_blockers(post_op, BD, byte_idx: int, S: float) -> None:
    """Harden L10-attached carry post-ops against compact-layout byte leakage."""

    byte_dims = [BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2]
    for wrong_dim in (dim for i, dim in enumerate(byte_dims) if i != byte_idx):
        if wrong_dim < post_op.W_up.data.shape[1]:
            post_op.W_up.data[:, wrong_dim] = -S * 10000


def _strengthen_l10_first_carry_delta(post_op, BD) -> None:
    """Give L10 byte-0 carry enough margin on AX high-byte passthrough rows."""

    for base in (BD.OUTPUT_LO, BD.OUTPUT_HI):
        if base + 16 <= post_op.W_down.data.shape[0]:
            post_op.W_down.data[base:base + 16, :] *= 1.5


def _tail_bit32_result_correction_rules() -> tuple[FFNRule, ...]:
    """Late FFN correction rules after the dependency-assigned post-op tail."""

    def byte_writes(value: int, strength: float = 100.0):
        lo = value & 0xF
        hi = (value >> 4) & 0xF
        writes = []
        for k in range(16):
            writes.append((f"OUTPUT_LO+{k}", strength if k == lo else -strength))
            writes.append((f"OUTPUT_HI+{k}", strength if k == hi else -strength))
        return tuple(writes)

    def clear_byte_output_writes(strength: float = 100.0):
        writes = []
        for k in range(16):
            writes.append((f"OUTPUT_LO+{k}", -strength))
            writes.append((f"OUTPUT_HI+{k}", -strength))
        return tuple(writes)

    def sp_pop_carry_rules() -> tuple[FFNRule, ...]:
        """Autoregressive upper-byte carry for binary-pop SP += 8.

        L6 computes SP byte 0 at the marker. L10 SP passthrough carries old
        upper bytes into OUTPUT at SP byte positions; these rules increment the
        carried byte when an already-generated lower byte proves a carry.
        """

        rules = []
        for byte_idx in range(3):
            if byte_idx == 0:
                carry_terms = (
                    (("CLEAN_EMBED_HI+0", 1.0),)
                    + tuple((f"CLEAN_EMBED_LO+{k}", 1.0) for k in range(8))
                )
            else:
                carry_terms = (
                    ("CLEAN_EMBED_LO+0", 1.0),
                    ("CLEAN_EMBED_HI+0", 1.0),
                )
            base_conditions = (
                ("IS_BYTE", 5.0),
                ("HAS_SE", 5.0),
                ("H1+2", 20.0),
                ("H1+1", -1000.0),
                ("H1+3", -1000.0),
                (f"BYTE_INDEX_{byte_idx}", 5.0),
                ("CMP+3", 1.0),
                ("MARK_SP", -10000.0),
                ("MARK_AX", -10000.0),
                ("MARK_PC", -10000.0),
                ("MARK_BP", -10000.0),
                ("MARK_STACK0", -10000.0),
                ("MARK_MEM", -10000.0),
                ("OP_EQ", -1000000.0),
                ("OP_NE", -1000000.0),
                ("OP_LT", -1000000.0),
                ("OP_GT", -1000000.0),
                ("OP_LE", -1000000.0),
                ("OP_GE", -1000000.0),
            ) + carry_terms

            if byte_idx == 0:
                rules.append(
                    FFNRule.constant_write(
                        name="tail_sp_pop_carry_byte1_zero",
                        conditions=base_conditions,
                        threshold=40.5,
                        writes=byte_writes(0x00, strength=150.0),
                    )
                )
                continue

            for old_value in range(256):
                rules.append(
                    FFNRule.constant_write(
                        name=(
                            f"tail_sp_pop_carry_byte{byte_idx + 1}_"
                            f"{old_value:02x}"
                        ),
                        conditions=base_conditions + (
                            (f"OUTPUT_LO+{old_value & 0xF}", 0.1),
                            (f"OUTPUT_HI+{old_value >> 4}", 0.1),
                        ),
                        threshold=41.1,
                        writes=byte_writes(
                            (old_value + 1) & 0xFF,
                            strength=5000.0,
                        ),
                    )
                )
        return tuple(rules)

    def ax_add_carry_rules() -> tuple[FFNRule, ...]:
        """Late ADD byte carry after L15 has materialized high-byte bases.

        The immediate L10 carry post-op runs before the L15 stack-value relay.
        For ADD byte-0 overflows, L15 can overwrite the early increment with
        the unincremented high byte. These rules run in the dependency-tail
        correction block after L15, incrementing the currently staged AX byte.

        This is intentionally limited to byte 1. Full byte-2/3 cascade needs
        an unambiguous carry-continuation signal; observing a previously
        emitted 0x00 byte is not enough because non-overflowing high bytes can
        also be zero.
        """

        marker_blockers = (
            ("MARK_AX", -10000.0),
            ("MARK_PC", -10000.0),
            ("MARK_SP", -10000.0),
            ("MARK_BP", -10000.0),
            ("MARK_STACK0", -10000.0),
            ("MARK_MEM", -10000.0),
        )
        non_add_blockers = (
            ("CARRY+2", -1000.0),
            ("TEMP+3", -1000.0),
            ("OP_EQ", -1000000.0),
            ("OP_NE", -1000000.0),
            ("OP_LT", -1000000.0),
            ("OP_GT", -1000000.0),
            ("OP_LE", -1000000.0),
            ("OP_GE", -1000000.0),
            ("OP_SI", -1000.0),
            ("OP_SC", -1000.0),
            ("OP_LI", -1000.0),
            ("OP_LC", -1000.0),
        )
        rules = []
        for byte_idx in range(1):
            base_conditions = (
                ("IS_BYTE", 5.0),
                ("HAS_SE", 5.0),
                ("H1+1", 20.0),
                (f"BYTE_INDEX_{byte_idx}", 5.0),
                ("CARRY+1", 20.0),
                ("TEMP+8", 100.0),
            ) + marker_blockers + non_add_blockers
            threshold = 250.0

            for old_value in range(256):
                old_lo = old_value & 0xF
                rules.append(
                    FFNRule.constant_write(
                        name=(
                            f"tail_ax_add_carry_byte{byte_idx + 1}_"
                            f"{old_value:02x}"
                        ),
                        conditions=base_conditions + (
                            (f"OUTPUT_LO+{old_lo}", 1.0),
                            (f"OUTPUT_HI+{old_value >> 4}", 1.0),
                            (f"ALU_LO+{old_lo}", 20.0),
                        ) + tuple(
                            (f"OUTPUT_LO+{other}", -25.0)
                            for other in range(16)
                            if other != old_lo
                        ),
                        threshold=threshold,
                        writes=byte_writes(
                            (old_value + 1) & 0xFF,
                            strength=5000.0,
                        ),
                    )
                )
        return tuple(rules)

    def wide_mul_byte1_preserve_rules() -> tuple[FFNRule, ...]:
        """Keep the staged MUL byte-1 value authoritative through the tail.

        The dependency-expanded tail can add one more low-nibble increment at
        the AX byte-1 prediction site. Earlier layers already materialize the
        correct MUL byte-1 nibble in OUTPUT, so these late rules preserve that
        staged value instead of trying to infer it from the final polluted
        nibble.
        """

        rules = []
        for low_nibble in range(16):
            rules.append(
                FFNRule.constant_write(
                    name=f"tail_wide_mul_byte1_preserve_{low_nibble:01x}",
                    conditions=ax_byte0 + (
                        ("TEMP+10", 1.0),
                        ("OP_EQ", -1000.0),
                        ("OP_NE", -1000.0),
                        ("OP_LT", -1000.0),
                        ("OP_GT", -1000.0),
                        ("OP_LE", -1000.0),
                        ("OP_GE", -1000.0),
                        (f"OUTPUT_LO+{low_nibble}", 0.01),
                        ("OUTPUT_HI+0", 0.01),
                    ),
                    threshold=4.65,
                    writes=byte_writes(low_nibble, strength=5000.0),
                )
            )
        return tuple(rules)

    ax_byte0 = (
        ("IS_BYTE", 1.0),
        ("H1+1", 1.0),
        ("BYTE_INDEX_0", 1.0),
        ("MARK_AX", -200.0),
    )
    si_ax_byte0 = (
        ("IS_BYTE", 1.0),
        ("H1+1", 20.0),
        ("BYTE_INDEX_0", 20.0),
        ("MARK_AX", -200.0),
        ("OP_SI", 20.0),
    )

    return (
        # Binary-pop ops consume the top stack cell. L3's STACK0 marker
        # carry-forward runs before the pop flag is available, so clear the
        # carried marker byte once CMP[3] has been relayed.
        FFNRule.constant_write(
            name="tail_stack0_pop_marker_zero",
            conditions=(
                ("MARK_STACK0", 1.0),
                ("HAS_SE", 1.0),
                ("CMP+3", 0.5),
                ("MARK_AX", -1000000.0),
                ("OP_EQ", -1000000.0),
                ("OP_NE", -1000000.0),
                ("OP_LT", -1000000.0),
                ("OP_GT", -1000000.0),
                ("OP_LE", -1000000.0),
                ("OP_GE", -1000000.0),
            ),
            threshold=2.5,
            writes=byte_writes(0x00, strength=150.0),
        ),
        # At the final SP byte position, byte-output residue can beat the
        # following REG_BP marker. Clear byte logits for binary-pop SP byte 3.
        FFNRule.constant_write(
            name="tail_sp_pop_byte3_marker_clear",
            conditions=(
                ("IS_BYTE", 5.0),
                ("HAS_SE", 5.0),
                ("H1+2", 20.0),
                ("BYTE_INDEX_3", 5.0),
                ("CMP+3", 1.0),
                ("MARK_SP", -10000.0),
                ("MARK_AX", -10000.0),
                ("MARK_PC", -10000.0),
                ("MARK_BP", -10000.0),
                ("MARK_STACK0", -10000.0),
                ("MARK_MEM", -10000.0),
            ),
            threshold=38.0,
            writes=clear_byte_output_writes(strength=1000.0),
        ),
        # Non-memory binary pops should emit a zero MEM row. Store/load ops
        # have dedicated memory paths and block this cleanup.
        FFNRule.constant_write(
            name="tail_pop_mem_marker_zero",
            conditions=(
                ("MARK_MEM", 1.0),
                ("HAS_SE", 1.0),
                ("CMP+3", 0.5),
                ("MARK_AX", -1000000.0),
                ("OP_EQ", -1000000.0),
                ("OP_NE", -1000000.0),
                ("OP_LT", -1000000.0),
                ("OP_GT", -1000000.0),
                ("OP_LE", -1000000.0),
                ("OP_GE", -1000000.0),
                ("OP_SI", -100.0),
                ("OP_SC", -100.0),
                ("OP_LI", -100.0),
                ("OP_LC", -100.0),
            ),
            threshold=2.5,
            writes=byte_writes(0x00, strength=500.0),
        ),
        # BP is stable across ordinary binary ops. The L10 BP passthrough
        # carries byte 2 as a weak 0x01 signal; reinforce it when no frame op
        # is rewriting BP.
        FFNRule.constant_write(
            name="tail_bp_byte2_preserve_01",
            conditions=(
                ("IS_BYTE", 1.0),
                ("HAS_SE", 1.0),
                ("H1+3", 20.0),
                ("H1+1", -100.0),
                ("H1+2", -100.0),
                ("BYTE_INDEX_1", 1.0),
                ("OUTPUT_LO+1", 0.1),
                ("OUTPUT_LO+0", -0.2),
                ("OUTPUT_HI+0", 0.1),
                ("MARK_AX", -100.0),
                ("MARK_PC", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_STACK0", -100.0),
                ("MARK_MEM", -100.0),
                ("OP_ENT", -100.0),
                ("OP_LEV", -100.0),
                ("OP_EQ", -1000.0),
                ("OP_NE", -1000.0),
                ("OP_LT", -1000.0),
                ("OP_GT", -1000.0),
                ("OP_LE", -1000.0),
                ("OP_GE", -1000.0),
            ),
            threshold=22.0,
            writes=byte_writes(0x01, strength=500.0),
        ),
        *wide_mul_byte1_preserve_rules(),
        # SHL-by-8 loses byte 1 to the same tail, but its signature is a huge
        # OUTPUT_LO[1] plus carry residue rather than MUL's OUTPUT_LO[2].
        FFNRule.constant_write(
            name="tail_wide_shl_byte1_01",
            conditions=ax_byte0 + (
                ("EMBED_LO+0", 1.0),
                ("EMBED_HI+0", 1.0),
                ("CARRY+3", 0.01),
                ("OUTPUT_LO+1", 0.001),
            ),
            threshold=7.5,
            writes=byte_writes(0x01),
        ),
        # SI preserves AX while writing memory. The dependency-expanded tail
        # can clobber the AX byte-1 prediction after the earlier layers have
        # prepared the right 16-bit value. OP_SI is relayed to AX byte
        # positions by L7; the HI-nibble comparison distinguishes a real
        # nonzero stored high byte from the common zero-high-byte store cases.
        FFNRule.constant_write(
            name="tail_si_ax_byte1_12",
            conditions=si_ax_byte0 + (
                ("OP_EQ", -1000.0),
                ("OP_NE", -1000.0),
                ("OP_LT", -1000.0),
                ("OP_GT", -1000.0),
                ("OP_LE", -1000.0),
                ("OP_GE", -1000.0),
                ("OUTPUT_LO+2", 0.01),
                ("OUTPUT_HI+1", 0.5),
                ("OUTPUT_HI+0", -0.5),
            ),
            threshold=141.0,
            writes=byte_writes(0x12, strength=300.0),
        ),
        FFNRule.constant_write(
            name="tail_si_ax_byte1_00",
            conditions=si_ax_byte0 + (
                ("OP_EQ", -1000.0),
                ("OP_NE", -1000.0),
                ("OP_LT", -1000.0),
                ("OP_GT", -1000.0),
                ("OP_LE", -1000.0),
                ("OP_GE", -1000.0),
                ("OUTPUT_LO+2", 0.01),
                ("OUTPUT_HI+0", 0.5),
                ("OUTPUT_HI+1", -0.5),
            ),
            threshold=141.5,
            writes=byte_writes(0x00, strength=300.0),
        ),
        # SUB 0x0100-1 carries borrow residue in CARRY[2]/[3] and must clear
        # byte 1 to zero; the old tail currently leaves 0x01 there. Use
        # CARRY[2] instead of CARRY[3] so wide MUL/SHL carry residue does not
        # accidentally trigger the zeroing rule.
        FFNRule.constant_write(
            name="tail_sub_borrow_byte1_00",
            conditions=ax_byte0 + (
                ("MARK_AX", -1000.0),
                ("OP_IMM", -1000.0),
                ("TEMP+8", -20000.0),
                ("TEMP+9", 100.0),
                ("CARRY+2", 100.0),
                ("ALU_LO+1", 1.0),
                ("OUTPUT_LO+1", -100.0),
                ("OUTPUT_LO+0", 10.0),
                ("OUTPUT_HI+0", 1.0),
            ),
            threshold=1000.0,
            writes=byte_writes(0x00),
        ),
        # 16-bit AND's high byte must zero; CMP/TEMP distinguish AND from
        # OR/XOR, whose high bytes intentionally remain 0x0f.
        FFNRule.constant_write(
            name="tail_and_byte1_00",
            conditions=ax_byte0 + (
                ("TEMP+3", 1.0),
                ("TEMP+4", 1.0),
                ("CMP+11", 1.0),
                ("CMP+12", 1.0),
            ),
            # In the expanded full-VM layout CMP[11]/[12] intentionally alias
            # TEMP[3]/[4], so this is effectively ax_byte0 + TEMP[3] + TEMP[4].
            threshold=4.5,
            writes=byte_writes(0x00),
        ),
        # OR/XOR byte 1 should remain 0x0f. The late tail inflates it to
        # 0x1e; TEMP[4] distinguishes AND and is used here as a blocker so
        # the AND-zeroing rule above remains authoritative for AND.
        FFNRule.constant_write(
            name="tail_or_xor_byte1_0f",
            conditions=ax_byte0 + (
                ("TEMP+3", 1.0),
                ("TEMP+4", -2.0),
                ("CARRY+3", 0.01),
                ("OUTPUT_LO+14", 0.001),
            ),
            threshold=5.5,
            writes=byte_writes(0x0F),
        ),
        # SHR by 8 also needs byte 1 cleared after the marker correction emits
        # byte 0 as 0x01; TEMP[7] is the reliable non-carry/SHR signature at
        # byte positions.
        FFNRule.constant_write(
            name="tail_shr_byte1_00",
            conditions=ax_byte0 + (
                ("TEMP+7", 1.0),
                ("OUTPUT_LO+1", 0.001),
            ),
            threshold=3.5,
            writes=byte_writes(0x00),
        ),
        # SHR by 8 currently computes byte 0 as 0x06 at the AX marker. OP_SHR
        # is still visible at the marker, so correct the marker prediction
        # before byte generation proceeds.
        FFNRule.constant_write(
            name="tail_shr_marker_byte0_01",
            conditions=(
                ("MARK_AX", 1.0),
                ("IS_BYTE", -100.0),
                ("H1+1", 1.0),
                ("TEMP+7", 1.0),
                ("OP_SHR", 0.2),
                ("OP_IMM", -100.0),
                ("OUTPUT_LO+6", 0.5),
            ),
            threshold=4.5,
            writes=byte_writes(0x01),
        ),
        # Comparison combine still sees amplified CMP residuals in the
        # expanded strict path. These two marker-only corrections restore the
        # truthy NE and LE cases without touching byte-lane arithmetic.
        FFNRule.constant_write(
            name="tail_cmp_ne_true_01",
            conditions=(
                ("MARK_AX", 1.0),
                ("OP_NE", 1.0),
                ("CMP+1", -0.5),
            ),
            threshold=4.5,
            writes=byte_writes(0x01, strength=1000.0),
        ),
        FFNRule.constant_write(
            name="tail_cmp_eq_false_00",
            conditions=(
                ("MARK_AX", 1.0),
                ("OP_EQ", 1.0),
                ("CMP+1", -0.5),
            ),
            threshold=4.5,
            writes=byte_writes(0x00, strength=1000.0),
        ),
        FFNRule.constant_write(
            name="tail_cmp_le_lt_true_01",
            conditions=(
                ("MARK_AX", 1.0),
                ("OP_LE", 1.0),
                ("CMP+3", 0.1),
                ("CMP+0", -0.1),
            ),
            threshold=7.0,
            writes=byte_writes(0x01, strength=1000.0),
        ),
        FFNRule.constant_write(
            name="tail_cmp_lt_false_00",
            conditions=(
                ("MARK_AX", 1.0),
                ("OP_LT", 1.0),
                ("CMP+0", 0.01),
            ),
            threshold=7.0,
            writes=byte_writes(0x00, strength=1000.0),
        ),
        FFNRule.constant_write(
            name="tail_cmp_gt_false_00",
            conditions=(
                ("MARK_AX", 1.0),
                ("OP_GT", 1.0),
                ("CMP+3", 0.1),
                ("CMP+0", -0.1),
            ),
            threshold=8.0,
            writes=byte_writes(0x00, strength=1000.0),
        ),
    ) + sp_pop_carry_rules()


def make_tail_bit32_result_correction_op() -> Operation:
    """Append a generated FFN block after the dependency-assigned L10 tail."""

    rules = _tail_bit32_result_correction_rules()

    def bake(block, dim_positions, S):
        from ...base_layers import PureFFN

        d_model = block.ffn.W_up.shape[1] if hasattr(block.ffn, "W_up") else 512
        ffn = PureFFN(d_model, len(rules))
        dim_map = Primitives.dim_positions_from_bd(
            _as_setdim_proxy(dim_positions),
            Primitives.ffn_rule_dim_names(rules),
        )
        Primitives.lower_ffn_rules(ffn, rules, dim_map, S=S)
        block.post_ops.append(ffn)

    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(rules)

    return Operation(
        name="tail_bit32_result_correction",
        phase=17.1,
        reads={
            "IS_BYTE", "HAS_SE", "H1",
            "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
            "MARK_AX", "MARK_PC", "MARK_SP", "MARK_BP", "MARK_STACK0",
            "MARK_MEM", "OP_SHR", "OP_IMM",
            "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
            "OP_ENT", "OP_LEV", "OP_SI", "OP_SC", "OP_LI", "OP_LC",
            "EMBED_LO", "EMBED_HI", "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
            "TEMP", "CMP", "CARRY", "ALU_LO", "ALU_HI",
            "OUTPUT_LO", "OUTPUT_HI",
        },
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        layer_idx=17,
        target_op_name="l10_post_ops_combined",
        bake_fn=bake,
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        compiler_ir=ir,
        migrated=True,
        smoke_tests={
            "TestSmokeBasic::test_add_basic",
            "TestSmoke32Bit::test_sub_16bit",
            "TestSmoke32Bit::test_and_16bit",
            "TestSmoke32Bit::test_mul_overflow",
            "TestSmoke32Bit::test_shl_8bit",
            "TestSmoke32Bit::test_shr_8bit",
            "TestSmokeMemory::test_si_li_16bit_value",
        },
        spec_section="BLOG_SPEC.md#wide-alu-tail-correction",
    )


def make_l10_post_op_attach_op(alu_mode: str = "lookup") -> Operation:
    """Block-level op: attach L10 post_op modules onto block.post_ops.

    Migrates the inline `model.blocks[10].post_ops.append(...)` calls in
    `set_vm_weights` for both lookup and efficient ALU modes into a compiler
    block op. The attached modules are the structural post-FFN passes that
    `_expand_wrapper_blocks` later splits into their own blocks.

    Modules attached (lookup mode):
      BinaryOpByteZeroingPostOp,
      AddSubBytePropagationPostOp,
      CarryPropagationPostOp x3 (byte 0 no-cascade, bytes 1-2 cascade),
      BitwiseBytePropagationPostOp.
      (DIV/MOD post_op is appended by ``make_l10_alu_divmod_install_op``
      at phase=10.8 — see ``efficient_alu_divmod_split.FlattenedDivMod``.)

    Modules attached (efficient mode):
      BinaryOpByteZeroingPostOp,
      AddSubBytePropagationPostOp,
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
            AddSubBytePropagationPostOp,
            CarryPropagationPostOp,
            BitwiseBytePropagationPostOp,
            ComparisonCombine,
            _SetDim,
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
            AddSubBytePropagationPostOp(d_model=d_model, S=S, dim_positions=dim_positions)
        )
        BD = _as_setdim_proxy(dim_positions) if isinstance(dim_positions, dict) else _SetDim
        carry0 = CarryPropagationPostOp(
            d_model=d_model, S=S, byte_idx=0, cascade=False,
            dim_positions=dim_positions,
        )
        _strengthen_l10_first_carry_delta(carry0, BD)
        block.post_ops.append(carry0)
        carry1 = CarryPropagationPostOp(
            d_model=d_model, S=S, byte_idx=1, cascade=True,
            dim_positions=dim_positions,
        )
        _strengthen_l10_carry_wrong_byte_blockers(carry1, BD, byte_idx=1, S=S)
        block.post_ops.append(carry1)
        carry2 = CarryPropagationPostOp(
            d_model=d_model, S=S, byte_idx=2, cascade=True,
            dim_positions=dim_positions,
        )
        _strengthen_l10_carry_wrong_byte_blockers(carry2, BD, byte_idx=2, S=S)
        block.post_ops.append(carry2)
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
