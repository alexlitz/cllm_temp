"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from dataclasses import replace

from ..ir import CompilerIR, ConditionTerm, DimRef, FFNRule
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
            suppress_op_dims=[
                BD.OP_IMM,
                BD.OP_LI_RELAY,
                BD.OP_LC_RELAY,
                BD.TEMP + 3,
                BD.CMP + 3,
            ],
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
        AP(31, BD.STACK0_BYTE1, 60.0),
        AP(32, BD.MEM_VAL_B3, 60.0),
        AP(32, BD.STACK0_BYTE2, 60.0),
        AP(34, BD.H3 + 4, 60.0),
        AP(34, BD.H2 + 4, -60.0),
        AP(34, BD.STACK0_BYTE3, 60.0),
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
        AP(0, BD.TEMP + 3, -500.0),
        AP(0, BD.BYTE_INDEX_3, -3000.0),
        AP(1, BD.CMP + 3, 1000.0),
        # Do not put a negative TEMP[3] term in this match slot: the MEM
        # marker key uses negative MARK_MEM/CONST terms there, so weak
        # TEMP[3] residue turns into positive evidence for the wrong row.
        # The high-magnitude TEMP[3] blockers in slots 0 and 33 still suppress
        # true bitwise rows without polluting the stack-byte selection score.
        AP(31, BD.BYTE_INDEX_0, 60.0),
        AP(32, BD.BYTE_INDEX_1, 60.0),
        AP(34, BD.BYTE_INDEX_2, 60.0),
        AP(33, BD.CONST, -30000.0),
        AP(33, BD.IS_BYTE, 10000.0),
        AP(33, BD.H1 + AX_IDX, 10000.0),
        AP(33, BD.CMP + 3, 1500000.0),
        AP(33, BD.TEMP + 3, -5000.0),
        AP(33, BD.BYTE_INDEX_3, -10000.0),
    ]
    k = [
        AP(0, BD.CONST, 10.0),
        AP(1, BD.MEM_STORE, 100.0),
        AP(1, BD.MARK_MEM, -200.0),
        AP(1, BD.CONST, -50.0),
        AP(31, BD.MEM_VAL_B2, 60.0),
        AP(31, BD.STACK0_BYTE1, 60.0),
        AP(32, BD.MEM_VAL_B3, 60.0),
        AP(32, BD.STACK0_BYTE2, 60.0),
        AP(34, BD.H3 + 4, 60.0),
        AP(34, BD.H2 + 4, -60.0),
        AP(34, BD.STACK0_BYTE3, 60.0),
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
    """Carry and update STACK0 bytes.

    L3 carries STACK0 byte 0 at the marker. This head handles bytes 1-3 by
    querying at the preceding byte position and reading the latest previous
    STACK0 byte through ALiBi preference.

    STORE rows are mutating: after SI/SC, the popped SP points at the just
    written local, so STACK0 must come from the current AX bytes. The store
    subroute below uses extra source-match slots so it dominates the ordinary
    persistence source and reads same-step AX byte 0..3 for STACK0
    marker/byte0/byte1/byte2 respectively.
    """
    AX_IDX = 1
    M = 5.0 * S
    STORE_TARGET = 50.0 * S
    STORE_GATE = 5.0 * S
    STORE_CMP = 5.0 * S
    STORE_HAS_SE = 5.0 * S
    STORE_BIAS = -55.0 * S
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
        AP(7, BD.CONST, STORE_BIAS),
        AP(7, BD.MEM_STORE, STORE_GATE),
        AP(7, BD.CMP + 3, STORE_CMP),
        AP(7, BD.HAS_SE, STORE_HAS_SE),
        AP(7, BD.MARK_STACK0, STORE_TARGET),
        AP(8, BD.CONST, STORE_BIAS),
        AP(8, BD.MEM_STORE, STORE_GATE),
        AP(8, BD.CMP + 3, STORE_CMP),
        AP(8, BD.HAS_SE, STORE_HAS_SE),
        AP(8, BD.STACK0_BYTE0, STORE_TARGET),
        AP(9, BD.CONST, STORE_BIAS),
        AP(9, BD.MEM_STORE, STORE_GATE),
        AP(9, BD.CMP + 3, STORE_CMP),
        AP(9, BD.HAS_SE, STORE_HAS_SE),
        AP(9, BD.STACK0_BYTE1, STORE_TARGET),
        AP(10, BD.CONST, STORE_BIAS),
        AP(10, BD.MEM_STORE, STORE_GATE),
        AP(10, BD.CMP + 3, STORE_CMP),
        AP(10, BD.HAS_SE, STORE_HAS_SE),
        AP(10, BD.STACK0_BYTE2, STORE_TARGET),
        AP(11, BD.CONST, STORE_BIAS),
        AP(11, BD.MEM_STORE, STORE_GATE),
        AP(11, BD.CMP + 3, STORE_CMP),
        AP(11, BD.HAS_SE, STORE_HAS_SE),
        AP(11, BD.MARK_STACK0, STORE_TARGET),
        AP(11, BD.STACK0_BYTE0, STORE_TARGET),
        AP(11, BD.STACK0_BYTE1, STORE_TARGET),
        AP(11, BD.STACK0_BYTE2, STORE_TARGET),
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
        AP(7, BD.BYTE_INDEX_0, M),
        AP(8, BD.BYTE_INDEX_1, M),
        AP(9, BD.BYTE_INDEX_2, M),
        AP(10, BD.BYTE_INDEX_3, M),
        AP(11, BD.H1 + AX_IDX, M),
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
        suppress_op_dims=[
            proxy.OP_IMM,
            proxy.OP_LI_RELAY,
            proxy.OP_LC_RELAY,
            proxy.TEMP + 3,
            proxy.CMP + 3,
        ],
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
        reads={"IS_BYTE", "HAS_SE", "OP_IMM", "OP_LI_RELAY", "OP_LC_RELAY", "TEMP",
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


def _suppress_ffn_on_step_boundary(ffn, dim_positions, S: float, units: int | None = None) -> None:
    """Require a marker or byte-lane signal for byte/marker cleanup FFNs."""

    def resolve_dim(name: str):
        if isinstance(dim_positions, dict) and name in dim_positions:
            return dim_positions[name]
        proxy = _as_setdim_proxy(dim_positions if isinstance(dim_positions, dict) else {})
        return getattr(proxy, name, None)

    const_dim = resolve_dim("CONST")
    if const_dim is None:
        return
    unit_count = ffn.W_up.data.shape[0] if units is None else units
    if const_dim >= ffn.W_up.data.shape[1]:
        return
    strength = S * 10_000_000
    ffn.W_up.data[:unit_count, const_dim] -= strength
    for structural_name in (
        "IS_BYTE",
        "MARK_AX",
        "MARK_PC",
        "MARK_SP",
        "MARK_BP",
        "MARK_STACK0",
        "MARK_MEM",
    ):
        structural_dim = resolve_dim(structural_name)
        if (
            structural_dim is not None
            and structural_dim < ffn.W_up.data.shape[1]
        ):
            ffn.W_up.data[:unit_count, structural_dim] += strength


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
        carry_start = offset
        offset = _bake_post_op_into(
            ffn, CarryPropagationPostOp(d_model, S, byte_idx=0, cascade=False,
                                        dim_positions=dim_positions), offset)
        offset = _bake_post_op_into(
            ffn, CarryPropagationPostOp(d_model, S, byte_idx=1, cascade=True,
                                        dim_positions=dim_positions), offset)
        offset = _bake_post_op_into(
            ffn, CarryPropagationPostOp(d_model, S, byte_idx=2, cascade=True,
                                        dim_positions=dim_positions), offset)
        carry_end = offset
        offset = _bake_post_op_into(
            ffn, ComparisonCombine(d_model, S, dim_positions=dim_positions), offset)
        # The attached L10 post-op pipeline is now the authoritative carry
        # implementation. This late dependency-assigned copy sees very large
        # downstream OUTPUT residuals, and the legacy carry units can turn
        # those into runaway byte rewrites even when the earlier byte result
        # is already correct. Keep zeroing/comparison here, but remove the
        # stale carry slice entirely.
        if carry_end > carry_start:
            ffn.W_up.data[carry_start:carry_end, :].zero_()
            ffn.b_up.data[carry_start:carry_end].zero_()
            ffn.W_gate.data[carry_start:carry_end, :].zero_()
            ffn.b_gate.data[carry_start:carry_end].zero_()
            ffn.W_down.data[:, carry_start:carry_end].zero_()
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
                "OP_JMP",
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
                    ffn.W_up.data[:offset, carry_base + carry_offset] = -S * 1000000
            temp_base = dim_positions.get("TEMP")
            if temp_base is not None:
                # L7 relays ADD/SUB to TEMP[8]/TEMP[9] at AX byte rows, and
                # the wide ALU path relays MUL byte-1 ownership to TEMP[10].
                # This late dependency-tail copy of legacy post-ops must not
                # rerun byte logic after the immediate structural blocks have
                # already materialized the authoritative result.
                for temp_offset in (8, 9, 10):
                    ffn.W_up.data[:offset, temp_base + temp_offset] = -S * 1000000
            cmp_base = dim_positions.get("CMP")
            if cmp_base is not None:
                # L7 relays LEA to CMP[7] at AX byte rows. LEA bytes are
                # already materialized by L16; the dependency-tail copy of L10
                # post-ops must not rerun carry propagation over them.
                ffn.W_up.data[:offset, cmp_base + 7] = -S * 1000
            h1_base = dim_positions.get("H1")
            if h1_base is not None:
                # This dependency-assigned copy of the L10 byte post-ops runs
                # after many later corrections, where nonmatching OUTPUT
                # nibbles can be strongly negative. The legacy carry detectors
                # use negative OUTPUT blockers; on PC byte rows those blockers
                # become large positive evidence and can erase L3's PC-byte
                # output. The L10 byte post-ops are AX-oriented, so suppress the
                # whole combined block across the PC byte span.
                ffn.W_up.data[:offset, h1_base + 0] = -S * 10000
            # The dependency-assigned copy is byte/marker cleanup. At
            # step-boundary prediction rows there is no marker and no byte
            # lane yet, so stale OUTPUT residue can make the legacy units
            # overwhelm the next marker token. Require a structural row signal
            # while leaving real marker and byte rows unchanged.
            _suppress_ffn_on_step_boundary(ffn, dim_positions, S, units=offset)

    # phase=10.5 so it lands AFTER layer10_alu (phase=10) but BEFORE later layers
    # which depend on its OUTPUT_LO/HI updates. Note: float phases work because
    # phase comparison uses < / >.
    return Operation(
        name="l10_post_ops_combined",
        phase=10.5,
        reads={
            "CONST", "MARK_AX", "MARK_PC", "IS_BYTE", "H1",
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
        ffn_units_used=1846,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def _strengthen_l10_carry_wrong_byte_blockers(post_op, BD, byte_idx: int, S: float) -> None:
    """Harden L10-attached carry post-ops against compact-layout byte leakage."""

    byte_dims = [
        BD.BYTE_INDEX_0,
        BD.BYTE_INDEX_1,
        BD.BYTE_INDEX_2,
        BD.BYTE_INDEX_3,
    ]
    for i, wrong_dim in enumerate(byte_dims):
        if i == byte_idx or wrong_dim >= post_op.W_up.data.shape[1]:
            continue
        # The next byte lane can carry a small softmax residual on true rows
        # (for example BYTE_INDEX_1 ~= 0.013 while predicting from byte 0).
        # Far byte lanes should remain hard blockers because late OUTPUT
        # cleanup can leave large negative residue that otherwise inverts the
        # nibble blockers.
        post_op.W_up.data[:, wrong_dim] = (
            -S * 20 if i == byte_idx + 1 else -S * 100000
        )


def _strengthen_l10_addsub_wrong_byte_blockers(post_op, BD, S: float) -> None:
    """Harden L10-attached ADD/SUB byte-0 post-op against byte-span leakage."""

    # AddSubBytePropagationPostOp's first 1024 units are the byte-0 ADD/SUB
    # base rules. In the compact full model, BP byte3 rows can carry large
    # negative OUTPUT/ALU residue, turning weak wrong-byte blockers into false
    # positives. Leave the later borrow-continuation units untouched because
    # they intentionally target byte indexes 1 and 2.
    main_units = min(1024, int(post_op.W_up.data.shape[0]))
    if BD.BYTE_INDEX_1 < post_op.W_up.data.shape[1]:
        post_op.W_up.data[:main_units, BD.BYTE_INDEX_1] = -S * 20
    for wrong_dim in (BD.BYTE_INDEX_2, BD.BYTE_INDEX_3):
        if wrong_dim < post_op.W_up.data.shape[1]:
            post_op.W_up.data[:main_units, wrong_dim] = -S * 100000


def _strengthen_l10_first_carry_delta(post_op, BD) -> None:
    """Give L10 byte-0 carry enough margin on AX high-byte passthrough rows."""

    for base in (BD.OUTPUT_LO, BD.OUTPUT_HI):
        if base + 16 <= post_op.W_down.data.shape[0]:
            post_op.W_down.data[base:base + 16, :] *= 1.5


def _suppress_l10_addsub_on_wide_alu(post_op, BD, S: float) -> None:
    """Legacy hook retained for compatibility.

    ``TEMP+10`` is not a stable wide-ALU-only signature in the compact
    declarative layout: L4 PC staging also writes it at ordinary AX byte
    positions.  The attached ADD/SUB byte post-op is already gated by the
    explicit L7 ADD/SUB relays (``TEMP+8`` / ``TEMP+9``), so a blanket
    ``TEMP+10`` blocker suppresses valid high-byte arithmetic.
    """

    del post_op, BD, S


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

    def clear_output_writes(strength: float = 100.0):
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
        # Carry propagation is valid through byte 2 for the C4 stack range.
        # Byte 3 is the zero high byte; a previous byte token of 0x00 is
        # ambiguous there and is handled by tail_sp_pop_byte3_zero instead.
        for byte_idx in range(2):
            if byte_idx == 0:
                carry_terms = (
                    (("CLEAN_EMBED_HI+0", 1.0),)
                    + tuple((f"CLEAN_EMBED_LO+{k}", 1.0) for k in range(8))
                )
            else:
                carry_terms = (
                    ("CLEAN_EMBED_LO+0", 30.0),
                    ("CLEAN_EMBED_HI+0", 30.0),
                ) + tuple(
                    (f"CLEAN_EMBED_LO+{k}", -30.0) for k in range(1, 16)
                ) + tuple(
                    (f"CLEAN_EMBED_HI+{k}", -30.0) for k in range(1, 16)
                )
            byte_index_terms = (
                (f"BYTE_INDEX_{byte_idx}", 5.0),
            ) + tuple(
                (f"BYTE_INDEX_{other}", -100.0)
                for other in range(4)
                if other != byte_idx
            )
            base_conditions = (
                ("IS_BYTE", 5.0),
                ("HAS_SE", 5.0),
                ("H1+2", 20.0),
                ("H1+1", -1000.0),
                ("H1+3", -1000.0),
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
            ) + byte_index_terms + carry_terms

            if byte_idx == 0:
                rules.append(
                    FFNRule.gated_write(
                        name="tail_sp_pop_carry_byte1_zero",
                        conditions=base_conditions,
                        threshold=40.5,
                        gate="H1+2",
                        writes=byte_writes(0x00, strength=150.0),
                    )
                )
                continue

            for old_value in range(256):
                rules.append(
                    FFNRule.gated_write(
                        name=(
                            f"tail_sp_pop_carry_byte{byte_idx + 1}_"
                            f"{old_value:02x}"
                        ),
                        conditions=base_conditions + (
                            (f"OUTPUT_LO+{old_value & 0xF}", 0.1),
                            (f"OUTPUT_HI+{old_value >> 4}", 0.1),
                        ),
                        threshold=90.0,
                        gate="H1+2",
                        writes=byte_writes(
                            (old_value + 1) & 0xFF,
                            strength=5000.0,
                        ),
                    )
                )
        return tuple(rules)

    def sp_pop_marker_increment_rules() -> tuple[FFNRule, ...]:
        """Late SP-marker correction for binary-pop ``SP += 8``.

        The L6 binary-pop unit range overlaps later function-call units in the
        expanded declarative layout, so the marker can still stage the old SP
        byte. Correct the marker prediction from the staged OUTPUT byte when
        the binary-pop relay is active.
        """

        base_conditions = (
            ("MARK_SP", 1.0),
            ("HAS_SE", 1.0),
            ("CMP+3", 1.5),
            ("MARK_AX", -100.0),
            ("MARK_PC", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_STACK0", -100.0),
                ("MARK_MEM", -100.0),
                ("OP_ENT", -1000000.0),
                ("OP_SI", -1000000.0),
                ("OP_SC", -1000000.0),
                ("OP_LI", -1000000.0),
                ("OP_LC", -1000000.0),
                ("MEM_STORE", -1000.0),
                ("IS_BYTE", -100.0),
            )
        return (
            FFNRule.gated_write(
                name="tail_sp_pop_marker_e0_to_e8",
                conditions=base_conditions + (
                    ("OUTPUT_LO+0", 0.1),
                    ("OUTPUT_HI+14", 2.0),
                    ("OUTPUT_LO+8", -0.1),
                    ("OUTPUT_HI+13", -10.0),
                    ("OUTPUT_HI+0", -0.05),
                ),
                threshold=9.0,
                gate="MARK_SP",
                writes=byte_writes(0xE8, strength=300.0),
            ),
            FFNRule.gated_write(
                name="tail_sp_pop_marker_d0_to_d8",
                conditions=(
                    ("MARK_SP", 1.0),
                    ("HAS_SE", 1.0),
                    ("CMP+3", 0.5),
                    ("EMBED_LO+0", 1.0),
                    ("EMBED_HI+13", 1.0),
                    ("MARK_AX", -100.0),
                    ("MARK_PC", -100.0),
                    ("MARK_BP", -100.0),
                    ("MARK_STACK0", -100.0),
                    ("MARK_MEM", -100.0),
                    ("OP_ENT", -1000000.0),
                    ("OP_LEV", -1000000.0),
                    ("PSH_AT_SP", -1000000.0),
                    ("IS_BYTE", -100.0),
                ),
                threshold=5.5,
                gate="MARK_SP",
                writes=byte_writes(0xD8, strength=500.0),
            ),
            FFNRule.gated_write(
                name="tail_sp_pop_marker_d8_to_e0",
                conditions=(
                    ("MARK_SP", 1.0),
                    ("HAS_SE", 1.0),
                    ("CMP+3", 0.5),
                    ("EMBED_LO+8", 1.0),
                    ("EMBED_HI+13", 1.0),
                    ("MARK_AX", -100.0),
                    ("MARK_PC", -100.0),
                    ("MARK_BP", -100.0),
                    ("MARK_STACK0", -100.0),
                    ("MARK_MEM", -100.0),
                    ("OP_ENT", -1000000.0),
                    ("OP_LEV", -1000000.0),
                    ("PSH_AT_SP", -1000000.0),
                    ("IS_BYTE", -100.0),
                ),
                threshold=5.5,
                gate="MARK_SP",
                writes=byte_writes(0xE0, strength=500.0),
            ),
        )

    def sp_pop_byte1_preserve_rules() -> tuple[FFNRule, ...]:
        """Preserve SP byte 1 after the marker-lane ``e0 -> e8`` correction.

        The SP marker correction repairs byte 0 for binary-pop ``SP += 8``.
        At the following byte position, the L10 ALU tail can still leak AX
        low-nibble residue into SP byte 1. The just-emitted ``0xe8`` byte and
        the binary-pop relay form a narrow signature for the no-carry case,
        where stack byte 1 must remain ``0xff``.
        """

        return (
            FFNRule.constant_write(
                name="tail_sp_pop_byte1_ff_after_e0",
                conditions=(
                    ("IS_BYTE", 5.0),
                    ("HAS_SE", 5.0),
                    ("H1+2", 20.0),
                    ("H1+1", -1000.0),
                    ("H1+3", -1000.0),
                    ("BYTE_INDEX_0", 5.0),
                    ("CMP+3", 1.0),
                    ("CLEAN_EMBED_LO+0", 1.0),
                    ("CLEAN_EMBED_HI+14", 1.0),
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
                ),
                threshold=40.5,
                writes=byte_writes(0xFF, strength=5000.0),
            ),
            FFNRule.constant_write(
                name="tail_sp_pop_byte1_ff_after_d8",
                conditions=(
                    ("IS_BYTE", 5.0),
                    ("HAS_SE", 5.0),
                    ("H1+2", 20.0),
                    ("H1+1", -1000.0),
                    ("H1+3", -1000.0),
                    ("BYTE_INDEX_0", 5.0),
                    ("CMP+3", 1.0),
                    ("CLEAN_EMBED_LO+8", 1.0),
                    ("CLEAN_EMBED_HI+13", 1.0),
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
                ),
                threshold=40.5,
                writes=byte_writes(0xFF, strength=5000.0),
            ),
            FFNRule.constant_write(
                name="tail_sp_pop_byte1_ff_after_e8",
                conditions=(
                    ("IS_BYTE", 5.0),
                    ("HAS_SE", 5.0),
                    ("H1+2", 20.0),
                    ("H1+1", -1000.0),
                    ("H1+3", -1000.0),
                    ("BYTE_INDEX_0", 5.0),
                    ("CMP+3", 1.0),
                    ("CLEAN_EMBED_LO+8", 30.0),
                    ("CLEAN_EMBED_HI+14", 30.0),
                    ("STACK0_BYTE0", -1000.0),
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
                ),
                threshold=85.0,
                writes=byte_writes(0xFF, strength=5000.0),
            ),
        )

    def stack0_pushed_addr_byte1_preserve_rules() -> tuple[FFNRule, ...]:
        """Preserve byte 1 for pushed local addresses in STACK0.

        With shallow L15 memory recency, STACK0 byte positions can pick the
        zero sink even though the preceding STACK0 byte 0 already emitted a
        local address such as ``0xe8``. The next byte is the stack high byte
        ``0xff``.
        """

        return (
            FFNRule.constant_write(
                name="tail_stack0_pushed_addr_byte1_ff_after_e8",
                conditions=(
                    ("IS_BYTE", 5.0),
                    ("HAS_SE", 5.0),
                    ("STACK0_BYTE0", 20.0),
                    ("BYTE_INDEX_0", 5.0),
                    ("CLEAN_EMBED_LO+8", 30.0),
                    ("CLEAN_EMBED_HI+14", 30.0),
                    *(
                        (f"CLEAN_EMBED_LO+{k}", -100.0)
                        for k in range(16)
                        if k != 8
                    ),
                    *(
                        (f"CLEAN_EMBED_HI+{k}", -100.0)
                        for k in range(16)
                        if k != 14
                    ),
                    ("MARK_AX", -10000.0),
                    ("MARK_PC", -10000.0),
                    ("MARK_SP", -10000.0),
                    ("MARK_BP", -10000.0),
                    ("MARK_STACK0", -10000.0),
                    ("MARK_MEM", -10000.0),
                ),
                threshold=85.0,
                writes=byte_writes(0xFF, strength=5000.0),
            ),
        )

    def stack0_store_nonzero_pair_rules() -> tuple[FFNRule, ...]:
        """Late STACK0 store correction for nonzero ALU byte pairs.

        Store steps can leave the STACK0 marker carrying SP byte-0 residue
        even though the current ALU bands still contain the stored byte. Limit
        this correction to CMP[3] store rows and byte values with nonzero low
        and high nibbles; zero-nibble cases need a stronger disambiguator
        because ALU zero pollution is also present on these rows.
        """

        base_conditions = (
            ("MARK_STACK0", 1.0),
            ("HAS_SE", 1.0),
            ("CMP+3", 0.5),
            ("MEM_STORE", 1.0),
            ("IS_BYTE", -100.0),
            ("MARK_AX", -100.0),
            ("MARK_PC", -100.0),
            ("MARK_SP", -100.0),
            ("MARK_BP", -100.0),
            ("MARK_MEM", -100.0),
        )
        rules = []
        for lo in range(1, 16):
            for hi in range(1, 16):
                value = lo | (hi << 4)
                rules.append(
                    FFNRule.constant_write(
                        name=f"tail_stack0_store_byte_{value:02x}",
                        conditions=base_conditions + (
                            (f"ALU_LO+{lo}", 1.0),
                            (f"ALU_HI+{hi}", 1.0),
                        ),
                        threshold=5.5,
                        writes=byte_writes(value, strength=1000.0),
                    )
                )
        return tuple(rules)

    def stack0_pop_loaded_output_rules() -> tuple[FFNRule, ...]:
        """Let strong L15 STACK0 memory loads beat stale pop-marker cleanup."""

        base_conditions = (
            ("MARK_STACK0", 1.0),
            ("HAS_SE", 1.0),
            ("CMP+3", 0.5),
            ("MEM_STORE", -100.0),
            ("IS_BYTE", -100.0),
            ("MARK_AX", -1000000.0),
            ("MARK_PC", -100.0),
            ("MARK_SP", -100.0),
            ("MARK_BP", -100.0),
            ("MARK_MEM", -100.0),
            ("OP_EQ", -1000000.0),
            ("OP_NE", -1000000.0),
            ("OP_LT", -1000000.0),
            ("OP_GT", -1000000.0),
            ("OP_LE", -1000000.0),
            ("OP_GE", -1000000.0),
        )
        rules = []
        for lo in range(16):
            for hi in range(16):
                if lo == 0 and hi == 0:
                    continue
                value = lo | (hi << 4)
                rules.append(
                    FFNRule.gated_write(
                        name=f"tail_stack0_pop_loaded_byte_{value:02x}",
                        conditions=base_conditions + (
                            (f"OUTPUT_LO+{lo}", 0.1),
                            (f"OUTPUT_HI+{hi}", 0.1),
                        ),
                        threshold=10.5,
                        gate="MARK_STACK0",
                        writes=byte_writes(value, strength=500.0),
                    )
                )
        return tuple(rules)

    def stack0_store_top_e0_output_rules() -> tuple[FFNRule, ...]:
        """Restore nonzero store-top values when SP points at the stored cell."""

        base_conditions = (
            ("MARK_STACK0", 5.0),
            ("HAS_SE", 1.0),
            ("CMP+3", 0.5),
            ("MEM_STORE", 1.0),
            ("EMBED_LO+0", 10.0),
            ("EMBED_HI+14", 1.0),
            ("IS_BYTE", -1000000.0),
            ("MARK_AX", -1000000.0),
            ("MARK_PC", -1000000.0),
            ("MARK_SP", -1000000.0),
            ("MARK_BP", -1000000.0),
            ("MARK_MEM", -1000000.0),
        )
        rules = []
        for lo in range(16):
            for hi in range(16):
                if lo == 0 and hi == 0:
                    continue
                value = lo | (hi << 4)
                if value == 0xE0:
                    continue
                rules.append(
                    FFNRule.gated_write(
                        name=f"tail_stack0_store_top_e0_byte_{value:02x}",
                        conditions=base_conditions + (
                            (f"ALU_LO+{lo}", 1.0),
                            (f"ALU_HI+{hi}", 1.0),
                            (f"OUTPUT_LO+{lo}", 0.001),
                            (f"OUTPUT_HI+{hi}", 0.001),
                        ),
                        threshold=25.0,
                        gate="MARK_STACK0",
                        writes=byte_writes(value, strength=2000.0),
                    )
                )
        return tuple(rules)

    def stack0_store_loaded_output_rules() -> tuple[FFNRule, ...]:
        """Let strong L15 store/pop STACK0 memory loads beat stale cleanup.

        SI/SC pops the address and exposes memory[post-pop SP] as the next
        STACK0.  L10/L14 can still leave the just-stored AX byte in OUTPUT,
        and the stale cleanup rules above zero that residue.  When L15 has
        actually resolved a historical nonzero stack value, its OUTPUT signal
        is much larger than the current-AX residue, so restore it here.
        """

        base_conditions = (
            ("HAS_SE", 1.0),
            ("CMP+3", 0.5),
            ("MEM_STORE", 1.0),
            ("IS_BYTE", -100.0),
            ("MARK_AX", -1000000.0),
            ("MARK_PC", -1000000.0),
            ("MARK_SP", -1000000.0),
            ("MARK_BP", -1000000.0),
            ("MARK_MEM", -1000000.0),
        )
        rules = []
        for lo in range(16):
            for hi in range(16):
                if lo == 0 and hi == 0:
                    continue
                value = lo | (hi << 4)
                rules.append(
                    FFNRule.gated_write(
                        name=f"tail_stack0_store_loaded_byte_{value:02x}",
                        conditions=base_conditions + (
                            (f"OUTPUT_LO+{lo}", 1.0),
                            (f"OUTPUT_HI+{hi}", 1.0),
                        ),
                        threshold=25.0,
                        gate="MARK_STACK0",
                        writes=byte_writes(value, strength=5000.0),
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
            ("MARK_AX", -1000000.0),
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

    def ax_add_no_carry_zero_rules() -> tuple[FFNRule, ...]:
        """Clear ADD byte 1 when both operand high bytes and carry are zero."""

        marker_blockers = (
            ("MARK_AX", -1000000.0),
            ("MARK_PC", -10000.0),
            ("MARK_SP", -20000000.0),
            ("MARK_BP", -10000.0),
            ("MARK_STACK0", -20000000.0),
            ("MARK_MEM", -10000.0),
            ("H1+2", -1000000.0),
            ("H1+3", -1000000.0),
            ("H1+4", -1000000.0),
        )
        non_add_blockers = (
            ("OP_IMM", -1000000.0),
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
            ("OP_ENT", -1000000.0),
            ("MEM_STORE", -1000.0),
        )
        return (
            FFNRule.constant_write(
                name="tail_ax_add_no_carry_byte1_00",
                conditions=(
                    ("IS_BYTE", 5.0),
                    ("HAS_SE", 5.0),
                    ("H1+1", 20.0),
                    ("BYTE_INDEX_0", 5.0),
                    ("TEMP+8", 100.0),
                    ("CARRY+1", -10000.0),
                    ("ALU_LO+0", 10.0),
                    ("ALU_HI+0", 20.0),
                    ("AX_CARRY_HI+0", 20.0),
                ) + tuple(
                    (f"ALU_LO+{other}", -50.0)
                    for other in range(1, 16)
                ) + tuple(
                    (f"AX_CARRY_LO+{other}", -2000.0)
                    for other in range(1, 16)
                ) + marker_blockers + non_add_blockers,
                threshold=300.0,
                writes=byte_writes(0x00, strength=5000.0),
            ),
        )

    def ax_add_byte1_high_zero_rules() -> tuple[FFNRule, ...]:
        """Assert high nibble zero for low 16-bit ADD byte-1 rows.

        The ADD byte-1 low nibble can be correct while stale cleanup residue
        leaves OUTPUT_HI[1..15] above OUTPUT_HI[0]. Limit this to rows where
        both operand byte high nibbles are zero; wider byte-1 sums need a
        separate carry-aware high-nibble rule.
        """

        marker_blockers = (
            ("MARK_AX", -1000000.0),
            ("MARK_PC", -10000.0),
            ("MARK_SP", -10000.0),
            ("MARK_BP", -10000.0),
            ("MARK_STACK0", -10000.0),
            ("MARK_MEM", -10000.0),
        )
        non_add_blockers = (
            ("CARRY+2", -1000.0),
            ("TEMP+9", -1000.0),
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
        writes = [("OUTPUT_HI+0", 5000.0)]
        writes.extend((f"OUTPUT_HI+{other}", -5000.0) for other in range(1, 16))
        return (
            FFNRule.constant_write(
                name="tail_ax_add_byte1_hi_zero",
                conditions=(
                    ("IS_BYTE", 5.0),
                    ("HAS_SE", 5.0),
                    ("H1+1", 20.0),
                    ("BYTE_INDEX_0", 5.0),
                    ("TEMP+8", 100.0),
                    ("ALU_HI+0", 20.0),
                    ("AX_CARRY_HI+0", 20.0),
                ) + marker_blockers + non_add_blockers,
                threshold=250.0,
                writes=tuple(writes),
            ),
        )

    def wide_mul_byte1_preserve_rules() -> tuple[FFNRule, ...]:
        """Keep the staged MUL byte-1 value authoritative through the tail.

        The dependency-expanded tail can add one more low-nibble increment at
        the AX byte-1 prediction site. Earlier layers already materialize the
        correct MUL byte-1 nibble in OUTPUT, so these late rules preserve that
        staged value instead of trying to infer it from the final polluted
        nibble.
        """

        non_mul_block = -1_000_000_000_000_000_000_000_000_000_000.0
        non_mul_blockers = (
            ("OP_ADD", non_mul_block),
            ("OP_SUB", non_mul_block),
            ("OP_DIV", non_mul_block),
            ("OP_MOD", non_mul_block),
            ("OP_SHL", non_mul_block),
            ("OP_SHR", non_mul_block),
            ("OP_AND", non_mul_block),
            ("OP_OR", non_mul_block),
            ("OP_XOR", non_mul_block),
        )
        rules = []
        for high_nibble in range(16):
            for low_nibble in range(16):
                byte_value = (high_nibble << 4) | low_nibble
                name = (
                    f"tail_wide_mul_byte1_preserve_{low_nibble:01x}"
                    if high_nibble == 0
                    else f"tail_wide_mul_byte1_preserve_{high_nibble:01x}{low_nibble:01x}"
                )
                temp_block = non_mul_block if high_nibble == 0 else -1_000_000_000.0
                add_temp_block = non_mul_block if high_nibble == 0 else -1_000_000_000.0
                sub_temp_block = non_mul_block
                has_weight = (
                    3_000_000_000_000_000.0
                    if high_nibble == 0
                    else 30_000_000_000_000.0
                )
                output_weight = (
                    100_000_000.0
                    if high_nibble == 0
                    else 1_000_000_000.0
                )
                threshold = (
                    3_000_005_000_000_000.0
                    if high_nibble == 0
                    else 30_030_000_000_000.0
                )
                span_blockers = (
                    ("H1+2", -100_000_000_000_000_000.0),
                    ("H1+3", -100_000_000_000_000_000.0),
                    ("H1+4", -100_000_000_000_000_000.0),
                )
                rules.append(
                    FFNRule.gated_write(
                        name=name,
                        conditions=ax_byte0 + (
                            ("HAS_SE", has_weight),
                            ("TEMP+10", 1_000_000_000.0),
                            ("OP_EQ", -1000.0),
                            ("OP_NE", -1000.0),
                            ("OP_LT", -1000.0),
                            ("OP_GT", -1000.0),
                            ("OP_LE", -1000.0),
                            ("OP_GE", -1000.0),
                            ("OP_JSR", -1_000_000_000_000.0),
                            ("OP_ENT", -1_000_000_000_000.0),
                            ("OP_LEV", -1_000_000_000_000.0),
                            (f"OUTPUT_LO+{low_nibble}", output_weight),
                            (f"OUTPUT_HI+{high_nibble}", output_weight),
                            ("TEMP+4", temp_block),
                            ("TEMP+5", temp_block),
                            ("TEMP+6", temp_block),
                            ("TEMP+8", add_temp_block),
                            ("TEMP+9", sub_temp_block),
                        ) + span_blockers + non_mul_blockers,
                        threshold=threshold,
                        gate="BYTE_INDEX_0",
                        writes=byte_writes(byte_value, strength=10_000_000.0),
                    )
                )
        return tuple(rules)

    ax_byte0 = (
        ("IS_BYTE", 1.0),
        ("H1+1", 1.0),
        ("H1+2", -1_000_000_000_000.0),
        ("H1+3", -1_000_000_000_000.0),
        ("H1+4", -1_000_000_000_000.0),
        ("BYTE_INDEX_0", 1.0),
        ("BYTE_INDEX_1", -1_000_000_000.0),
        ("BYTE_INDEX_2", -1_000_000_000.0),
        ("BYTE_INDEX_3", -1_000_000_000.0),
        ("MARK_AX", -1000000.0),
        ("MARK_PC", -1000000.0),
        ("MARK_SP", -1000000.0),
        ("MARK_BP", -1000000.0),
        ("MARK_STACK0", -1000000.0),
        ("MARK_MEM", -1000000.0),
        ("OP_LEA", -1000.0),
        ("OP_JMP", -1000000.0),
        ("OP_ADJ", -1000.0),
        ("OP_ENT", -1000.0),
    )
    si_ax_byte0 = (
        ("IS_BYTE", 1.0),
        ("H1+1", 20.0),
        ("BYTE_INDEX_0", 20.0),
        ("BYTE_INDEX_1", -1_000_000_000_000_000.0),
        ("BYTE_INDEX_2", -1_000_000_000_000_000.0),
        ("BYTE_INDEX_3", -1_000_000_000_000_000.0),
        ("MARK_AX", -200.0),
        ("OP_SI", 20.0),
        ("MEM_STORE", 20.0),
    )

    def pc_byte_span_blocked(rules: tuple[FFNRule, ...]) -> tuple[FFNRule, ...]:
        """Keep late tail fixes from rewriting PC byte predictions."""

        blocker = ConditionTerm(DimRef.parse("H1+0"), -1_000_000.0)
        blocked = []
        for rule in rules:
            if rule.name == "tail_clear_output_after_byte3":
                blocked.append(rule)
            else:
                blocked.append(replace(rule, conditions=rule.conditions + (blocker,)))
        return tuple(blocked)

    def step_end_transition_blocked(rules: tuple[FFNRule, ...]) -> tuple[FFNRule, ...]:
        """Keep byte-tail repairs from competing with STEP_END marker emission."""

        blocker = ConditionTerm(DimRef.parse("NEXT_SE"), -1_000_000.0)
        blocked = []
        for rule in rules:
            if rule.name == "tail_clear_output_before_step_end":
                blocked.append(rule)
            else:
                blocked.append(replace(rule, conditions=rule.conditions + (blocker,)))
        return tuple(blocked)

    rules = (
        # Binary-pop ops consume the top stack cell. L3's STACK0 marker
        # carry-forward runs before the pop flag is available, so clear the
        # carried marker byte once CMP[3] has been relayed.
        FFNRule.constant_write(
            name="tail_stack0_pop_marker_zero",
            conditions=(
                ("MARK_STACK0", 1.0),
                ("HAS_SE", 1.0),
                ("CMP+3", 0.5),
                ("IS_BYTE", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_MEM", -100.0),
                ("MEM_STORE", -100.0),
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
        # When a binary op pops the value at SP=d8, the next stack cell is the
        # saved local address at e8. The generic pop-marker zero rule above
        # clears stale carried STACK0 bytes; this narrower rule restores the
        # revealed address needed by update/store expressions.
        FFNRule.constant_write(
            name="tail_stack0_pop_reveals_saved_addr_e8",
            conditions=(
                ("MARK_STACK0", 1.0),
                ("HAS_SE", 1.0),
                ("CMP+3", 25.0),
                ("ADDR_B0_LO+8", 2.0),
                ("ADDR_B0_HI+13", 2.0),
                ("ADDR_B0_LO+0", -5.0),
                ("MEM_STORE", -100.0),
                ("IS_BYTE", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_MEM", -100.0),
            ),
            threshold=100.0,
            writes=byte_writes(0xE8, strength=500.0),
        ),
        # Store pops only make the stored AX value the new STACK0 when the
        # store address equals the post-pop SP. For local stores such as
        # BP-8 with a larger frame, the store address remains above the new
        # SP; the STACK0 marker should therefore be zero, not the stored AX.
        FFNRule.constant_write(
            name="tail_stack0_store_non_top_zero",
            conditions=(
                ("MARK_STACK0", 1.0),
                ("HAS_SE", 1.0),
                ("CMP+3", 10.0),
                ("MEM_STORE", 20.0),
                ("EMBED_LO+0", -20.0),
                ("EMBED_LO+8", 1.0),
                ("EMBED_HI+14", 1.0),
                ("ADDR_B0_LO+8", 1.0),
                ("ADDR_B0_HI+13", 1.0),
                ("ADDR_B0_HI+14", -10.0),
                ("IS_BYTE", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_MEM", -100.0),
            ),
            threshold=50.0,
            writes=byte_writes(0x00, strength=1000.0),
        ),
        FFNRule.constant_write(
            name="tail_stack0_store_non_top_zero_e8_from_e0",
            conditions=(
                ("MARK_STACK0", 100.0),
                ("HAS_SE", 1.0),
                ("CMP+3", 10.0),
                ("MEM_STORE", 200.0),
                ("EMBED_LO+0", -100.0),
                ("EMBED_LO+8", 1.0),
                ("EMBED_HI+14", 1.0),
                ("ADDR_B0_LO+0", 1.0),
                ("ADDR_B0_HI+14", 1.0),
                ("ADDR_B0_LO+8", -20.0),
                ("IS_BYTE", -1000.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -1000.0),
                ("MARK_SP", -1000.0),
                ("MARK_BP", -1000.0),
                ("MARK_MEM", -1000.0),
            ),
            threshold=240.0,
            writes=byte_writes(0x00, strength=1000.0),
        ),
        FFNRule.constant_write(
            name="tail_stack0_store_non_top_zero_e0",
            conditions=(
                ("MARK_STACK0", 1.0),
                ("HAS_SE", 1.0),
                ("CMP+3", 0.5),
                ("MEM_STORE", 1.0),
                ("EMBED_LO+0", 1.0),
                ("EMBED_HI+14", 1.0),
                ("IS_BYTE", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_MEM", -100.0),
            ),
            threshold=5.8,
            writes=byte_writes(0x00, strength=1000.0),
        ),
        # At the final SP byte position, byte-output residue can beat the
        # stack-base high byte. Assert the zero byte for binary-pop SP byte 3.
        FFNRule.gated_write(
            name="tail_sp_pop_byte3_zero",
            conditions=(
                ("IS_BYTE", 5.0),
                ("HAS_SE", 5.0),
                ("H1+2", 20.0),
                ("H1+1", -1_000_000_000_000_000_000_000_000_000_000.0),
                ("BYTE_INDEX_0", -10000.0),
                ("BYTE_INDEX_1", -10000.0),
                ("BYTE_INDEX_3", -10000.0),
                ("CMP+3", 1.0),
                ("OUTPUT_LO+0", 100.0),
                ("MARK_SP", -100000000.0),
                ("MARK_AX", -100000000.0),
                ("MARK_PC", -100000000.0),
                ("MARK_BP", -100000000.0),
                ("MARK_STACK0", -100000000.0),
                ("MARK_MEM", -100000000.0),
                ("STACK0_BYTE0", -10000.0),
                ("STACK0_BYTE1", -10000.0),
                ("STACK0_BYTE2", -10000.0),
                ("STACK0_BYTE3", -10000.0),
            ),
            threshold=180.0,
            gate="BYTE_INDEX_2",
            writes=byte_writes(0x00, strength=10000.0),
        ),
        *sp_pop_marker_increment_rules(),
        *sp_pop_byte1_preserve_rules(),
        *stack0_pushed_addr_byte1_preserve_rules(),
        *stack0_pop_loaded_output_rules(),
        *stack0_store_loaded_output_rules(),
        *stack0_store_top_e0_output_rules(),
        # Non-memory binary pops should emit a zero MEM row. Store/load ops
        # have dedicated memory paths and block this cleanup.
        FFNRule.constant_write(
            name="tail_pop_mem_marker_zero",
            conditions=(
                ("MARK_MEM", 1.0),
                ("HAS_SE", 1.0),
                ("CMP+3", 0.5),
                ("IS_BYTE", -100.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -100.0),
                ("MARK_SP", -100.0),
                ("MARK_BP", -100.0),
                ("MARK_STACK0", -100.0),
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
                ("MEM_STORE", -100.0),
            ),
            threshold=2.5,
            writes=byte_writes(0x00, strength=500.0),
        ),
        # BP is stable across ordinary binary ops. The L10 BP passthrough
        # carries byte 2 as a weak 0x01 signal; reinforce it when no frame op
        # is rewriting BP.
        FFNRule.gated_write(
            name="tail_bp_byte2_preserve_01",
            conditions=(
                ("IS_BYTE", 1.0),
                ("HAS_SE", 1.0),
                ("H1+3", 20.0),
                ("H1+1", -100.0),
                ("H1+2", -100.0),
                ("BYTE_INDEX_1", 1.0),
                ("BYTE_INDEX_0", -1_000_000.0),
                ("BYTE_INDEX_2", -1_000_000.0),
                ("BYTE_INDEX_3", -1_000_000.0),
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
                ("OP_LEA", -1000.0),
                ("OP_LEV", -100.0),
                ("OP_EQ", -1000.0),
                ("OP_NE", -1000.0),
                ("OP_LT", -1000.0),
                ("OP_GT", -1000.0),
                ("OP_LE", -1000.0),
                ("OP_GE", -1000.0),
            ),
            threshold=22.0,
            gate="H1+3",
            writes=byte_writes(0x01, strength=500.0),
        ),
        *ax_add_no_carry_zero_rules(),
        *ax_add_byte1_high_zero_rules(),
        *wide_mul_byte1_preserve_rules(),
        FFNRule.gated_write(
            name="tail_ax_add_byte1_carry_high2_03",
            conditions=ax_byte0 + (
                ("BYTE_INDEX_1", -1000.0),
                ("BYTE_INDEX_2", -1000.0),
                ("BYTE_INDEX_3", -1000.0),
                ("TEMP+8", 10.0),
                ("TEMP+9", -1000.0),
                ("CARRY+1", 1.0),
                ("CARRY+2", -1000.0),
                ("FETCH_HI+1", 0.1),
                ("OUTPUT_LO+3", 0.001),
                ("OUTPUT_LO+6", -10.0),
                ("OUTPUT_LO+7", -10.0),
                ("OP_IMM", -1000.0),
            ),
            threshold=1000.0,
            gate="CARRY+1",
            writes=byte_writes(0x03, strength=5000.0),
        ),
        # SHL-by-8 loses byte 1 to the same tail, but its signature is a huge
        # OUTPUT_LO[1] plus carry residue rather than MUL's OUTPUT_LO[2].
        FFNRule.gated_write(
            name="tail_wide_shl_byte1_01",
            conditions=ax_byte0 + (
                ("OP_SHL", 100.0),
                ("EMBED_LO+0", 1.0),
                ("EMBED_HI+0", 1.0),
            ),
            threshold=90.0,
            gate="CARRY+3",
            writes=byte_writes(0x01),
        ),
        # SI preserves AX while writing memory. The dependency-expanded tail
        # can clobber the AX byte-1 prediction after the earlier layers have
        # prepared the right 16-bit value. OP_SI is relayed to AX byte
        # positions by L7; the HI-nibble comparison distinguishes a real
        # nonzero stored high byte from the common zero-high-byte store cases.
        FFNRule.gated_write(
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
            threshold=161.0,
            gate="MEM_STORE",
            writes=byte_writes(0x12, strength=300.0),
        ),
        FFNRule.gated_write(
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
            threshold=161.5,
            gate="MEM_STORE",
            writes=byte_writes(0x00, strength=300.0),
        ),
        # SUB 0x0100-1 carries borrow residue in CARRY[2]/[3] and must clear
        # byte 1 to zero; the old tail currently leaves 0x01 there. Use
        # CARRY[2] instead of CARRY[3] so wide MUL/SHL carry residue does not
        # accidentally trigger the zeroing rule.
        FFNRule.gated_write(
            name="tail_sub_borrow_byte1_00",
            conditions=ax_byte0 + (
                ("MARK_AX", -1000.0),
                ("OP_IMM", -1000.0),
                ("TEMP+8", -20000.0),
                ("TEMP+9", 100.0),
                ("CARRY+2", 100.0),
                ("ALU_LO+1", 1.0),
                ("OUTPUT_LO+1", -5.0),
                ("OUTPUT_LO+3", -5.0),
                ("OUTPUT_LO+5", -5.0),
                ("OUTPUT_LO+6", -5.0),
            ),
            threshold=250.0,
            gate="CARRY+2",
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
        FFNRule.gated_write(
            name="tail_or_xor_byte1_0f",
            conditions=ax_byte0 + (
                ("TEMP+3", 1.0),
                ("TEMP+4", -2.0),
                ("CARRY+3", 0.01),
                ("OUTPUT_LO+14", 0.001),
            ),
            threshold=5.5,
            gate="BYTE_INDEX_0",
            writes=byte_writes(0x0F),
        ),
        # SHR by 8 also needs byte 1 cleared after the marker correction emits
        # byte 0 as 0x01; TEMP[7] is the reliable non-carry/SHR signature at
        # byte positions.
        FFNRule.gated_write(
            name="tail_shr_byte1_00",
            conditions=ax_byte0 + (
                ("TEMP+7", 1.0),
                ("OUTPUT_LO+1", 0.001),
            ),
            threshold=3.9,
            gate="TEMP+7",
            writes=byte_writes(0x00),
        ),
        # SHR by 8 currently computes byte 0 as 0x06 at the AX marker. OP_SHR
        # is still visible at the marker, so correct the marker prediction
        # before byte generation proceeds.
        FFNRule.gated_write(
            name="tail_shr_marker_byte0_01",
            conditions=(
                ("MARK_AX", 1.0),
                ("MARK_PC", -10000.0),
                ("MARK_SP", -10000.0),
                ("MARK_BP", -10000.0),
                ("MARK_STACK0", -10000.0),
                ("MARK_MEM", -10000.0),
                ("IS_BYTE", -100.0),
                ("H1+1", 1.0),
                ("TEMP+7", 1.0),
                ("OP_SHR", 100.0),
                ("OP_IMM", -100.0),
                ("OUTPUT_LO+6", 0.1),
            ),
            threshold=102.0,
            gate="MARK_AX",
            writes=byte_writes(0x01),
        ),
        FFNRule.constant_write(
            name="tail_ax_add_byte1_missing_stack_high_02",
            conditions=(
                ("IS_BYTE", 10.0),
                ("HAS_SE", 10.0),
                ("H1+1", 20.0),
                ("BYTE_INDEX_0", 10.0),
                ("BYTE_INDEX_1", -1000.0),
                ("TEMP+8", 50.0),
                ("TEMP+9", -1_000_000_000_000_000_000_000_000_000_000.0),
                ("CARRY+1", 100.0),
                ("FETCH_HI+1", 1000.0),
                ("OUTPUT_LO+1", 0.01),
                ("OUTPUT_LO+3", -1.0),
                ("OUTPUT_LO+4", -1.0),
                ("OUTPUT_LO+5", -1.0),
                ("OUTPUT_LO+6", -1.0),
                ("OUTPUT_LO+7", -1.0),
                ("MARK_AX", -1000000.0),
                ("MARK_PC", -1000000.0),
                ("MARK_SP", -1000000.0),
                ("MARK_BP", -1000000.0),
                ("MARK_STACK0", -1000000.0),
                ("MARK_MEM", -1000000.0),
                ("H1+2", -1000000.0),
                ("H1+3", -1000000.0),
                ("H1+4", -1000000.0),
            ),
            threshold=1450.0,
            writes=byte_writes(0x02, strength=5000.0),
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
                ("CMP+1", -1.0),
                ("CMP+2", -1.0),
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
            name="tail_cmp_le_eq_prefix_false_00",
            conditions=(
                ("MARK_AX", 1.0),
                ("OP_LE", 1.0),
                ("CMP+1", 0.1),
                ("CMP+2", -1.0),
                ("CMP+3", -1.0),
            ),
            threshold=6.0,
            writes=byte_writes(0x00, strength=1000.0),
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
        # Keep this appended after the legacy tail rules so existing generated
        # unit indexes remain stable. It repairs SP marker d8->e0 when the
        # staged value exists only in OUTPUT, not in EMBED.
        FFNRule.gated_write(
            name="tail_sp_pop_marker_output_d8_to_e0",
            conditions=(
                ("CONST", -100000000.0),
                ("MARK_SP", 100000000.0),
                ("HAS_SE", 1000.0),
                ("CMP+3", 100.0),
                ("OUTPUT_LO+8", 10.0),
                ("OUTPUT_HI+13", 1.0),
                ("MARK_AX", -1000.0),
                ("MARK_PC", -1000.0),
                ("MARK_BP", -1000.0),
                ("MARK_STACK0", -1000.0),
                ("MARK_MEM", -1000.0),
                ("OP_ENT", -1000000.0),
                ("OP_LEV", -1000000.0),
                ("PSH_AT_SP", -1000000.0),
                ("IS_BYTE", -100.0),
            ),
            threshold=1405.0,
            gate="MARK_SP",
            writes=byte_writes(0xE0, strength=5000.0),
        ),
        FFNRule.gated_write(
            name="tail_clear_output_after_byte3",
            conditions=(
                ("IS_BYTE", 1.0),
                ("BYTE_INDEX_3", 1.0),
                ("MARK_AX", -1000.0),
                ("MARK_PC", -1000.0),
                ("MARK_SP", -1000.0),
                ("MARK_BP", -1000.0),
                ("MARK_STACK0", -1000.0),
                ("MARK_MEM", -1000.0),
                ("MEM_VAL_B0", -1000.0),
            ),
            threshold=1.5,
            gate="BYTE_INDEX_3",
            writes=clear_output_writes(strength=10_000_000_000.0),
        ),
        FFNRule.gated_write(
            name="tail_clear_output_before_step_end",
            conditions=(
                ("IS_BYTE", 1.0),
                ("NEXT_SE", 1.0),
                ("MARK_AX", -1000.0),
                ("MARK_PC", -1000.0),
                ("MARK_SP", -1000.0),
                ("MARK_BP", -1000.0),
                ("MARK_STACK0", -1000.0),
                ("MARK_MEM", -1000.0),
            ),
            threshold=1.5,
            gate="NEXT_SE",
            writes=clear_output_writes(strength=10_000_000_000.0),
        ),
    ) + sp_pop_carry_rules()
    return step_end_transition_blocked(pc_byte_span_blocked(rules))


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
        _suppress_ffn_on_step_boundary(ffn, dim_map, S)
        block.post_ops.append(ffn)

    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(rules)

    return Operation(
        name="tail_bit32_result_correction",
        phase=17.1,
        reads={
            "CONST", "IS_BYTE", "HAS_SE", "H1",
            "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
            "MARK_AX", "MARK_PC", "MARK_SP", "MARK_BP", "MARK_STACK0",
            "MARK_MEM", "NEXT_SE", "OP_SHL", "OP_SHR", "OP_IMM", "OP_JSR",
            "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
            "OP_ENT", "OP_LEV", "OP_SI", "OP_SC", "OP_LI", "OP_LC",
            "OP_ADD", "OP_SUB", "OP_DIV", "OP_MOD", "OP_AND", "OP_OR",
            "OP_XOR",
            "EMBED_LO", "EMBED_HI", "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
            "TEMP", "CMP", "CARRY", "ALU_LO", "ALU_HI", "AX_CARRY_LO", "FETCH_HI",
            "OUTPUT_LO", "OUTPUT_HI", "MEM_STORE",
            "ADDR_B0_LO", "ADDR_B0_HI",
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
        zeroing = BinaryOpByteZeroingPostOp(
            d_model=d_model, S=S, dim_positions=dim_positions
        )
        _suppress_ffn_on_step_boundary(zeroing, dim_positions, S)
        block.post_ops.append(zeroing)
        BD = _as_setdim_proxy(dim_positions) if isinstance(dim_positions, dict) else _SetDim
        addsub = AddSubBytePropagationPostOp(
            d_model=d_model,
            S=S,
            dim_positions=dim_positions,
        )
        _strengthen_l10_addsub_wrong_byte_blockers(addsub, BD, S)
        _suppress_l10_addsub_on_wide_alu(addsub, BD, S)
        _suppress_ffn_on_step_boundary(addsub, dim_positions, S)
        block.post_ops.append(addsub)
        carry0 = CarryPropagationPostOp(
            d_model=d_model, S=S, byte_idx=0, cascade=False,
            dim_positions=dim_positions,
        )
        _strengthen_l10_carry_wrong_byte_blockers(carry0, BD, byte_idx=0, S=S)
        _strengthen_l10_first_carry_delta(carry0, BD)
        _suppress_ffn_on_step_boundary(carry0, dim_positions, S)
        block.post_ops.append(carry0)
        carry1 = CarryPropagationPostOp(
            d_model=d_model, S=S, byte_idx=1, cascade=True,
            dim_positions=dim_positions,
        )
        _strengthen_l10_carry_wrong_byte_blockers(carry1, BD, byte_idx=1, S=S)
        _suppress_ffn_on_step_boundary(carry1, dim_positions, S)
        block.post_ops.append(carry1)
        carry2 = CarryPropagationPostOp(
            d_model=d_model, S=S, byte_idx=2, cascade=True,
            dim_positions=dim_positions,
        )
        _strengthen_l10_carry_wrong_byte_blockers(carry2, BD, byte_idx=2, S=S)
        _suppress_ffn_on_step_boundary(carry2, dim_positions, S)
        block.post_ops.append(carry2)
        bitwise = BitwiseBytePropagationPostOp(
            d_model=d_model, S=S, dim_positions=dim_positions
        )
        _suppress_ffn_on_step_boundary(bitwise, dim_positions, S)
        block.post_ops.append(bitwise)
        if alu_mode == "efficient":
            # Pass the model's actual d_model so the underlying PureFFN's
            # Linear input dim matches the residual stream width. Without
            # this, ComparisonCombine builds a Linear(512, 18) which fails
            # forward when d_model != 512 (e.g., pin_io_only=True paths).
            compare = ComparisonCombine(
                d_model=d_model, S=S, dim_positions=dim_positions
            )
            _suppress_ffn_on_step_boundary(compare, dim_positions, S)
            block.post_ops.append(compare)
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
