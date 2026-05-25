"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ..ir import FFNRule
from ..layer_compiler import Operation
from ..primitives import AO, AP, DeclarativeAttentionHeadSpec, Primitives
from .shared import _as_setdim_proxy


def make_layer3_ffn_op() -> Operation:
    """L3 FFN: PC/SP/BP first-step defaults + PC byte-0 increment.

    Originally: `_set_layer3_ffn` at vm_step.py:2929.

    Reads MARK_PC, MARK_SP, MARK_BP, MARK_STACK0, HAS_SE, EMBED_LO/HI,
    H1, H4, BYTE_INDEX_*, OP_LEV, NEXT_STACK0.
    Writes OUTPUT_LO/HI, EMBED_LO/HI, NEXT_STACK0.

    Pinned to ``layer_idx=3`` via ``kind="block"`` because the legacy
    ``set_vm_weights`` pipeline targets ``model.blocks[3].ffn``. Without
    pinning, the dep-graph layer assignment placed this op at block 4,
    which would conflict with the L4 FFN bake (the same regression noted
    on ``make_layer4_pc_relay_op``). The companion
    ``_layer3_ffn_dep_anchor`` op (kind="ffn") declares identical
    reads/writes so the LayerCompiler's dep graph still reserves a layer
    slot for it; otherwise removing the kind="ffn" entry shrinks the
    longest-chain length and shifts downstream migrated kind="attn" ops
    (e.g. ``layer14_mem_generation``) to the wrong block.
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer3_ffn
        proxy = _as_setdim_proxy(dim_positions)
        _set_layer3_ffn(block.ffn, S, proxy)
        _add_layer3_pc_byte1_output_rules(block.ffn, S, proxy)

    return Operation(
        name="layer3_ffn",
        phase=3,
        reads={"MARK_PC", "MARK_SP", "MARK_BP", "MARK_STACK0", "HAS_SE",
               "EMBED_LO", "EMBED_HI", "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
               "TEMP", "IS_BYTE", "H1", "H4", "OP_LEV",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
               "NEXT_STACK0"},
        writes={"OUTPUT_LO", "OUTPUT_HI", "EMBED_LO", "EMBED_HI",
                "NEXT_STACK0"},
        kind="block",
        layer_idx=3,
        bake_fn=bake,
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        migrated=True,
        postcondition={
            "OUTPUT_LO": "monotonic_non_decreasing",
        },
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
        compaction_safe=True,
    )


def _next_free_ffn_unit(ffn) -> int:
    active = (
        (ffn.W_up.data.abs().sum(dim=1) > 0)
        | (ffn.W_gate.data.abs().sum(dim=1) > 0)
        | (ffn.b_up.data.abs() > 0)
        | (ffn.b_gate.data.abs() > 0)
        | (ffn.W_down.data.abs().sum(dim=0) > 0)
    )
    used = active.nonzero(as_tuple=True)[0]
    return int(used[-1].item() + 1) if len(used) else 0


def _write_pc_byte1_one(ffn, unit: int, BD, S: float, conditions) -> int:
    for dim, weight in conditions:
        ffn.W_up.data[unit, dim] = S * weight
    ffn.b_up.data[unit] = -S * 5.5
    ffn.b_gate.data[unit] = 1.0
    ffn.W_down.data[BD.OUTPUT_LO + 0, unit] = -500.0 / S
    ffn.W_down.data[BD.OUTPUT_LO + 1, unit] = 500.0 / S
    ffn.W_down.data[BD.OUTPUT_HI + 0, unit] = 500.0 / S
    return unit + 1


def _add_layer3_pc_byte1_output_rules(ffn, S, BD) -> None:
    """Emit PC byte1 for programs whose linear PC crosses 0x100.

    Legacy L3 owns ordinary PC emission but only increments byte 0 and then
    defaults bytes 1-3 to zero. Head 7 below stages the previous step's PC
    byte1 into TEMP at the current PC byte0 row. These two late units make
    byte1 equal one either on the wrap token (new byte0 == 0x02) or while the
    previous byte1 was already one. The preserve rule is intentionally bounded
    to byte0 high nibbles 0..4 because the current 1096 corpus never runs
    past 0x14a; branch targets below 0x100 such as 0x62 must not preserve the
    prior high byte.
    """

    PC_I = 0
    unit = _next_free_ffn_unit(ffn)
    if unit + 2 > ffn.W_up.shape[0]:
        raise RuntimeError("L3 FFN has no room for PC byte1 carry repair")

    common = (
        (BD.H1 + PC_I, 1.0),
        (BD.BYTE_INDEX_0, 1.0),
        (BD.IS_BYTE, 1.0),
        (BD.HAS_SE, 1.0),
    )
    unit = _write_pc_byte1_one(
        ffn,
        unit,
        BD,
        S,
        common + (
            (BD.CLEAN_EMBED_LO + 2, 1.0),
            (BD.CLEAN_EMBED_HI + 0, 1.0),
        ),
    )

    for dim, weight in common:
        ffn.W_up.data[unit, dim] = S * weight
    ffn.W_up.data[unit, BD.TEMP + 1] = S
    ffn.W_up.data[unit, BD.TEMP + 16] = S
    for hi in range(5):
        ffn.W_up.data[unit, BD.CLEAN_EMBED_HI + hi] = S
    ffn.b_up.data[unit] = -S * 6.5
    ffn.b_gate.data[unit] = 1.0
    ffn.W_down.data[BD.OUTPUT_LO + 0, unit] = -500.0 / S
    ffn.W_down.data[BD.OUTPUT_LO + 1, unit] = 500.0 / S
    ffn.W_down.data[BD.OUTPUT_HI + 0, unit] = 500.0 / S


def make_layer3_ffn_dep_anchor_op() -> Operation:
    """No-op companion for ``layer3_ffn``: declares identical reads/writes so
    the LayerCompiler's dep graph reserves a layer slot for it. Mirrors
    ``_layer5_fetch_dep_anchor``: the actual bake happens in
    ``layer3_ffn`` (kind="block", layer_idx=3); this op's bake is a no-op
    (its layout-assigned ffn block is unrelated to block[3]).
    """
    def bake(ffn, dim_positions, S):
        # No-op: actual bake is in `layer3_ffn` block op above.
        return

    return Operation(
        name="_layer3_ffn_dep_anchor",
        phase=3,
        reads={"MARK_PC", "MARK_SP", "MARK_BP", "MARK_STACK0", "HAS_SE",
               "EMBED_LO", "EMBED_HI", "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
               "TEMP", "IS_BYTE", "H1", "H4", "OP_LEV",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
               "NEXT_STACK0"},
        writes={"OUTPUT_LO", "OUTPUT_HI", "EMBED_LO", "EMBED_HI",
                "NEXT_STACK0"},
        kind="ffn",
        bake_fn=bake,
        migrated=True,
        declarative_authority="topology_anchor",
        smoke_tests=set(),
        spec_section=None,
    )


def make_layer3_carry_forward_attn_op() -> Operation:
    """L3 attention: 8 carry-forward heads (PC, AX, SP, BP, STACK0 + relays).

    Heads 0-3 use ``Primitives.carry_forward_attention`` (the canonical
    proxy-aware implementation; the legacy ``_set_carry_forward_attn``
    helper was deleted per BD_SETDIM_HARDCODE_AUDIT M1). Head 4 uses
    ``_set_stack0_carry_attn`` (different K source). Heads 5-7 are
    declarative relays for AX_FULL, LEV BP->PC, and PC byte1 preservation.
    """
    def bake(attn, dim_positions, S):
        from ..primitives import Primitives
        proxy = _as_setdim_proxy(dim_positions)
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes.fill_(0.5)
        HD = attn.W_q.shape[0] // attn.num_heads
        PC_I, AX_I, SP_I, BP_I = 0, 1, 2, 3
        cf = Primitives.carry_forward_attention
        # Pass proxy as bd= so pin_io_only=True layouts resolve L1H0/L1H1/CONST
        # to the compiler-allocated positions rather than legacy _SetDim ones.
        cf(attn, 0, proxy.MARK_PC, PC_I, PC_I, proxy.EMBED_LO, proxy.EMBED_HI, HD=HD, bd=proxy)
        cf(attn, 1, proxy.MARK_AX, AX_I, AX_I, proxy.AX_CARRY_LO, proxy.AX_CARRY_HI, HD=HD, bd=proxy)
        cf(attn, 2, proxy.MARK_SP, SP_I, SP_I, proxy.EMBED_LO, proxy.EMBED_HI, HD=HD, bd=proxy)
        cf(attn, 3, proxy.MARK_BP, BP_I, BP_I, proxy.EMBED_LO, proxy.EMBED_HI, HD=HD, bd=proxy)
        Primitives.generate_attention_heads(
            attn,
            (
                _stack0_carry_head_spec(proxy),
                _ax_full_relay_head_spec(proxy),
                _lev_bp_to_pc_head_spec(proxy),
                _pc_byte1_prev_head_spec(proxy),
            ),
            HD,
        )

    # Dim-ownership claims: 7 carry-forward attention heads.
    #   Heads 0-3: Primitives.carry_forward_attention writes V slots 1..32:
    #     W_v[h*HD + 1 + k, src_lo + k]    for k=0..15 (slot 1..16)
    #     W_v[h*HD + 17 + k, src_hi + k]   for k=0..15 (slot 17..32)
    #   Plus W_q[base, marker], W_k[base, L1H1], W_k[base, L1H0] and the
    #   GATE=33 row.  We capture the V/O row claims (the load-bearing
    #   slot/column pairs that can collide with other ops).
    #
    #   Head 0 (PC): src=EMBED_LO/HI, out=EMBED_LO/HI
    #   Head 1 (AX): src=EMBED_LO/HI, out=AX_CARRY_LO/HI
    #   Head 2 (SP): src=EMBED_LO/HI, out=EMBED_LO/HI
    #   Head 3 (BP): src=EMBED_LO/HI, out=EMBED_LO/HI
    #   Head 4 (STACK0): declarative STACK0_BYTE0 carry head spec below
    #   Head 5 (AX_FULL): inline V[OUTPUT_LO/HI] → AX_FULL_LO/HI
    #   Head 6 (BP→PC LEV): inline V[CLEAN_EMBED_LO/HI] → (out via inline)
    #   Head 7 (PC byte1): previous PC byte1 CLEAN_EMBED_LO/HI → TEMP
    _claims = set()
    _heads_cf = [
        (0, "EMBED_LO", "EMBED_HI"),
        (1, "EMBED_LO", "EMBED_HI"),
        (2, "EMBED_LO", "EMBED_HI"),
        (3, "EMBED_LO", "EMBED_HI"),
    ]
    for h, src_lo, src_hi in _heads_cf:
        for k in range(16):
            _claims.add((3, "attn_W_v", f"{h}_{1 + k}", f"{src_lo}+{k}"))
            _claims.add((3, "attn_W_v", f"{h}_{17 + k}", f"{src_hi}+{k}"))
    # Head 5: AX_FULL relay V slots from OUTPUT_LO/HI.
    for k in range(16):
        _claims.add((3, "attn_W_v", f"5_{1 + k}", f"OUTPUT_LO+{k}"))
        _claims.add((3, "attn_W_v", f"5_{17 + k}", f"OUTPUT_HI+{k}"))
    # Head 6: BP→PC LEV relay V slots from CLEAN_EMBED_LO/HI.
    for k in range(16):
        _claims.add((3, "attn_W_v", f"6_{1 + k}", f"CLEAN_EMBED_LO+{k}"))
        _claims.add((3, "attn_W_v", f"6_{17 + k}", f"CLEAN_EMBED_HI+{k}"))
    # Head 7: previous PC byte1 relay V slots from CLEAN_EMBED_LO/HI.
    for k in range(16):
        _claims.add((3, "attn_W_v", f"7_{1 + k}", f"CLEAN_EMBED_LO+{k}"))
        _claims.add((3, "attn_W_v", f"7_{17 + k}", f"CLEAN_EMBED_HI+{k}"))

    return Operation(
        name="layer3_carry_forward_attn",
        phase=3,
        reads={"MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP",
               "L1H0", "L1H1", "STACK0_BYTE0", "OP_LEV", "HAS_SE",
               "H1", "IS_BYTE", "BYTE_INDEX_0", "BYTE_INDEX_1",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
               "EMBED_LO", "EMBED_HI", "OUTPUT_LO", "OUTPUT_HI", "CONST"},
        writes={"EMBED_LO", "EMBED_HI", "AX_CARRY_LO", "AX_CARRY_HI",
                "AX_FULL_LO", "AX_FULL_HI", "OUTPUT_LO", "OUTPUT_HI",
                "TEMP", "ADDR_KEY"},
        kind="attn",
        layer_idx=3,
        bake_fn=bake,
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        migrated=True,
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def _stack0_carry_head_spec(BD) -> DeclarativeAttentionHeadSpec:
    """Declarative L3 head 4: previous STACK0 byte0 -> current STACK0 marker."""

    L = 15.0
    q = [AP(0, BD.MARK_STACK0, L)]
    k = [AP(0, BD.STACK0_BYTE0, L), AP(33, BD.CONST, L)]
    v = []
    o = []
    for k_idx in range(16):
        v.append(AP(1 + k_idx, BD.EMBED_LO + k_idx, 1.0))
        v.append(AP(17 + k_idx, BD.EMBED_HI + k_idx, 1.0))
        o.append(AO(BD.EMBED_LO + k_idx, 1 + k_idx, 1.0))
        o.append(AO(BD.EMBED_HI + k_idx, 17 + k_idx, 1.0))
    q.append(AP(33, BD.MARK_STACK0, L))
    q.append(AP(33, BD.CONST, -L / 2))
    return DeclarativeAttentionHeadSpec(
        head_idx=4,
        q=tuple(q),
        k=tuple(k),
        v=tuple(v),
        o=tuple(o),
    )


def _ax_full_relay_head_spec(BD) -> DeclarativeAttentionHeadSpec:
    """Declarative L3 head 5: current AX output byte -> AX_FULL."""

    L = 15.0
    GATE = 33
    q = [
        AP(0, BD.MARK_AX, L),
        AP(0, BD.HAS_SE, L),
        AP(0, BD.CONST, -L * 1.5),
        AP(GATE, BD.MARK_AX, L),
        AP(GATE, BD.CONST, -L / 2),
    ]
    k = [AP(0, BD.MARK_AX, L), AP(GATE, BD.CONST, L)]
    v = []
    o = []
    for k_idx in range(16):
        v.append(AP(1 + k_idx, BD.OUTPUT_LO + k_idx, 1.0))
        v.append(AP(17 + k_idx, BD.OUTPUT_HI + k_idx, 1.0))
        o.append(AO(BD.AX_FULL_LO + k_idx, 1 + k_idx, 1.0))
        o.append(AO(BD.AX_FULL_HI + k_idx, 17 + k_idx, 1.0))
    return DeclarativeAttentionHeadSpec(
        head_idx=5,
        q=tuple(q),
        k=tuple(k),
        v=tuple(v),
        o=tuple(o),
    )


def _lev_bp_to_pc_head_spec(BD) -> DeclarativeAttentionHeadSpec:
    """Declarative L3 head 6: BP byte carry source for LEV return address."""

    L = 15.0
    BP_I = 3
    GATE = 33
    q = [
        AP(0, BD.MARK_PC, L),
        AP(0, BD.OP_LEV, L / 5),
        AP(0, BD.CONST, -L * 1.5),
        AP(GATE, BD.MARK_PC, L),
        AP(GATE, BD.CONST, -L / 2),
    ]
    k = [
        AP(0, BD.L1H1 + BP_I, L),
        AP(0, BD.L1H0 + BP_I, -L),
        AP(GATE, BD.CONST, L),
    ]
    v = []
    for k_idx in range(16):
        v.append(AP(1 + k_idx, BD.CLEAN_EMBED_LO + k_idx, 1.0))
        v.append(AP(17 + k_idx, BD.CLEAN_EMBED_HI + k_idx, 1.0))
    return DeclarativeAttentionHeadSpec(
        head_idx=6,
        q=tuple(q),
        k=tuple(k),
        v=tuple(v),
    )


def _pc_byte1_prev_head_spec(BD) -> DeclarativeAttentionHeadSpec:
    """Declarative L3 head 7: previous PC byte1 -> current PC byte0 TEMP.

    The L3 FFN predicts PC byte1 at the PC byte0 position. To preserve byte1
    after PC has crossed 0x100, that FFN needs the previous state's byte1; this
    head copies it from the prior PC byte1 token into TEMP at PC byte0 rows.
    It also stages the low nibble of byte1 into ADDR_KEY[32..47] at PC rows so
    later declarative fetch heads can match the full 12-bit code address.
    """

    L = 15.0
    PC_I = 0
    GATE = 33
    v = []
    o = []
    for k_idx in range(16):
        v.append(AP(1 + k_idx, BD.CLEAN_EMBED_LO + k_idx, 1.0))
        v.append(AP(17 + k_idx, BD.CLEAN_EMBED_HI + k_idx, 1.0))
        o.append(AO(BD.TEMP + k_idx, 1 + k_idx, 1.0))
        o.append(AO(BD.TEMP + 16 + k_idx, 17 + k_idx, 1.0))
        o.append(AO(BD.ADDR_KEY + 32 + k_idx, 1 + k_idx, 1.0))

    code_prefix_blockers = tuple(
        AP(0, BD.ADDR_KEY + k_idx, -L)
        for k_idx in range(48)
    )

    return DeclarativeAttentionHeadSpec(
        head_idx=7,
        q=(
            AP(0, BD.IS_BYTE, L),
            AP(0, BD.H1 + PC_I, L),
            AP(0, BD.BYTE_INDEX_0, L),
            AP(0, BD.MARK_PC, 3.0 * L),
            AP(0, BD.HAS_SE, L),
            AP(0, BD.CONST, -3.0 * L),
            *code_prefix_blockers,
            AP(GATE, BD.IS_BYTE, 500.0),
            AP(GATE, BD.MARK_PC, 500.0),
            AP(GATE, BD.CONST, -500.0),
        ),
        k=(
            AP(0, BD.IS_BYTE, L),
            AP(0, BD.H1 + PC_I, L),
            AP(0, BD.BYTE_INDEX_1, L),
            AP(0, BD.CONST, -2.0 * L),
            AP(GATE, BD.CONST, 5.0),
        ),
        v=tuple(v),
        o=tuple(o),
    )


def make_layer3_convo_io_state_init_op(
    enable_conversational_io: bool = False,
) -> Operation:
    """L3 FFN addition: initialize output mode when LAST_WAS_THINKING_END.

    Originally an inline call in ``set_vm_weights`` (gated by
    ``enable_conversational_io``):
        _set_conversational_io_state_init(ffn3, S, BD)

    Migrated as ``kind="block"`` pinned to ``layer_idx=3`` with
    ``migrated=True``. Phase=3.1 so this runs AFTER
    ``make_layer3_ffn_op`` (phase=3) and writes into a distinct FFN
    unit range (starts at unit 1034, above the L3 / L6-routing unit
    counters), so the writes layer cleanly on top.

    The bake is unconditional in shape (always registered to keep the
    dep-graph stable), but the body is a no-op when
    ``enable_conversational_io`` is False so no FFN units are touched
    outside of conversational-I/O mode.
    """
    def bake(block, dim_positions, S):
        if not enable_conversational_io:
            return
        proxy = _as_setdim_proxy(dim_positions)
        _lower_layer3_convo_io_state_init_ir(block.ffn, S, proxy)

    return Operation(
        name="layer3_convo_io_state_init",
        phase=3.1,
        # Reads/writes use LAST_WAS_THINKING_END and IO_IN_OUTPUT_MODE,
        # which are not declared in declare_setdim_compat_dims
        # (conversational-I/O-only dims); the bake resolves them via the
        # _SetDim fallback in _as_setdim_proxy, so no compiler-tracked
        # edges are needed.
        reads=set(),
        writes=set(),
        kind="block",
        layer_idx=3,
        bake_fn=bake,
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        migrated=True,
        ffn_units_used=1035 if enable_conversational_io else None,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#printing-and-reading-input",
    )


def _layer3_convo_io_state_init_rules(S: float) -> tuple[FFNRule, ...]:
    return (
        FFNRule.constant_write(
            name="convo_io_enter_output_mode",
            conditions=(("LAST_WAS_THINKING_END", 1.0),),
            threshold=0.5,
            writes=(("IO_IN_OUTPUT_MODE", 2.0 / S),),
        ),
    )


def _lower_layer3_convo_io_state_init_ir(ffn, S: float, BD) -> int:
    rules = _layer3_convo_io_state_init_rules(S)
    dim_positions = Primitives.dim_positions_from_bd(
        BD,
        Primitives.ffn_rule_dim_names(rules),
    )
    return Primitives.lower_ffn_rules(
        ffn,
        rules,
        dim_positions,
        start_unit=1034,
        S=S,
    )
