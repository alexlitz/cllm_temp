"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ..layer_compiler import Operation
from .shared import _as_setdim_proxy


def make_layer3_ffn_op() -> Operation:
    """L3 FFN: PC/SP/BP first-step defaults + PC byte-0 increment.

    Originally: `_set_layer3_ffn` at vm_step.py:2929.

    Reads MARK_PC, MARK_SP, MARK_BP, MARK_STACK0, HAS_SE, EMBED_LO/HI,
    H1, H4, BYTE_INDEX_*, OP_LEV.
    Writes OUTPUT_LO/HI, EMBED_LO/HI.

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

    return Operation(
        name="layer3_ffn",
        phase=3,
        reads={"MARK_PC", "MARK_SP", "MARK_BP", "MARK_STACK0", "HAS_SE",
               "EMBED_LO", "EMBED_HI", "H1", "H4", "OP_LEV",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2"},
        writes={"OUTPUT_LO", "OUTPUT_HI", "EMBED_LO", "EMBED_HI"},
        kind="block",
        layer_idx=3,
        bake_fn=bake,
        migrated=True,
    )


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
               "EMBED_LO", "EMBED_HI", "H1", "H4", "OP_LEV",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2"},
        writes={"OUTPUT_LO", "OUTPUT_HI", "EMBED_LO", "EMBED_HI"},
        kind="ffn",
        bake_fn=bake,
    )


def make_layer3_carry_forward_attn_op() -> Operation:
    """L3 attention: 7 carry-forward heads (PC, AX, SP, BP, STACK0 + relays).

    Heads 0-3 use ``Primitives.carry_forward_attention`` (byte-identical to
    the legacy ``_set_carry_forward_attn`` — see
    ``tests/test_primitives_l3_carry_equivalence.py``). Head 4 uses
    ``_set_stack0_carry_attn`` (different K source). Heads 5-6 stay inline
    (head 5 reads OUTPUT_*, head 6 has OP_LEV gating + CLEAN_EMBED_*).
    """
    def bake(attn, dim_positions, S):
        from ...vm_step import _set_stack0_carry_attn
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
        _set_stack0_carry_attn(attn, 4, HD, BD=proxy)
        # Heads 5-6: AX_FULL relay + BP→PC for LEV. These reference _SetDim
        # directly inside _set_carry_forward_attn so the proxy fallback handles
        # them. For now we replicate the inline code from _set_layer3_attn block:
        L = 15.0
        base = 5 * HD
        attn.W_q[base, proxy.MARK_AX] = L
        attn.W_q[base, proxy.HAS_SE] = L
        attn.W_q[base, proxy.CONST] = -L * 1.5
        attn.W_k[base, proxy.MARK_AX] = L
        for k in range(16):
            attn.W_v[base + 1 + k, proxy.OUTPUT_LO + k] = 1.0
            attn.W_v[base + 17 + k, proxy.OUTPUT_HI + k] = 1.0
        for k in range(16):
            attn.W_o[proxy.AX_FULL_LO + k, base + 1 + k] = 1.0
            attn.W_o[proxy.AX_FULL_HI + k, base + 17 + k] = 1.0
        GATE = 33
        attn.W_q[base + GATE, proxy.MARK_AX] = L
        attn.W_q[base + GATE, proxy.CONST] = -L / 2
        attn.W_k[base + GATE, proxy.CONST] = L
        # Head 6: BP carry to PC marker for LEV return_addr
        base = 6 * HD
        attn.W_q[base, proxy.MARK_PC] = L
        attn.W_q[base, proxy.OP_LEV] = L / 5
        attn.W_q[base, proxy.CONST] = -L * 1.5
        attn.W_k[base, proxy.L1H1 + BP_I] = L
        attn.W_k[base, proxy.L1H0 + BP_I] = -L
        for k in range(16):
            attn.W_v[base + 1 + k, proxy.CLEAN_EMBED_LO + k] = 1.0
            attn.W_v[base + 17 + k, proxy.CLEAN_EMBED_HI + k] = 1.0
        attn.W_q[base + GATE, proxy.MARK_PC] = L
        attn.W_q[base + GATE, proxy.CONST] = -L / 2
        attn.W_k[base + GATE, proxy.CONST] = L

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
    #   Head 4 (STACK0): _set_stack0_carry_attn — see helper for details
    #   Head 5 (AX_FULL): inline V[OUTPUT_LO/HI] → AX_FULL_LO/HI
    #   Head 6 (BP→PC LEV): inline V[CLEAN_EMBED_LO/HI] → (out via inline)
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

    return Operation(
        name="layer3_carry_forward_attn",
        phase=3,
        reads={"MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP",
               "L1H0", "L1H1", "STACK0_BYTE0", "OP_LEV", "HAS_SE",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
               "EMBED_LO", "EMBED_HI", "OUTPUT_LO", "OUTPUT_HI", "CONST"},
        writes={"EMBED_LO", "EMBED_HI", "AX_CARRY_LO", "AX_CARRY_HI",
                "AX_FULL_LO", "AX_FULL_HI"},
        kind="attn",
        layer_idx=3,
        bake_fn=bake,
        migrated=True,
        claims=_claims,
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
        from ...vm_step import _set_conversational_io_state_init
        _set_conversational_io_state_init(block.ffn, S, proxy)

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
        migrated=True,
    )


