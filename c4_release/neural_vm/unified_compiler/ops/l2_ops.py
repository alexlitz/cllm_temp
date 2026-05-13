"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ..layer_compiler import Operation
from .shared import _as_setdim_proxy


def make_layer2_mem_byte_flags_op() -> Operation:
    """L2 FFN: MEM val byte position flags + extended BYTE_INDEX for STACK0.

    Uses units [0, 8). Records the next free unit on
    ``ffn._l2_unit_counter`` so chained ops (e.g.
    ``make_layer2_initial_pc_bake_cancel_op``) can start above this range.
    """
    def bake(ffn, dim_positions, S):
        from ...vm_step import _set_layer2_mem_byte_flags
        _set_layer2_mem_byte_flags(ffn, S, _as_setdim_proxy(dim_positions))
        # The legacy bake fills units 0..7 (4 MEM_VAL_BN + 4 BYTE_INDEX_*).
        ffn._l2_unit_counter = max(getattr(ffn, "_l2_unit_counter", 0), 8)

    # Dim-ownership claims: ``_set_layer2_mem_byte_flags`` writes units 0..7
    # (see setup_helpers.py:_set_layer2_mem_byte_flags). Each unit writes a
    # unique W_down output dim:
    #   unit 0: MEM_VAL_B0
    #   unit 1: MEM_VAL_B1
    #   unit 2: MEM_VAL_B2
    #   unit 3: MEM_VAL_B3
    #   unit 4: BYTE_INDEX_0  (STACK0 byte 0)
    #   unit 5: BYTE_INDEX_1  (STACK0 byte 1)
    #   unit 6: BYTE_INDEX_2  (STACK0 byte 2)
    #   unit 7: BYTE_INDEX_3  (STACK0 byte 3)
    _claims = set()
    _outputs = [
        "MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
        "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
    ]
    for u, out_dim in enumerate(_outputs):
        _claims.add((2, "ffn_W_down", str(u), f"{out_dim}+0"))

    return Operation(
        name="layer2_mem_byte_flags",
        phase=2,
        reads={"H0", "H1", "H4", "IS_BYTE", "BYTE_INDEX_0",
               "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3"},
        writes={"MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
                "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3"},
        kind="ffn",
        layer_idx=2,
        bake_fn=bake,
        migrated=True,
        claims=_claims,
        # ``_set_layer2_mem_byte_flags`` writes 8 units (4 MEM_VAL_B* +
        # 4 BYTE_INDEX_*); ``_l2_unit_counter`` is bumped to 8 after this op
        # so the cancel op below starts at unit 8.
        ffn_units_used=8,
        # Staleness invariants: MEM_VAL_B0..3 fire at the token positions
        # immediately preceding each MEM val byte (d=4..7 from REG_MEM),
        # so the LM head can predict each val byte. L13/L14 mem-write
        # paths consume MEM_VAL_B* as "we are at val byte N" position
        # selectors.
        produces={
            "MEM_VAL_B0": "MEM_ADDR_byte3",
            "MEM_VAL_B1": "MEM_VAL_byte0",
            "MEM_VAL_B2": "MEM_VAL_byte1",
            "MEM_VAL_B3": "MEM_VAL_byte2",
        },
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#memory",
    )


def make_layer2_initial_pc_bake_cancel_op() -> Operation:
    """L2 FFN: cancel the REG_PC token-embedding initial-PC bake at MARK_PC AND HAS_SE.

    The token-embedding bake ``make_initial_pc_bake_op`` (phase=1001.5) adds
    ``EMBED_LO[PC_OFFSET & 0xF] += 1.0`` and ``EMBED_HI[(PC_OFFSET >> 4) & 0xF]
    += 1.0`` at EVERY REG_PC token. This is intentional for step 0 (the
    first PC marker needs PC=PC_OFFSET in EMBED so L4 attention can relay
    it to AX for L5's first-step opcode fetch), but at step 1+ it pollutes
    the residual stream — specifically EMBED_LO[init_pc_lo] persists at
    +1.0 even though the carry-forward attention has written the real
    prev-PC nibbles into EMBED_LO/HI.

    The legacy fix inlined two cancel units in ``_set_layer3_ffn`` (vm_step
    lines ~2598-2609) that subtract from EMBED_LO/HI at MARK_PC AND
    HAS_SE. Those cancels live in the SAME FFN block as the PC INCREMENT
    units, so the FFN's input residual still carries the +1.0 spurious
    contribution — and PC INCREMENT (which reads EMBED_LO[k] for every
    nibble k) leaks a phantom +1.0 contribution at OUTPUT_LO[
    (init_pc_lo + INSTR_WIDTH) % 16]. For PC_OFFSET=2, INSTR_WIDTH=8 the
    leak lands at OUTPUT_LO[10], aliasing with the FIRST-STEP DEFAULT's
    target (also OUTPUT_LO[10]) and corrupting step-1+ PC byte 0 by
    +INSTR_WIDTH, breaking ``test_two_imms``, ``test_jmp_from_step_2``,
    ``test_jmp_backward``, and the smoke ``test_add_basic`` multi-step.

    Cancelling the bake at L2 FFN (which runs strictly BEFORE L3 FFN)
    makes L3 see EMBED_LO[init_pc_lo]=0 at step 1+ MARK_PC, eliminating
    the leak entirely. Step 0's MARK_PC has HAS_SE=0 (no STEP_END in
    context yet), so the cancel does not fire and EMBED_LO[init_pc_lo]
    remains +1.0 for L4 first-step relay. The strength is -2.0/S × 50 =
    -1.0 (exact match for the +1.0 bake).

    Pinned to ``layer_idx=2``, ``kind="block"`` (so the bake can access
    the same FFN as ``make_layer2_mem_byte_flags_op`` and chain unit
    allocation via ``ffn._l2_unit_counter``). Phase=2.5 so it runs after
    ``make_layer2_mem_byte_flags_op`` (phase=2).
    """
    def bake(block, dim_positions, S):
        from ...constants import PC_OFFSET

        def D(name):
            if dim_positions is not None and name in dim_positions:
                return dim_positions[name]
            from ...vm_step import _SetDim
            return getattr(_SetDim, name)

        ffn = block.ffn
        unit = getattr(ffn, "_l2_unit_counter", 0)
        init_pc_lo = PC_OFFSET & 0xF
        init_pc_hi = (PC_OFFSET >> 4) & 0xF

        MARK_PC = D("MARK_PC")
        HAS_SE = D("HAS_SE")
        EMBED_LO = D("EMBED_LO")
        EMBED_HI = D("EMBED_HI")

        # Cancel EMBED_LO[init_pc_lo] when MARK_PC AND HAS_SE.
        # up = S * HAS_SE - S/2 → +S/2 at HAS_SE=1, silu(+S/2) ≈ S/2.
        # gate = MARK_PC → 1.0 at PC marker.
        # hidden = S/2, W_down = -2.0/S → contribution = -1.0.
        ffn.W_up.data[unit, HAS_SE] = S
        ffn.b_up.data[unit] = -S * 0.5
        ffn.W_gate.data[unit, MARK_PC] = 1.0
        ffn.W_down.data[EMBED_LO + init_pc_lo, unit] = -2.0 / S
        unit += 1

        # Cancel EMBED_HI[init_pc_hi] (mirror of the LO cancel).
        ffn.W_up.data[unit, HAS_SE] = S
        ffn.b_up.data[unit] = -S * 0.5
        ffn.W_gate.data[unit, MARK_PC] = 1.0
        ffn.W_down.data[EMBED_HI + init_pc_hi, unit] = -2.0 / S
        unit += 1

        ffn._l2_unit_counter = unit

    # Dim-ownership claims: two FFN units (allocated at the L2 unit counter,
    # which is 8 after _set_layer2_mem_byte_flags). Each writes one EMBED
    # nibble.  PC_OFFSET is a runtime constant from constants.py, so resolve
    # init_pc_lo/hi at op-construction time for the claim columns.
    from ...constants import PC_OFFSET as _PC_OFFSET
    _init_pc_lo = _PC_OFFSET & 0xF
    _init_pc_hi = (_PC_OFFSET >> 4) & 0xF
    _claims = set()
    # Unit 8: cancels EMBED_LO[init_pc_lo]. W_down[EMBED_LO+init_pc_lo, 8].
    _claims.add((2, "ffn_W_down", "8", f"EMBED_LO+{_init_pc_lo}"))
    # Unit 9: cancels EMBED_HI[init_pc_hi].
    _claims.add((2, "ffn_W_down", "9", f"EMBED_HI+{_init_pc_hi}"))

    return Operation(
        name="layer2_initial_pc_bake_cancel",
        phase=2.5,
        reads={"MARK_PC", "HAS_SE"},
        writes={"EMBED_LO", "EMBED_HI"},
        kind="block",
        layer_idx=2,
        bake_fn=bake,
        migrated=True,
        claims=_claims,
        # Allocates 2 FFN units starting at ``ffn._l2_unit_counter`` (which
        # ``make_layer2_mem_byte_flags_op`` sets to 8). Result: writes units
        # 8 and 9 -> max index 10. The aggregator takes the per-block max
        # across all annotated ops, so reporting 10 here covers both.
        ffn_units_used=10,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#baking-prompts-programs-into-the-transformer-weights",
    )


def make_layer2_threshold_attn_op() -> Operation:
    """L2 attention: threshold 5.5 head."""
    def bake(attn, dim_positions, S):
        from ...vm_step import _set_threshold_attn
        proxy = _as_setdim_proxy(dim_positions)
        ALIBI_S = 10.0
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes.fill_(ALIBI_S)
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_threshold_attn(
            attn, [5.5], [proxy.L2H0], ALIBI_S, HD, heads=[0], BD=proxy,
        )

    # Dim-ownership claims: 1 threshold head on L2 attn, head 0 writing L2H0.
    _claims = set()
    _MARKS = ["MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP",
              "MARK_MEM", "MARK_SE", "MARK_CS"]
    for m, mark in enumerate(_MARKS):
        _claims.add((2, "attn_W_v", f"0_{1 + m}", f"{mark}+0"))
        _claims.add((2, "attn_W_o", f"0_{1 + m}", f"L2H0+{m}"))
    _claims.add((2, "attn_W_q", "0_0", "CONST+0"))
    _claims.add((2, "attn_W_k", "0_0", "IS_MARK+0"))

    return Operation(
        name="layer2_threshold_attn",
        phase=2,
        reads={"IS_MARK", "CONST"},
        writes={"L2H0"},
        kind="attn",
        layer_idx=2,
        bake_fn=bake,
        migrated=True,
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#the-attention-layer",
    )


def make_layer2_lookback_detection_head_op(
    enable_conversational_io: bool = False,
) -> Operation:
    """L2 attention head 1: detect previous token type for conversational I/O.

    Originally an inline call in ``set_vm_weights`` (gated by
    ``enable_conversational_io``):
        attn2.alibi_slopes[1] = 10.0
        _set_lookback_detection_head(attn2, S, BD, HD)

    Migrated as ``kind="block"`` pinned to ``layer_idx=2`` with
    ``migrated=True``. Phase=2.1 so this runs AFTER
    ``make_layer2_threshold_attn_op`` (phase=2), which fills all alibi
    slopes to 10.0 — the explicit ``[1] = 10.0`` is therefore a no-op
    today but preserved verbatim for parity with the legacy inline
    setup.

    The bake is unconditional in shape (always registered to keep the
    dep-graph stable), but the body is a no-op when
    ``enable_conversational_io`` is False so no weights are touched
    outside of conversational-I/O mode.
    """
    def bake(block, dim_positions, S):
        if not enable_conversational_io:
            return
        attn = block.attn
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes[1] = 10.0  # Steep slope to favor most recent token
        proxy = _as_setdim_proxy(dim_positions)
        HD = attn.W_q.shape[0] // attn.num_heads
        from ...vm_step import _set_lookback_detection_head
        _set_lookback_detection_head(attn, S, proxy, HD)

    return Operation(
        name="layer2_lookback_detection_head",
        phase=2.1,
        # Reads: CONST (Q/K gate), MARK_THINKING_START/END + IS_BYTE (V copy).
        # Writes go to LAST_WAS_THINKING_START/END/BYTE which are not
        # declared in declare_setdim_compat_dims (conversational-I/O-only
        # dims); the bake resolves them via the _SetDim fallback in
        # _as_setdim_proxy, so no compiler-tracked write edge is needed.
        reads={"CONST", "MARK_THINKING_START", "MARK_THINKING_END", "IS_BYTE"},
        writes=set(),
        kind="block",
        layer_idx=2,
        bake_fn=bake,
        migrated=True,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#the-attention-layer",
    )


