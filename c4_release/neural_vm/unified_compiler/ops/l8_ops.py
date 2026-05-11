"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ..layer_compiler import Operation
from .shared import _as_setdim_proxy


def make_layer8_alu_op() -> Operation:
    """L8 FFN: ADD/SUB lo nibble + carry/borrow + LEA + CMP_GROUP.

    MIGRATED 2026-05-10 (Wave 2 Unit 10): flipped from kind="ffn" to
    kind="block" with layer_idx=8, phase=8.2, migrated=True. The inline
    call ``_set_layer8_alu(ffn8, S, BD)`` in ``set_vm_weights`` (both
    ``alu_mode == 'lookup'`` and ``alu_mode == 'efficient'`` branches)
    has been removed; this op now owns the bake. Phase=8.2 places it
    after format_pointer_extraction (7.5) and the L8 multibyte_fetch
    bake (8.1), and before format_position_counter (8.5) — matching the
    legacy in-set_vm_weights ordering.
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer8_alu
        _set_layer8_alu(block.ffn, S, _as_setdim_proxy(dim_positions))

    return Operation(
        name="layer8_alu",
        phase=8.2,
        reads={"MARK_AX", "MARK_PC", "ALU_LO", "AX_CARRY_LO", "FETCH_LO",
               "OP_ADD", "OP_SUB", "OP_LEA",
               "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE"},
        writes={"OUTPUT_LO", "CARRY", "CMP_GROUP"},
        kind="block",
        bake_fn=bake,
        layer_idx=8,
        migrated=True,
    )


def make_format_position_counter_op(enable_conversational_io: bool = False) -> Operation:
    """L8 FFN: increment IO_FORMAT_POS after each output byte emission.

    Originally an inline call in ``set_vm_weights`` (nested under
    ``alu_mode == 'lookup'`` + ``enable_conversational_io``):
        ``_set_format_position_counter(ffn8, S, BD)``.

    Migrated as ``kind="block"`` pinned to ``layer_idx=8`` with
    ``migrated=True``. Registered unconditionally; the bake is a no-op
    when ``enable_conversational_io`` is False. The original lookup-mode
    nesting was a side-effect of convo-io initially being co-located with
    the lookup ALU bake — the helper itself writes IO_FORMAT_POS units
    (starting at unit 600) and has no dependency on lookup-mode-specific
    weights, so this op fires regardless of alu_mode whenever the flag
    is set. Phase=8.5 so this runs AFTER ``layer8_alu`` (phase=8) since
    the position-counter units (600-615) intentionally overwrite a slice
    of the ADD-carry block, matching legacy behavior.
    """
    def bake(block, dim_positions, S):
        if not enable_conversational_io:
            return
        from ...vm_step import _set_format_position_counter
        _set_format_position_counter(block.ffn, S, _as_setdim_proxy(dim_positions))

    return Operation(
        name="format_position_counter",
        phase=8.5,
        reads={"LAST_WAS_BYTE", "IO_IN_OUTPUT_MODE", "IO_FORMAT_POS"},
        writes={"IO_FORMAT_POS"},
        kind="block",
        bake_fn=bake,
        layer_idx=8,
        migrated=True,
    )


def make_layer8_multibyte_fetch_op() -> Operation:
    """No-op dep anchor for ``layer8_multibyte_fetch_bake``.

    Kept as a ``kind="attn"`` placeholder (no ``migrated`` flag, no
    ``layer_idx``): its declared reads/writes preserve the LayerCompiler
    dep-graph topology that places downstream ops at the right
    model.blocks indices. The actual weight bake now happens in
    ``make_layer8_multibyte_fetch_bake_op`` (kind="block", layer_idx=8,
    phase=8.1, migrated=True); this op's bake_fn is a no-op.
    """
    def bake(attn, dim_positions, S):
        # No-op: actual bake is in `layer8_multibyte_fetch_bake` block op.
        return

    return Operation(
        name="layer8_multibyte_fetch",
        phase=8,
        reads={"FETCH_LO", "FETCH_HI", "ADDR_KEY", "IS_BYTE", "H1",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"AX_CARRY_LO", "AX_CARRY_HI"},
        kind="attn",
        bake_fn=bake,
    )


def make_layer8_multibyte_fetch_bake_op() -> Operation:
    """Bake ``_set_layer8_multibyte_fetch`` into ``model.blocks[8].attn``.

    Originally an inline call in ``set_vm_weights``:
        ``_set_layer8_multibyte_fetch(attn8, S, BD, HD)``

    Migrated as ``kind="block"`` with ``layer_idx=8`` and ``migrated=True``:
    the inline call has been removed, so the bake must happen here.
    Phase=8.1 so this runs after ``layer8_sp_gather_bake`` (phase=8.0),
    preserving the legacy in-set_vm_weights ordering. The dep anchor
    ``layer8_multibyte_fetch`` (kind="attn") preserves the LayerCompiler
    topology so downstream ops remain placed at their legacy blocks.
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer8_multibyte_fetch
        proxy = _as_setdim_proxy(dim_positions)
        attn = block.attn
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer8_multibyte_fetch(attn, S, proxy, HD)

    return Operation(
        name="layer8_multibyte_fetch_bake",
        phase=8.1,
        reads={"FETCH_LO", "FETCH_HI", "ADDR_KEY", "IS_BYTE", "H1",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI", "CONST"},
        writes={"AX_CARRY_LO", "AX_CARRY_HI"},
        kind="block",
        bake_fn=bake,
        layer_idx=8,
        migrated=True,
    )


def make_layer8_multibyte_routing_op() -> Operation:
    """L8 FFN extension: route FETCH → OUTPUT at AX byte positions for IMM.

    MIGRATED 2026-05-10 (Wave 2 Unit 10): flipped from kind="ffn" to
    kind="block" with layer_idx=8, phase=8.3, migrated=True. The inline
    call ``_set_layer8_multibyte_routing(ffn8, S, BD)`` in
    ``set_vm_weights`` (both alu_mode branches) has been removed; this op
    now owns the bake. Phase=8.3 places it after ``layer8_alu`` (8.2)
    so the shared unit counter starts after the ALU units (the helper
    internally re-invokes ``_set_layer8_alu`` to compute ``unit_start``;
    that re-call is an idempotent overwrite of the same ALU weights).
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer8_multibyte_routing
        _set_layer8_multibyte_routing(block.ffn, S, _as_setdim_proxy(dim_positions))

    return Operation(
        name="layer8_multibyte_routing",
        phase=8.3,
        reads={"IS_BYTE", "H1", "OP_IMM", "MARK_AX",
               "AX_CARRY_LO", "AX_CARRY_HI"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        bake_fn=bake,
        layer_idx=8,
        migrated=True,
    )


def make_layer8_sp_gather_op() -> Operation:
    """No-op dep anchor for ``layer8_sp_gather_bake``.

    Kept as a ``kind="attn"`` placeholder (no ``migrated`` flag, no
    ``layer_idx``): its declared reads/writes preserve the LayerCompiler
    dep-graph topology. The actual weight bake now happens in
    ``make_layer8_sp_gather_bake_op`` (kind="block", layer_idx=8,
    phase=8.0, migrated=True); this op's bake_fn is a no-op.
    """
    def bake(attn, dim_positions, S):
        # No-op: actual bake is in `layer8_sp_gather_bake` block op.
        return

    return Operation(
        name="layer8_sp_gather",
        phase=8,
        reads={"MARK_AX", "MARK_SP", "OP_ADJ", "OP_ENT", "OP_LEA",
               "EMBED_LO", "EMBED_HI"},
        writes={"ALU_LO", "ALU_HI"},
        kind="attn",
        bake_fn=bake,
    )


def make_layer8_sp_gather_bake_op() -> Operation:
    """Bake ``_set_layer8_sp_gather`` into ``model.blocks[8].attn``.

    Originally an inline call in ``set_vm_weights``:
        ``_set_layer8_sp_gather(attn8, S, BD, HD)``

    Migrated as ``kind="block"`` with ``layer_idx=8`` and ``migrated=True``:
    the inline call has been removed, so the bake must happen here.
    Phase=8.0 so this runs before ``layer8_multibyte_fetch_bake``
    (phase=8.1), preserving the legacy in-set_vm_weights ordering. The
    dep anchor ``layer8_sp_gather`` (kind="attn") preserves the
    LayerCompiler topology so downstream ops remain placed at their
    legacy blocks.
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer8_sp_gather
        proxy = _as_setdim_proxy(dim_positions)
        attn = block.attn
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer8_sp_gather(attn, S, proxy, HD)

    return Operation(
        name="layer8_sp_gather_bake",
        phase=8.0,
        reads={"MARK_STACK0", "MARK_BP", "H1", "H3", "H4",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI", "CONST"},
        writes={"ADDR_B0_LO", "ADDR_B0_HI",
                "ADDR_B1_LO", "ADDR_B1_HI",
                "ADDR_B2_LO", "ADDR_B2_HI"},
        kind="block",
        bake_fn=bake,
        layer_idx=8,
        migrated=True,
    )


def make_layer8_op_imm_relay_op() -> Operation:
    """L8 head 4: Relay OP_IMM from AX marker to AX byte positions.

    Migrated 2026-05-11 from the inline block at vm_step.py:2029-2043 that
    programmed attn8 head 4's Q/K/V/O slots for OP_IMM relay (with a GATE4
    sub-head). At AX byte positions (IS_BYTE + H1[AX_I]), this head attends
    to AX marker (MARK_AX) and copies OP_IMM to byte positions.

    Phase=8.4 places it AFTER ``layer8_multibyte_routing`` (8.3) but BEFORE
    the L8 alu_postop_attach (phase 8.5) so attention bakes complete before
    the FFN wrap.
    """
    def _bake(block, dim_positions, S):
        BD = _as_setdim_proxy(dim_positions)
        attn8 = block.attn
        HD = attn8.W_q.shape[0] // attn8.num_heads
        base = 4 * HD
        AX_I = 1
        L8_relay = 20.0
        attn8.W_q[base, BD.IS_BYTE] = L8_relay
        attn8.W_q[base, BD.H1 + AX_I] = L8_relay
        attn8.W_q[base, BD.CONST] = -L8_relay * 1.5
        attn8.W_k[base, BD.MARK_AX] = L8_relay
        attn8.W_k[base, BD.IS_BYTE] = -L8_relay * 10
        attn8.W_k[base, BD.CONST] = L8_relay * 0.5
        attn8.W_v[base, BD.OP_IMM] = 1.0
        attn8.W_o[BD.OP_IMM, base] = 1.0
        GATE4 = 1
        attn8.W_q[base + GATE4, BD.IS_BYTE] = 500.0
        attn8.W_q[base + GATE4, BD.CONST] = -500.0
        attn8.W_k[base + GATE4, BD.CONST] = 5.0

    return Operation(
        name="layer8_op_imm_relay",
        reads={"IS_BYTE", "H1", "MARK_AX", "OP_IMM", "CONST"},
        writes={"OP_IMM"},
        kind="block",
        bake_fn=_bake,
        layer_idx=8,
        phase=8.4,
        migrated=True,
    )


