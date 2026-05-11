"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ..layer_compiler import Operation
from .shared import _as_setdim_proxy


def make_layer9_alu_op() -> Operation:
    """L9 FFN: ADD/SUB hi nibble + bitwise ops byte 0, plus marker suppression.

    Originally two inline calls inside ``set_vm_weights`` (in the
    ``alu_mode == 'lookup'`` branch):
        ``n9 = _set_layer9_alu(ffn9, S, BD)``
        ``_set_layer9_marker_suppress(ffn9, S, BD, n9)``

    Combined into a single migrated bake_fn that captures ``n9`` and
    threads it to ``_set_layer9_marker_suppress`` as ``start_unit`` so the
    two routines share the FFN's hidden-unit allocator. Mirrors the
    combined-bake pattern proven safe by Unit 9's diagnosis (see
    ``c4_release/docs/LOOKUP_MODE_BUG_DIAGNOSIS.md``).

    Migrated as ``kind="block"`` pinned to ``layer_idx=9`` with
    ``migrated=True``; the inline call pair has been removed from
    ``set_vm_weights`` to avoid double-bake. Phase stays at 9. Fires in
    both lookup and efficient ALU modes — the lookup-branch nesting was
    incidental and the helpers themselves are alu_mode-agnostic.
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer9_alu, _set_layer9_marker_suppress
        proxy = _as_setdim_proxy(dim_positions)
        n9 = _set_layer9_alu(block.ffn, S, proxy)
        _set_layer9_marker_suppress(block.ffn, S, proxy, n9)

    return Operation(
        name="layer9_alu",
        phase=9,
        reads={"MARK_AX", "MARK_PC", "ALU_HI", "AX_CARRY_HI", "FETCH_HI", "CARRY",
               "OP_ADD", "OP_SUB", "OP_OR", "OP_XOR", "OP_AND",
               "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
               "ALU_LO", "AX_CARRY_LO"},
        writes={"OUTPUT_HI", "CMP", "OUTPUT_LO"},
        kind="block",
        bake_fn=bake,
        layer_idx=9,
        migrated=True,
    )


def make_layer9_lev_addr_relay_op() -> Operation:
    """L9 attention head 0: BP byte 0 → ADDR_B0 at SP marker for LEV.

    Originally an inline call inside ``set_vm_weights`` (in the
    ``alu_mode == 'lookup'`` branch):
        ``_set_layer9_lev_addr_relay(attn9, S, BD, HD)``

    Migrated as ``kind="block"`` pinned to ``layer_idx=9`` with
    ``migrated=True``: the inline call has been removed to avoid
    double-bake. Phase=9.0 to preserve ordering with sibling
    ``layer9_lev_bp_to_pc_relay`` (phase=9.1). Fires in both lookup
    and efficient ALU modes — the helper performs identical setup
    regardless of alu_mode, and the model is built once.

    Also sets ``alibi_slopes[0] = 0.2`` (shallow slope for d=29 relay
    from SP marker back to previous BP byte 0); previously set inline
    alongside the legacy bake call.
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer9_lev_addr_relay
        attn = block.attn
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes[0] = 0.2  # head 0: shallow slope for d=29 relay
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer9_lev_addr_relay(attn, S, _as_setdim_proxy(dim_positions), HD)

    return Operation(
        name="layer9_lev_addr_relay",
        phase=9.0,
        reads={"MARK_SP", "OP_LEV", "L1H1", "BYTE_INDEX_0",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"ADDR_B0_LO", "ADDR_B0_HI"},
        kind="block",
        bake_fn=bake,
        layer_idx=9,
        migrated=True,
    )


def make_layer9_lev_bp_to_pc_relay_op() -> Operation:
    """L9 attention head 1: BP byte 0 → ADDR_B0 at PC marker for LEV return.

    Originally an inline call inside ``set_vm_weights`` (in the
    ``alu_mode == 'lookup'`` branch):
        ``_set_layer9_lev_bp_to_pc_relay(attn9, S, BD, HD)``

    Migrated as ``kind="block"`` pinned to ``layer_idx=9`` with
    ``migrated=True``: the inline call has been removed to avoid
    double-bake. Phase=9.1 so this op runs AFTER
    ``layer9_lev_addr_relay`` (phase=9.0), matching the legacy
    in-set_vm_weights ordering. Fires in both lookup and efficient
    ALU modes — the helper performs identical setup regardless of
    alu_mode.

    Also sets ``alibi_slopes[1] = 0.5`` (BP→PC relay slope for d=15
    tokens); previously set inline alongside the legacy bake call.
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer9_lev_bp_to_pc_relay
        attn = block.attn
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes[1] = 0.5  # head 1: BP→PC relay for LEV (d=15 tokens)
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer9_lev_bp_to_pc_relay(attn, S, _as_setdim_proxy(dim_positions), HD)

    return Operation(
        name="layer9_lev_bp_to_pc_relay",
        phase=9.1,
        reads={"MARK_PC", "OP_LEV", "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
               "L1H1", "BYTE_INDEX_0"},
        writes={"ADDR_B0_LO", "ADDR_B0_HI"},
        kind="block",
        bake_fn=bake,
        layer_idx=9,
        migrated=True,
    )


def make_format_string_fetch_head_op(enable_conversational_io: bool = False) -> Operation:
    """L9 attention head 0: fetch byte from format string at FORMAT_PTR+POS.

    Originally an inline call in ``set_vm_weights`` (nested under
    ``alu_mode == 'lookup'`` + ``enable_conversational_io``):
        ``_set_format_string_fetch_head(attn9, S, BD, HD)``
        plus ``attn9.alibi_slopes.fill_(0.5)``.

    Migrated as ``kind="block"`` pinned to ``layer_idx=9`` with
    ``migrated=True``. Registered unconditionally; the bake is a no-op
    when ``enable_conversational_io`` is False. The original lookup-mode
    nesting was incidental — the helper writes only attn head 0 weights
    and has no dependency on lookup-mode-specific weights, so this op
    fires regardless of alu_mode whenever the flag is set. Phase=9.5 so
    this runs AFTER ``layer9_lev_addr_relay`` (phase=9.0) and
    ``layer9_lev_bp_to_pc_relay`` (phase=9.1); the ``fill_(0.5)`` call
    intentionally clobbers slopes[0] and [1] that those ops set, matching
    legacy ordering (the legacy convo-io block ran ``fill_(0.5)`` AFTER
    the L9 LEV setup as well).
    """
    def bake(block, dim_positions, S):
        if not enable_conversational_io:
            return
        from ...vm_step import _set_format_string_fetch_head
        attn = block.attn
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes.fill_(0.5)
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_format_string_fetch_head(attn, S, _as_setdim_proxy(dim_positions), HD)

    return Operation(
        name="format_string_fetch_head",
        phase=9.5,
        reads={"IO_IN_OUTPUT_MODE", "FORMAT_PTR_LO", "FORMAT_PTR_HI",
               "ADDR_KEY", "EMBED_LO", "EMBED_HI"},
        writes={"OUTPUT_BYTE_LO", "OUTPUT_BYTE_HI"},
        kind="block",
        bake_fn=bake,
        layer_idx=9,
        migrated=True,
    )


def make_layer9_marker_suppress_op() -> Operation:
    """L9 FFN extension: marker suppression."""
    def bake(ffn, dim_positions, S):
        from ...vm_step import _set_layer9_marker_suppress
        # _set_layer9_marker_suppress takes (ffn, S, BD, start_unit). We need
        # to know what start_unit to use. The original code uses unit count
        # after _set_layer9_alu — for the migration shim we just pass start_unit=0.
        # This may overlap with layer9_alu's unit assignments; the original calls
        # them sequentially in the same FFN.
        _set_layer9_marker_suppress(ffn, S, _as_setdim_proxy(dim_positions), 0)

    return Operation(
        name="layer9_marker_suppress",
        phase=9,
        reads={"MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_STACK0",
               "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
               "OP_OR", "OP_XOR", "OP_AND"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="ffn",
        bake_fn=bake,
    )


