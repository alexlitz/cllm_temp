"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ..layer_compiler import Operation
from .shared import _as_setdim_proxy


def make_layer2_mem_byte_flags_op() -> Operation:
    """L2 FFN: MEM val byte position flags + extended BYTE_INDEX for STACK0."""
    def bake(ffn, dim_positions, S):
        from ...vm_step import _set_layer2_mem_byte_flags
        _set_layer2_mem_byte_flags(ffn, S, _as_setdim_proxy(dim_positions))

    return Operation(
        name="layer2_mem_byte_flags",
        phase=2,
        reads={"H0", "H1", "H4", "IS_BYTE", "BYTE_INDEX_0",
               "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3"},
        writes={"MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
                "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3"},
        kind="ffn",
        bake_fn=bake,
        migrated=True,
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
            attn, [5.5], [proxy.L2H0], ALIBI_S, HD, heads=[0]
        )

    return Operation(
        name="layer2_threshold_attn",
        phase=2,
        reads={"IS_MARK", "CONST"},
        writes={"L2H0"},
        kind="attn",
        bake_fn=bake,
        migrated=True,
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
    )


