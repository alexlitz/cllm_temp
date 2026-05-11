"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ..layer_compiler import Operation
from .shared import _as_setdim_proxy


def make_layer4_pc_relay_op() -> Operation:
    """L4 attention: relay PC marker EMBED → AX marker EMBED.

    Pinned to ``layer_idx=4`` via ``kind="block"`` because the legacy
    ``set_vm_weights`` pipeline targets block 4. Without pinning, the
    dep-graph layer assignment places this op at a later block (e.g. L5/L6),
    leaving block 4's attn zero-init and breaking the L5 fetch chain
    (the regression at commit b2d9f4c3).
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer4_pc_relay
        attn = block.attn
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes.fill_(0.5)
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer4_pc_relay(attn, S, _as_setdim_proxy(dim_positions), HD)

    return Operation(
        name="layer4_pc_relay",
        phase=4,
        reads={"MARK_PC", "MARK_AX", "EMBED_LO", "EMBED_HI", "CONST"},
        writes={"EMBED_LO", "EMBED_HI"},  # at AX marker
        kind="block",
        bake_fn=bake,
        layer_idx=4,
        migrated=True,
    )


def make_layer4_ffn_op() -> Operation:
    """L4 FFN: compute PC+1/2/3/4 in FETCH dims for L5 fetch.

    Pinned to ``layer_idx=4`` via ``kind="block"``; see
    ``make_layer4_pc_relay_op``.
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer4_ffn
        _set_layer4_ffn(block.ffn, S, _as_setdim_proxy(dim_positions))

    return Operation(
        name="layer4_ffn",
        phase=4,
        reads={"MARK_AX", "MARK_PC", "EMBED_LO", "EMBED_HI",
               "IS_BYTE", "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
               "H1"},
        writes={"FETCH_LO", "FETCH_HI"},
        kind="block",
        bake_fn=bake,
        layer_idx=4,
        migrated=True,
    )


