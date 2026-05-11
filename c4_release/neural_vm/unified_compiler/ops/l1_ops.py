"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ..layer_compiler import Operation
from .shared import _as_setdim_proxy


def make_layer1_ffn_op() -> Operation:
    """L1 FFN: STACK0_BYTE0 flag + BYTE_INDEX flags from threshold differences.

    Originally: `_set_layer1_ffn` at vm_step.py:2922.

    Reads L1H0/L1H1/L1H2/L1H4/H0/H1 threshold outputs and IS_BYTE.
    Writes STACK0_BYTE0, BYTE_INDEX_0, BYTE_INDEX_1, BYTE_INDEX_2, BYTE_INDEX_3.
    """
    def bake(ffn, dim_positions, S):
        from ...vm_step import _set_layer1_ffn
        proxy = _as_setdim_proxy(dim_positions)
        _set_layer1_ffn(ffn, S, proxy)

    return Operation(
        name="layer1_ffn",
        phase=1,
        reads={"L1H0", "L1H1", "L1H2", "L1H4", "H0", "H1", "IS_BYTE"},
        writes={"STACK0_BYTE0", "BYTE_INDEX_0", "BYTE_INDEX_1",
                "BYTE_INDEX_2", "BYTE_INDEX_3"},
        kind="ffn",
        bake_fn=bake,
        migrated=True,
    )


def make_layer1_threshold_attn_op() -> Operation:
    """L1 attention: 3 fine threshold heads + STEP_END + L1H4."""
    def bake(attn, dim_positions, S):
        from ...vm_step import _set_threshold_attn
        proxy = _as_setdim_proxy(dim_positions)
        ALIBI_S = 10.0
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes.fill_(ALIBI_S)
            attn.alibi_slopes[3] = 0.0  # global SE detection
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_threshold_attn(
            attn,
            [0.5, 1.5, 2.5],
            [proxy.L1H0, proxy.L1H1, proxy.L1H2],
            ALIBI_S, HD, heads=[0, 1, 2],
        )
        # Head 3: STEP_END existence detection (global)
        base = 3 * HD
        attn.W_q[base, proxy.CONST] = 10.0
        attn.W_k[base, proxy.MARK_SE_ONLY] = 10.0
        attn.W_v[base + 1, proxy.MARK_SE_ONLY] = 1.0
        attn.W_o[proxy.HAS_SE, base + 1] = 1.0
        # Head 4: threshold 6.5 for STACK0 byte 0 identification
        _set_threshold_attn(
            attn, [6.5], [proxy.L1H4], ALIBI_S, HD, heads=[4]
        )

    return Operation(
        name="layer1_threshold_attn",
        phase=1,
        reads={"IS_MARK", "MARK_SE_ONLY", "CONST"},
        writes={"L1H0", "L1H1", "L1H2", "L1H4", "HAS_SE"},
        kind="attn",
        bake_fn=bake,
        migrated=True,
    )


