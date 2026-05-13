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

    # Dim-ownership claims: L1 FFN writes 5 units at fixed positions:
    #   unit 0: STACK0_BYTE0
    #   unit 1: BYTE_INDEX_0
    #   unit 2: BYTE_INDEX_1
    #   unit 3: BYTE_INDEX_2
    #   unit 4: BYTE_INDEX_3
    _claims = set()
    _outputs = [
        "STACK0_BYTE0", "BYTE_INDEX_0", "BYTE_INDEX_1",
        "BYTE_INDEX_2", "BYTE_INDEX_3",
    ]
    for u, out_dim in enumerate(_outputs):
        _claims.add((1, "ffn_W_down", str(u), f"{out_dim}+0"))

    return Operation(
        name="layer1_ffn",
        phase=1,
        reads={"L1H0", "L1H1", "L1H2", "L1H4", "H0", "H1", "IS_BYTE"},
        writes={"STACK0_BYTE0", "BYTE_INDEX_0", "BYTE_INDEX_1",
                "BYTE_INDEX_2", "BYTE_INDEX_3"},
        kind="ffn",
        layer_idx=1,
        bake_fn=bake,
        migrated=True,
        claims=_claims,
        # ``_set_layer1_ffn`` writes 5 units (one per output: STACK0_BYTE0,
        # BYTE_INDEX_0..3). See setup_helpers.py:_set_layer1_ffn.
        ffn_units_used=5,
        # Staleness invariants + Tier A produces. STACK0_BYTE0 fires at
        # the STACK0 byte 0 token (d=6 from BP marker, identified by
        # L1H4[BP] AND NOT H1[BP] AND IS_BYTE). L3 head 4 (stack0
        # carry-forward) and L7 head 0 (operand gather) both target this
        # flag as their K-side. BYTE_INDEX_* are emitted at the matching
        # STACK0_byteN positions.
        produces={
            "STACK0_BYTE0": "STACK0_byte0",
            "BYTE_INDEX_0": "STACK0_byte0",
            "BYTE_INDEX_1": "STACK0_byte1",
            "BYTE_INDEX_2": "STACK0_byte2",
            "BYTE_INDEX_3": "STACK0_byte3",
        },
        # ``opcodes``: L1 emits STACK0_BYTE0 every step from token-position
        # features (threshold attention over L1H4); it is NOT opcode-gated,
        # so the set is empty. The opcode-coverage detector treats L0-L3
        # FFN ops with empty opcodes as legitimate emission ops.
        opcodes=set(),
        # ``reset_after_step``: STACK0_BYTE0 is re-emitted by this op on
        # every step from raw token features; it should NOT carry stale
        # values across the step boundary.
        reset_after_step={
            "STACK0_BYTE0",
            "BYTE_INDEX_0", "BYTE_INDEX_1",
            "BYTE_INDEX_2", "BYTE_INDEX_3",
        },
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
            BD=proxy,
        )
        # Head 3: STEP_END existence detection (global)
        base = 3 * HD
        attn.W_q[base, proxy.CONST] = 10.0
        attn.W_k[base, proxy.MARK_SE_ONLY] = 10.0
        attn.W_v[base + 1, proxy.MARK_SE_ONLY] = 1.0
        attn.W_o[proxy.HAS_SE, base + 1] = 1.0
        # Head 4: threshold 6.5 for STACK0 byte 0 identification
        _set_threshold_attn(
            attn, [6.5], [proxy.L1H4], ALIBI_S, HD, heads=[4], BD=proxy,
        )

    # Dim-ownership claims: 5 heads on L1 attn.
    #   Heads 0,1,2,4: threshold heads writing to L1H0/L1H1/L1H2/L1H4
    #                  (each writes V slots 1..7 across MARKS).
    #   Head 3: STEP_END detector — Q[CONST], K[MARK_SE_ONLY],
    #           V[3*HD+1, MARK_SE_ONLY], O[HAS_SE, 3*HD+1].
    _claims = set()
    _MARKS = ["MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP",
              "MARK_MEM", "MARK_SE", "MARK_CS"]
    # Threshold heads 0, 1, 2, 4 → L1H0, L1H1, L1H2, L1H4.
    for h, out_base in [(0, "L1H0"), (1, "L1H1"), (2, "L1H2"), (4, "L1H4")]:
        for m, mark in enumerate(_MARKS):
            _claims.add((1, "attn_W_v", f"{h}_{1 + m}", f"{mark}+0"))
            _claims.add((1, "attn_W_o", f"{h}_{1 + m}", f"{out_base}+{m}"))
        _claims.add((1, "attn_W_q", f"{h}_0", "CONST+0"))
        _claims.add((1, "attn_W_k", f"{h}_0", "IS_MARK+0"))
    # Head 3: STEP_END global detector.
    _claims.add((1, "attn_W_q", "3_0", "CONST+0"))
    _claims.add((1, "attn_W_k", "3_0", "MARK_SE_ONLY+0"))
    _claims.add((1, "attn_W_v", "3_1", "MARK_SE_ONLY+0"))
    _claims.add((1, "attn_W_o", "3_1", "HAS_SE+0"))

    return Operation(
        name="layer1_threshold_attn",
        phase=1,
        reads={"IS_MARK", "MARK_SE_ONLY", "CONST"},
        writes={"L1H0", "L1H1", "L1H2", "L1H4", "HAS_SE"},
        kind="attn",
        layer_idx=1,
        bake_fn=bake,
        migrated=True,
        claims=_claims,
    )


