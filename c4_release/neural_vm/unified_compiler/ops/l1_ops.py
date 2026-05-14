"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ..layer_compiler import Operation
from ..primitives import AO, AP, DeclarativeAttentionHeadSpec
from .shared import _as_setdim_proxy


def make_layer1_ffn_op() -> Operation:
    """L1 FFN: STACK0_BYTE0 flag + BYTE_INDEX flags from threshold differences.

    Originally: `_set_layer1_ffn` at vm_step.py:2922.

    Reads L1H0/L1H1/L1H2/L1H4/H0/H1 threshold outputs and IS_BYTE.
    Writes STACK0_BYTE0, BYTE_INDEX_0, BYTE_INDEX_1, BYTE_INDEX_2, BYTE_INDEX_3.
    """
    def bake(ffn, dim_positions, S):
        from ...setup_helpers import _set_layer1_ffn
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
        declarative_bake_fn=bake,
        migrated=True,
        claims=_claims,
        # ``_set_layer1_ffn`` writes 5 units (one per output: STACK0_BYTE0,
        # BYTE_INDEX_0..3). See setup_helpers.py:_set_layer1_ffn.
        ffn_units_used=5,
        postcondition={
            "STACK0_BYTE0": "0_or_1",
        },
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def _bake_layer1_ffn(ffn, S, BD):
    """Declarative L1 FFN spec: STACK0_BYTE0 and byte-index flags."""

    BP_I = 3
    NM = BD.NUM_MARKERS
    unit = 0

    ffn.W_up.data[unit, BD.L1H4 + BP_I] = S
    ffn.W_up.data[unit, BD.IS_BYTE] = S
    ffn.b_up.data[unit] = -S * 1.5
    ffn.W_gate.data[unit, BD.H1 + BP_I] = -1.0
    ffn.b_gate.data[unit] = 1.0
    ffn.W_down.data[BD.STACK0_BYTE0, unit] = 2.0 / S
    unit += 1

    for src_base, blocker_base, out_dim in (
        (BD.L1H1, BD.L1H0, BD.BYTE_INDEX_0),
        (BD.L1H2, BD.L1H1, BD.BYTE_INDEX_1),
        (BD.H0, BD.L1H2, BD.BYTE_INDEX_2),
        (BD.H1, BD.H0, BD.BYTE_INDEX_3),
    ):
        ffn.W_up.data[unit, BD.IS_BYTE] = S
        for i in range(NM):
            ffn.W_up.data[unit, src_base + i] = S
            ffn.W_gate.data[unit, blocker_base + i] = -1.0
        ffn.b_up.data[unit] = -S * 1.5
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[out_dim, unit] = 2.0 / S
        unit += 1


def make_layer1_threshold_attn_op() -> Operation:
    """L1 attention: 3 fine threshold heads + STEP_END + L1H4."""
    def bake(attn, dim_positions, S):
        from ..primitives import Primitives
        proxy = _as_setdim_proxy(dim_positions)
        ALIBI_S = 10.0
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes.fill_(ALIBI_S)
            attn.alibi_slopes[3] = 0.0  # global SE detection
        HD = attn.W_q.shape[0] // attn.num_heads
        Primitives.generate_threshold_attention_heads(
            attn,
            [0.5, 1.5, 2.5],
            [proxy.L1H0, proxy.L1H1, proxy.L1H2],
            ALIBI_S,
            HD,
            heads=[0, 1, 2],
            bd=proxy,
        )
        # Head 3: STEP_END existence detection (global)
        Primitives.generate_attention_head(
            attn,
            DeclarativeAttentionHeadSpec(
                head_idx=3,
                q=(AP(0, proxy.CONST, 10.0),),
                k=(AP(0, proxy.MARK_SE_ONLY, 10.0),),
                v=(AP(1, proxy.MARK_SE_ONLY, 1.0),),
                o=(AO(proxy.HAS_SE, 1, 1.0),),
            ),
            HD,
        )
        # Head 4: threshold 6.5 for STACK0 byte 0 identification
        Primitives.generate_threshold_attention_heads(
            attn, [6.5], [proxy.L1H4], ALIBI_S, HD, heads=[4], bd=proxy,
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
        declarative_bake_fn=bake,
        migrated=True,
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )
