"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ..layer_compiler import Operation
from .shared import _as_setdim_proxy


def make_phase_a_ffn_op() -> Operation:
    """Step-structure FFN: detect marker transitions and emit NEXT_* flags.

    Originally: `_set_phase_a_ffn` at vm_step.py:2872. Lives at L0 in the
    hand-set layout.

    Reads H0/H1/H2/H3/H4 threshold-head outputs (per marker type).
    Writes NEXT_PC, NEXT_AX, NEXT_SP, NEXT_BP, NEXT_STACK0, NEXT_MEM, NEXT_SE.

    Dispatched as a block op pinned to layer_idx=0 so the bake hits the same
    transformer block (block[0].ffn) the legacy path used. This sidesteps the
    LayerCompiler's dep-based assignment, which would otherwise place this FFN
    at L1 (advancing past L0 because it reads H0-H4 written by L0 attn).
    """
    PC_I, AX_I, SP_I, BP_I, MEM_I, SE_I = 0, 1, 2, 3, 4, 5

    def bake(block, dim_positions, S):
        from ...vm_step import _set_phase_a_ffn
        proxy = _as_setdim_proxy(dim_positions)
        _set_phase_a_ffn(block.ffn, S, proxy)

    # Dim-ownership claims: ``_set_phase_a_ffn`` writes units 0..6 (one per
    # marker transition) into the L0 FFN. The W_up rows read H0..H4 marker
    # threshold outputs; W_down rows write NEXT_* dims. The 7 units are
    # stable across builds (always start at unit 0, one per transition).
    _claims = set()
    # Each transition row writes (unit, NEXT_* slot). Use NEXT_* dim names
    # as the column tag since W_down[NEXT_*, unit] = 2.0/S.
    # See _set_phase_a_ffn in vm_step.py for the transition list.
    _next_dim_names = [
        "NEXT_PC", "NEXT_AX", "NEXT_SP", "NEXT_BP",
        "NEXT_STACK0", "NEXT_MEM", "NEXT_SE",
    ]
    for u, next_dim in enumerate(_next_dim_names):
        _claims.add((0, "ffn_W_down", str(u), f"{next_dim}+0"))

    # The threshold heads write 7 dims each (one per marker type), so we
    # express reads as the head-base names; the FFN reads any element in the
    # H0..H4 ranges, which are size-7 dims.
    return Operation(
        name="phase_a_ffn",
        phase=0,
        reads={"H0", "H1", "H2", "H3", "H4"},
        writes={"NEXT_PC", "NEXT_AX", "NEXT_SP", "NEXT_BP",
                "NEXT_STACK0", "NEXT_MEM", "NEXT_SE"},
        kind="block",
        layer_idx=0,
        bake_fn=bake,
        migrated=True,
        claims=_claims,
        # ``_set_phase_a_ffn`` writes one FFN hidden unit per transition in
        # the 7-entry ``transitions`` list (SE→PC, PC→AX, AX→SP, SP→BP,
        # BP→STACK0, STACK0→MEM, MEM→SE). Units 0..6.
        ffn_units_used=7,
        # Staleness invariants: NEXT_* flags fire at the token that
        # immediately precedes the corresponding marker (position N
        # predicts position N+1). The LM head consumes these to emit
        # the next marker token.
        produces={
            "NEXT_AX": "PC_byte3",
            "NEXT_SP": "AX_byte3",
            "NEXT_BP": "SP_byte3",
            "NEXT_STACK0": "BP_byte3",
            "NEXT_MEM": "STACK0_byte3",
            "NEXT_SE": "MEM_VAL_byte3",
            "NEXT_PC": "STEP_END",
        },
        # Tier C: fires every program; marker transitions are opcode-independent
        # so the MoE partition correctly keeps these units in the shared expert.
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
        compaction_safe=True,
    )


def make_layer0_threshold_attn_op() -> Operation:
    """L0 attention: 8 threshold heads detecting marker distance.

    Dispatched as a block op pinned to layer_idx=0 so the bake hits the same
    transformer block (block[0].attn) the legacy path used. Using kind="block"
    keeps the L0 op aligned with the hand-set block index regardless of
    LayerCompiler dep-based assignment.
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_threshold_attn
        attn = block.attn
        proxy = _as_setdim_proxy(dim_positions)
        ALIBI_S = 10.0
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes.fill_(ALIBI_S)
        HD = attn.W_q.shape[0] // attn.num_heads
        # Pass proxy as BD= so pin_io_only=True resolves CONST/IS_MARK/MARKS
        # via dim_positions. In practice these are all IO-pinned so the legacy
        # fallback agrees, but routing through the proxy keeps the bake honest
        # if the IO-pin contract ever changes.
        _set_threshold_attn(
            attn,
            [3.5, 4.5, 7.5, 8.5, 9.5, 14.5, 19.5, 24.5],
            [proxy.H0, proxy.H1, proxy.H2, proxy.H3, proxy.H4,
             proxy.H5, proxy.H6, proxy.H7],
            ALIBI_S, HD,
            BD=proxy,
        )
        # Softmax-sharpness fix (head 1): the H1 (d<=4.5) threshold head
        # passes its scoring budget on the synthetic K target (Q*K/sqrt(HD)
        # = slope*threshold = 45, then ALiBi at d=4 deducts 40 -> s_target=5)
        # but the softmax mass tops out at ~98.7% because s_target=5 only
        # barely clears the softmax1 anchor (need s_target >~ ln(99)+margin).
        # Audit doc 87442ad recommended "bump K-scale ~2.0x (raise s_target)";
        # the original implementation doubled W_k[head1_base, IS_MARK] only.
        #
        # Bug fix 2026-05-12 (fix-phase2-3-psh-stack0):
        # Doubling W_k alone doubled Q*K/sqrt(HD) from 45 to 90 *without*
        # changing the ALiBi slope, which shifted the H1 score
        # zero-crossing from d=4.5 to d=9. Concrete consequences:
        #
        #   - H1[BP_I] (and H1[PC_I], H1[AX_I], etc.) stayed near 1.0
        #     across d=5..8 from any IS_MARK token instead of falling to 0.
        #   - phase_a_ffn's NEXT_STACK0 unit fires on "H1[BP_I] AND NOT
        #     H0[BP_I]"; with the wider H1 this was true at d=4..8 (5
        #     positions in a row), so the LM head emitted 5 STACK0
        #     marker tokens on PSH steps instead of marker + 4 value
        #     bytes. L1 FFN's STACK0_BYTE0 flag attached at the wrong
        #     positions and L7 head 0 (operand gather) read CLEAN_EMBED
        #     from the wrong column, so ADD's operand A came back as 0
        #     (test_add_basic 10+32 -> 32 instead of 42).
        #
        # The fix: pair the K-scale bump with a matching slope bump on
        # head 1 so BOTH Q*K and slope*d sides scale together — keeping
        # the zero-crossing at d=4.5 while sharpening softmax mass at
        # d=4 from 98.7% to ~99.995%. Many downstream consumers
        # (L3 carry-forward, L4 PC relay, L7 operand gather, L8/L10 byte
        # passthrough heads, L16 LEV routing FFN, etc.) read H1[*_I] as
        # a position-band selector with the contract "fires only when
        # nearest IS_MARK is within d=4". Restoring that contract here
        # avoids fanning the fix out to every consumer.
        head1_base = 1 * HD
        attn.W_k[head1_base, proxy.IS_MARK] *= 2.0
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes[1] = ALIBI_S * 2.0

    # Dim-ownership claims: ``_set_threshold_attn`` writes per head h:
    #   W_q[h*HD, CONST]    : the head's bias (slot 0)
    #   W_k[h*HD, IS_MARK]  : K-side mark detector (slot 0)
    #   W_v[h*HD + 1 + m, MARKS[m]]  : V slot 1..7 (one per marker)
    #   W_o[out_base + m, h*HD + 1 + m] : O column 1..7
    # For 8 heads (h=0..7) writing to H0..H7 output bases.
    # MARKS = [MARK_PC, MARK_AX, MARK_SP, MARK_BP, MARK_MEM, MARK_SE, MARK_CS]
    _claims = set()
    _MARKS = ["MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP",
              "MARK_MEM", "MARK_SE", "MARK_CS"]
    _OUT_BASES = ["H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7"]
    for h in range(8):
        # V slots 1..7: each reads MARKS[m] column.
        for m, mark in enumerate(_MARKS):
            _claims.add((0, "attn_W_v", f"{h}_{1 + m}", f"{mark}+0"))
        # Q slot 0: CONST column.
        _claims.add((0, "attn_W_q", f"{h}_0", "CONST+0"))
        # K slot 0: IS_MARK column.
        _claims.add((0, "attn_W_k", f"{h}_0", "IS_MARK+0"))
        # O writes 7 columns of W_o[out_base+m, h*HD + 1 + m]; in the
        # attn_W_o scope the identifier is column-of-W_o = "<head>_<slot>".
        for m, out_base in enumerate([_OUT_BASES[h]] * 7):
            _claims.add((0, "attn_W_o", f"{h}_{1 + m}", f"{out_base}+{m}"))

    return Operation(
        name="layer0_threshold_attn",
        phase=0,
        reads={"IS_MARK", "CONST"},
        writes={"H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7"},
        kind="block",
        layer_idx=0,
        bake_fn=bake,
        migrated=True,
        claims=_claims,
    )


