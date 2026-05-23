"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ..ir import CompilerIR, FFNRule
from ..layer_compiler import Operation
from ..primitives import Primitives
from .shared import _as_setdim_proxy


def _phase_a_ffn_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for the L0 marker-transition detector."""

    PC_I, AX_I, SP_I, BP_I, MEM_I, SE_I = 0, 1, 2, 3, 4, 5
    write_scale = 2.0 / S
    transitions = (
        (f"H0+{SE_I}", None, "NEXT_PC"),
        (f"H1+{PC_I}", f"H0+{PC_I}", "NEXT_AX"),
        (f"H1+{AX_I}", f"H0+{AX_I}", "NEXT_SP"),
        (f"H1+{SP_I}", f"H0+{SP_I}", "NEXT_BP"),
        (f"H1+{BP_I}", f"H0+{BP_I}", "NEXT_STACK0"),
        (f"H4+{BP_I}", f"H3+{BP_I}", "NEXT_MEM"),
        (f"H3+{MEM_I}", f"H2+{MEM_I}", "NEXT_SE"),
    )
    rules = []
    for idx, (up_dim, gate_dim, out_dim) in enumerate(transitions):
        if gate_dim is None:
            rules.append(FFNRule.constant_write(
                name=f"phase_a_{idx}_{out_dim.lower()}",
                conditions=((up_dim, 1.0),),
                threshold=0.3,
                writes=((out_dim, write_scale),),
            ))
        else:
            rules.append(FFNRule.gated_write(
                name=f"phase_a_{idx}_{out_dim.lower()}",
                conditions=((up_dim, 1.0),),
                threshold=0.3,
                gate_terms=((gate_dim, -1.0),),
                gate_bias=1.0,
                writes=((out_dim, write_scale),),
            ))
    return tuple(rules)


def _phase_a_ffn_ir(S: float = 100.0) -> CompilerIR:
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_phase_a_ffn_rules(S))
    return ir


def _bake_phase_a_ffn(ffn, S, BD) -> int:
    rules = _phase_a_ffn_rules(S)
    dim_positions = Primitives.dim_positions_from_bd(
        BD,
        Primitives.ffn_rule_dim_names(rules),
    )
    return Primitives.lower_ffn_rules(ffn, rules, dim_positions, S=S)


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
    def bake(block, dim_positions, S):
        proxy = _as_setdim_proxy(dim_positions)
        _bake_phase_a_ffn(block.ffn, S, proxy)

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
        declarative_bake_fn=bake,
        compiler_ir=_phase_a_ffn_ir(),
        declarative_authority="spec_generated",
        migrated=True,
        claims=_claims,
        # ``_set_phase_a_ffn`` writes one FFN hidden unit per transition in
        # the 7-entry ``transitions`` list (SE→PC, PC→AX, AX→SP, SP→BP,
        # BP→STACK0, STACK0→MEM, MEM→SE). Units 0..6.
        ffn_units_used=7,
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
        from ..primitives import Primitives
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
        Primitives.generate_threshold_attention_heads(
            attn,
            [3.5, 4.5, 7.5, 8.5, 9.5, 14.5, 19.5, 24.5],
            [proxy.H0, proxy.H1, proxy.H2, proxy.H3, proxy.H4,
             proxy.H5, proxy.H6, proxy.H7],
            ALIBI_S,
            HD,
            bd=proxy,
        )
        # H1's 4.5-token cutoff is semantically load-bearing: L1 marks
        # STACK0 byte 0 with L1H4[BP] AND NOT H1[BP]. Scaling this head makes
        # H1 fire at STACK0 byte 0 and blocks ADD/POP operand gather.
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
        declarative_bake_fn=bake,
        migrated=True,
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )
