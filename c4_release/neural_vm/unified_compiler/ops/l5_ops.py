"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ..layer_compiler import Operation
from ..ir import CompilerIR, FFNRule
from ..primitives import AO, AP, DeclarativeAttentionHeadSpec, Primitives
from .shared import _as_setdim_proxy, _opcode_name_map


def make_layer5_fetch_op() -> Operation:
    """L5 attention: instruction-fetch heads (8 heads).

    Dispatched as a block op pinned to layer_idx=5 so the bake hits the same
    transformer block (block[5].attn) the legacy path used. Using kind="block"
    routes through compile_full_vm's block_ops dispatch even when legacy_bake
    is present, ensuring block[5] receives the L5 fetch logic for pure_neural
    execution. The companion `_layer5_fetch_dep_anchor` op declares the same
    reads/writes via kind="attn" so the LayerCompiler's dep graph still
    reserves a layer slot for it (preserving model n_layers).
    """
    def bake(block, dim_positions, S):
        attn = block.attn
        BD = _as_setdim_proxy(dim_positions)
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes.fill_(0.0)
            # L5 fetch heads read immutable code bytes by exact ADDR_KEY.
            # Recency bias makes later same-low-address aliases compete with
            # the exact code-prefix match once programs cross larger offsets.
        HD = attn.W_q.shape[0] // attn.num_heads
        Primitives.generate_attention_heads(
            attn,
            _layer5_fetch_head_specs(BD),
            HD,
        )

    # Dim-ownership claims (see c4_release/docs/DIM_OWNERSHIP_REGISTRY.md).
    # `_set_layer5_fetch` writes attn5.W_v rows {base+32+k, base+48+k} for
    # k in 0..15 across heads 0..5 (V slots 32..47 and 48..63 per head):
    #
    #   W_v[head*HD + 32 + k, CLEAN_EMBED_LO + k] = 1.0   (slot 32..47)
    #   W_v[head*HD + 48 + k, CLEAN_EMBED_HI + k] = 1.0   (slot 48..63)
    #
    # The 4-tuple claim's ``column`` carries the input-dim name + offset so
    # that a column-disjoint co-tenant on the same row (e.g.,
    # ``function_call_weights``' EMBED_HI+15 ENT relay on row 5_32) does
    # NOT register as a collision. Pre-column-granularity registry treated
    # this as ``(5, "attn_W_v", "5_32")`` and produced one false positive
    # that lived in ``KNOWN_BENIGN_COLLISIONS``; now retired.
    #
    # Heads 6 and 7 were deleted on 2026-05-11 (commit c1a5398) to break
    # the latent collision with `_set_function_call_weights`' head 6 ENT
    # relay (V slots 1..16) — the registry would have caught that as
    # (5, "attn_W_v", "6_<k>", "EMBED_LO+<k>") for k in 1..16.
    _claims = set()
    for head in range(6):  # heads 0..5
        for k in range(16):  # k = 0..15
            slot_lo = 32 + k
            slot_hi = 48 + k
            _claims.add(
                (5, "attn_W_v", f"{head}_{slot_lo}", f"CLEAN_EMBED_LO+{k}")
            )
            _claims.add(
                (5, "attn_W_v", f"{head}_{slot_hi}", f"CLEAN_EMBED_HI+{k}")
            )

    return Operation(
        name="layer5_fetch",
        phase=5,
        # Reads: PC/AX markers + FETCH addr (PC+K) + ADDR_KEY (per CODE byte) +
        #        CLEAN_EMBED (the value at the matched CODE byte).
        # Note: heads 6/7 also read OP_* via V projection but that's the DEPRECATED
        # path (OP_* flags were removed from embeddings 2026-04-13). Excluding from
        # reads since they're not semantically active inputs.
        reads={"MARK_PC", "MARK_AX", "HAS_SE",
               "FETCH_LO", "FETCH_HI", "EMBED_LO", "EMBED_HI",
               "ADDR_KEY", "CONST", "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"OPCODE_BYTE_LO", "OPCODE_BYTE_HI",
                "FETCH_LO", "FETCH_HI",
                "OP_IMM", "OP_LEA", "OP_EXIT", "OP_JMP", "OP_JSR",
                "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
                "OP_OR", "OP_XOR", "OP_AND",
                "OP_EQ", "OP_LT", "OP_SHL", "OP_SHR"},
        kind="block",
        layer_idx=5,
        bake_fn=bake,
        declarative_bake_fn=bake,
        compiler_ir_factory=_layer5_fetch_ir,
        migrated=True,
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#how-bytecode-is-passed-to-the-network",
    )


def _layer5_fetch_ir(dim_positions, HD) -> CompilerIR:
    BD = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.extend(_layer5_fetch_head_specs(BD))
    return ir


def _band_projection_writes(slot_base: int, dim_base: int, weight: float = 1.0):
    return tuple(AP(slot_base + k, dim_base + k, weight) for k in range(16))


def _band_output_writes(dim_base: int, slot_base: int, weight: float = 1.0):
    return tuple(AO(dim_base + k, slot_base + k, weight) for k in range(16))


def _addr_key_match_writes(BD, weight: float, top_slot_base: int = 35):
    return (
        tuple(AP(k, BD.ADDR_KEY + k, weight) for k in range(16))
        + tuple(AP(16 + k, BD.ADDR_KEY + 16 + k, weight) for k in range(16))
        + tuple(AP(top_slot_base + k, BD.ADDR_KEY + 32 + k, weight) for k in range(16))
    )


def _code_fetch_v_writes(BD, weight: float = 1.0):
    return (
        _band_projection_writes(32, BD.CLEAN_EMBED_LO, weight)
        + _band_projection_writes(48, BD.CLEAN_EMBED_HI, weight)
    )


def _layer5_fetch_head_specs(BD) -> tuple[DeclarativeAttentionHeadSpec, ...]:
    """Declarative replacement for ``setup_helpers._set_layer5_fetch``."""

    from ...constants import PC_OFFSET

    L = 20.0
    ADDR_L = 20.0

    def ax_gate(slot: int = 33):
        return (
            (AP(slot, BD.MARK_AX, 500.0), AP(slot, BD.CONST, -500.0)),
            (AP(slot, BD.CONST, 5.0),),
        )

    def pc_gate(slot: int = 33):
        return (
            (AP(slot, BD.MARK_PC, 500.0), AP(slot, BD.CONST, -500.0)),
            (AP(slot, BD.CONST, 5.0),),
        )

    ax_q_gate, ax_k_gate = ax_gate()
    pc_q_gate, pc_k_gate = pc_gate()
    pc_lo = PC_OFFSET & 0xF
    pc_hi = (PC_OFFSET >> 4) & 0xF
    pc_top = (PC_OFFSET >> 8) & 0xF
    TOP = 35

    def dynamic_top_q():
        return tuple(AP(TOP + k, BD.ADDR_KEY + 32 + k, ADDR_L) for k in range(16))

    def first_step_top0_q():
        return (
            AP(TOP, BD.CONST, ADDR_L),
            AP(TOP, BD.HAS_SE, -ADDR_L),
        )

    specs = [
        # Head 0: non-first-step immediate fetch at AX from TEMP=PC+1.
        DeclarativeAttentionHeadSpec(
            head_idx=0,
            q=(
                tuple(AP(k, BD.TEMP + k, ADDR_L) for k in range(16))
                + tuple(AP(16 + k, BD.TEMP + 16 + k, ADDR_L) for k in range(16))
                + (AP(32, BD.MARK_AX, L),)
                + dynamic_top_q()
                + ax_q_gate
                + (AP(34, BD.HAS_SE, 500.0), AP(34, BD.CONST, -500.0))
            ),
            k=_addr_key_match_writes(BD, ADDR_L) + ax_k_gate + (AP(34, BD.CONST, 5.0),),
            v=_code_fetch_v_writes(BD),
            o=(
                _band_output_writes(BD.FETCH_LO, 32)
                + _band_output_writes(BD.FETCH_HI, 48)
            ),
        ),
        # Head 1: non-first-step opcode fetch at AX from relayed PC.
        DeclarativeAttentionHeadSpec(
            head_idx=1,
            q=(
                tuple(AP(k, BD.EMBED_LO + k, ADDR_L) for k in range(16))
                + tuple(AP(16 + k, BD.EMBED_HI + k, ADDR_L) for k in range(16))
                + (AP(32, BD.MARK_AX, L),)
                + dynamic_top_q()
                + ax_q_gate
                + (AP(34, BD.HAS_SE, 500.0), AP(34, BD.CONST, -500.0))
            ),
            k=_addr_key_match_writes(BD, ADDR_L) + ax_k_gate + (AP(34, BD.CONST, 5.0),),
            v=_code_fetch_v_writes(BD),
            o=(
                _band_output_writes(BD.OPCODE_BYTE_LO, 32)
                + _band_output_writes(BD.OPCODE_BYTE_HI, 48)
            ),
        ),
        # Head 2: first-step opcode fetch at PC marker.
        DeclarativeAttentionHeadSpec(
            head_idx=2,
            q=(
                AP(pc_lo, BD.CONST, ADDR_L),
                AP(16 + pc_hi, BD.CONST, ADDR_L),
                AP(32, BD.MARK_PC, L),
                AP(TOP + pc_top, BD.CONST, ADDR_L),
                *pc_q_gate,
                AP(34, BD.HAS_SE, -500.0),
            ),
            k=_addr_key_match_writes(BD, ADDR_L) + pc_k_gate + (AP(34, BD.CONST, 5.0),),
            v=_code_fetch_v_writes(BD),
            o=(
                _band_output_writes(BD.OPCODE_BYTE_LO, 32)
                + _band_output_writes(BD.OPCODE_BYTE_HI, 48)
            ),
        ),
        # Head 3: dynamic immediate fetch at PC marker.
        DeclarativeAttentionHeadSpec(
            head_idx=3,
            q=(
                tuple(AP(k, BD.FETCH_LO + k, ADDR_L) for k in range(16))
                + tuple(AP(16 + k, BD.FETCH_HI + k, ADDR_L) for k in range(16))
                + (AP(32, BD.MARK_PC, L),)
                + dynamic_top_q()
                + first_step_top0_q()
                + pc_q_gate
            ),
            k=_addr_key_match_writes(BD, ADDR_L) + pc_k_gate,
            v=_code_fetch_v_writes(BD),
            o=(
                _band_output_writes(BD.FETCH_LO, 32, 40.0)
                + _band_output_writes(BD.FETCH_HI, 48, 40.0)
            ),
        ),
        # Head 4: first-step opcode fetch at AX marker.
        DeclarativeAttentionHeadSpec(
            head_idx=4,
            q=(
                AP(pc_lo, BD.CONST, ADDR_L),
                AP(16 + pc_hi, BD.CONST, ADDR_L),
                AP(32, BD.MARK_AX, L),
                AP(TOP + pc_top, BD.CONST, ADDR_L),
                *ax_q_gate,
                AP(34, BD.HAS_SE, -500.0),
            ),
            k=_addr_key_match_writes(BD, ADDR_L) + ax_k_gate + (AP(34, BD.CONST, 5.0),),
            v=_code_fetch_v_writes(BD),
            o=(
                _band_output_writes(BD.OPCODE_BYTE_LO, 32)
                + _band_output_writes(BD.OPCODE_BYTE_HI, 48)
            ),
        ),
        # Head 5: non-first-step opcode fetch at PC marker.
        DeclarativeAttentionHeadSpec(
            head_idx=5,
            q=(
                tuple(AP(k, BD.EMBED_LO + k, ADDR_L) for k in range(16))
                + tuple(AP(16 + k, BD.EMBED_HI + k, ADDR_L) for k in range(16))
                + (AP(32, BD.MARK_PC, L),)
                + dynamic_top_q()
                + pc_q_gate
                + (AP(34, BD.HAS_SE, 500.0), AP(34, BD.CONST, -500.0))
            ),
            k=_addr_key_match_writes(BD, ADDR_L) + pc_k_gate + (AP(34, BD.CONST, 5.0),),
            v=_code_fetch_v_writes(BD),
            o=(
                _band_output_writes(BD.OPCODE_BYTE_LO, 32)
                + _band_output_writes(BD.OPCODE_BYTE_HI, 48)
            ),
        ),
    ]

    return tuple(specs)


def make_layer5_fetch_dep_anchor_op() -> Operation:
    """No-op companion for layer5_fetch: declares identical reads/writes so
    the LayerCompiler's dep graph reserves a layer slot for it. The actual
    bake happens in `layer5_fetch` (kind="block", layer_idx=5); this op's
    bake is a no-op (its layout-assigned attention block is unrelated to
    block[5] and is overwritten by legacy_bake of the corresponding L6 attn).
    """
    def bake(attn, dim_positions, S):
        # No-op: actual bake is in `layer5_fetch` block op above.
        return

    return Operation(
        name="_layer5_fetch_dep_anchor",
        phase=5,
        reads={"MARK_PC", "MARK_AX", "HAS_SE",
               "FETCH_LO", "FETCH_HI", "EMBED_LO", "EMBED_HI",
               "ADDR_KEY", "CONST", "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"OPCODE_BYTE_LO", "OPCODE_BYTE_HI",
                "FETCH_LO", "FETCH_HI",
                "OP_IMM", "OP_LEA", "OP_EXIT", "OP_JMP", "OP_JSR",
                "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
                "OP_OR", "OP_XOR", "OP_AND",
                "OP_EQ", "OP_LT", "OP_SHL", "OP_SHR"},
        kind="attn",
        bake_fn=bake,
        migrated=True,
        declarative_authority="topology_anchor",
        smoke_tests=set(),
        spec_section=None,
    )


def make_opcode_decode_ffn_op() -> Operation:
    """L5 FFN: decode opcode byte → 34 one-hot OP_* flags at OPCODE_BASE.

    Dispatched as a block op pinned to layer_idx=5 so the bake hits the same
    transformer block (block[5].ffn) the legacy path used. Using kind="block"
    routes through compile_full_vm's block_ops dispatch even when legacy_bake
    is present, ensuring block[5] receives the opcode decode logic for
    pure_neural execution. The companion `_opcode_decode_ffn_dep_anchor` op
    declares the same reads/writes via kind="ffn" so the LayerCompiler's dep
    graph still reserves a layer slot for it (preserving model n_layers).
    """
    def bake(block, dim_positions, S):
        _bake_opcode_decode_ffn(
            block.ffn,
            S,
            _as_setdim_proxy(dim_positions),
        )

    # Dim-ownership claims (W_down output cells). The bake programs four
    # blocks of L5 FFN units (see ``_bake_opcode_decode_ffn``):
    #   units 0..33:   main rules — one unit per opcode at MARK_AX,
    #                  writing the matching OP_* dim. Unit 35 is the JSR
    #                  detector writing TEMP+0.
    #   units 34..51:  first-step PC-marker decode (18 rules).
    #                  Unit 34: JMP, 35: JSR (TEMP+0), 36: IMM, 37: LEA,
    #                  38: EXIT, 39: NOP, 40..44: ADD..MOD,
    #                  45/46: OR/XOR, 47: AND, 48: EQ, 49: LT,
    #                  50: SHL, 51: SHR.
    #   unit 52:       reserved blank (preserves legacy unit numbering for
    #                  the first-step JSR-flag slot above).
    #   units 53..83:  TEMP[1..31] clear at MARK_PC — unit (52+k) writes
    #                  TEMP+k.
    #   units 84..88:  all-step PC-marker decode (BZ, BNZ, LEV, EXIT, JMP).
    _claims = set()
    # Main per-opcode units at MARK_AX, in the order from _opcode_decode_main_rules.
    _main_outputs = [
        "OP_LEA", "OP_IMM", "OP_JMP", "OP_JSR", "OP_BZ", "OP_BNZ",
        "OP_ENT", "OP_ADJ", "OP_LEV", "OP_LI", "OP_LC", "OP_SI",
        "OP_SC", "OP_PSH", "OP_OR", "OP_XOR", "OP_AND",
        "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
        "OP_SHL", "OP_SHR", "OP_ADD", "OP_SUB", "OP_MUL",
        "OP_DIV", "OP_MOD", "OP_EXIT", "OP_NOP",
        "OP_PUTCHAR", "OP_GETCHAR",
    ]
    for unit, op_dim in enumerate(_main_outputs):
        _claims.add((5, "ffn_W_down", str(unit), f"{op_dim}+0"))
    # First-step PC-marker decode (18 rules at units 34..51).
    _first_step_outputs = [
        ("34", "OP_JMP+0"),
        ("35", "TEMP+0"),  # JSR's IS_JSR flag
        ("36", "OP_IMM+0"),
        ("37", "OP_LEA+0"),
        ("38", "OP_EXIT+0"),
        ("39", "OP_NOP+0"),
        ("40", "OP_ADD+0"),
        ("41", "OP_SUB+0"),
        ("42", "OP_MUL+0"),
        ("43", "OP_DIV+0"),
        ("44", "OP_MOD+0"),
        ("45", "OP_OR+0"),
        ("46", "OP_XOR+0"),
        ("47", "OP_AND+0"),
        ("48", "OP_EQ+0"),
        ("49", "OP_LT+0"),
        ("50", "OP_SHL+0"),
        ("51", "OP_SHR+0"),
    ]
    for unit, col in _first_step_outputs:
        _claims.add((5, "ffn_W_down", unit, col))
    # Unit 52 is the reserved blank for first-step JSR's TEMP[0] (see
    # ``_bake_opcode_decode_ffn`` -- ``unit += 1`` before TEMP-clear).
    # TEMP[1..31] clear units 53..83.
    for k in range(1, 32):
        _claims.add((5, "ffn_W_down", str(52 + k), f"TEMP+{k}"))
    # All-step PC-marker decode (BZ, BNZ, LEV, EXIT, JMP at units 84..88).
    for unit, col in (
        ("84", "OP_BZ+0"),
        ("85", "OP_BNZ+0"),
        ("86", "OP_LEV+0"),
        ("87", "OP_EXIT+0"),
        ("88", "OP_JMP+0"),
    ):
        _claims.add((5, "ffn_W_down", unit, col))

    return Operation(
        name="opcode_decode_ffn",
        phase=5,
        reads={"OPCODE_BYTE_LO", "OPCODE_BYTE_HI", "MARK_AX", "MARK_PC", "HAS_SE"},
        writes={"OP_LEA", "OP_IMM", "OP_JMP", "OP_JSR", "OP_BZ", "OP_BNZ",
                "OP_ENT", "OP_ADJ", "OP_LEV", "OP_LI", "OP_LC", "OP_SI",
                "OP_SC", "OP_PSH", "OP_OR", "OP_XOR", "OP_AND",
                "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
                "OP_SHL", "OP_SHR", "OP_ADD", "OP_SUB", "OP_MUL",
                "OP_DIV", "OP_MOD", "OP_EXIT", "OP_NOP",
                "OP_PUTCHAR", "OP_GETCHAR",
                "TEMP"},  # JSR writes IS_JSR to TEMP[0]
        kind="block",
        layer_idx=5,
        bake_fn=bake,
        declarative_bake_fn=bake,
        migrated=True,
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#how-bytecode-is-passed-to-the-network",
    )


def _opcode_decode_main_rules(S):
    """CompilerIR rules for opcode byte decode at the AX marker."""

    from ...embedding import Opcode

    opcodes = [
        (Opcode.LEA, 0, 0),
        (Opcode.IMM, 1, 0),
        (Opcode.JMP, 2, 0),
        (Opcode.JSR, 3, 0),
        (Opcode.BZ, 4, 0),
        (Opcode.BNZ, 5, 0),
        (Opcode.ENT, 6, 0),
        (Opcode.ADJ, 7, 0),
        (Opcode.LEV, 8, 0),
        (Opcode.LI, 9, 0),
        (Opcode.LC, 10, 0),
        (Opcode.SI, 11, 0),
        (Opcode.SC, 12, 0),
        (Opcode.PSH, 13, 0),
        (Opcode.OR, 14, 0),
        (Opcode.XOR, 15, 0),
        (Opcode.AND, 0, 1),
        (Opcode.EQ, 1, 1),
        (Opcode.NE, 2, 1),
        (Opcode.LT, 3, 1),
        (Opcode.GT, 4, 1),
        (Opcode.LE, 5, 1),
        (Opcode.GE, 6, 1),
        (Opcode.SHL, 7, 1),
        (Opcode.SHR, 8, 1),
        (Opcode.ADD, 9, 1),
        (Opcode.SUB, 10, 1),
        (Opcode.MUL, 11, 1),
        (Opcode.DIV, 12, 1),
        (Opcode.MOD, 13, 1),
        (Opcode.EXIT, 6, 2),
        (Opcode.NOP, 7, 2),
        (Opcode.PUTCHAR, 1, 4),
        (Opcode.GETCHAR, 0, 4),
    ]
    op_names = _opcode_name_map()
    return tuple(
        FFNRule.gated_write(
            name=f"l5_decode_{op_names[op_val].lower()}_at_ax",
            conditions=(
                (f"OPCODE_BYTE_LO+{lo}", 1.0),
                (f"OPCODE_BYTE_HI+{hi}", 1.0),
            ),
            threshold=1.5,
            gate="MARK_AX",
            writes=((op_names[op_val], 10.0 / S),),
        )
        for op_val, lo, hi in opcodes
    )


def _opcode_decode_first_step_rules(S):
    """CompilerIR rules for first-step PC-marker opcode decode."""

    first_step_opcodes = [
        (2, 0, "OP_JMP"),
        (3, 0, "TEMP+0"),
        (1, 0, "OP_IMM"),
        (0, 0, "OP_LEA"),
        (6, 2, "OP_EXIT"),
        (7, 2, "OP_NOP"),
        (9, 1, "OP_ADD"),
        (10, 1, "OP_SUB"),
        (11, 1, "OP_MUL"),
        (12, 1, "OP_DIV"),
        (13, 1, "OP_MOD"),
        (14, 0, "OP_OR"),
        (15, 0, "OP_XOR"),
        (0, 1, "OP_AND"),
        (1, 1, "OP_EQ"),
        (3, 1, "OP_LT"),
        (7, 1, "OP_SHL"),
        (8, 1, "OP_SHR"),
    ]
    return tuple(
        FFNRule.constant_write(
            name=f"l5_first_step_decode_{out_dim.lower().replace('+', '_')}",
            conditions=(
                (f"OPCODE_BYTE_LO+{lo}", 1.0),
                (f"OPCODE_BYTE_HI+{hi}", 1.0),
                ("MARK_PC", 1.0),
                ("HAS_SE", -1.0),
            ),
            threshold=2.5,
            writes=((out_dim, 10.0 / S),),
        )
        for lo, hi, out_dim in first_step_opcodes
    )


def _opcode_decode_temp_clear_rules(S):
    """CompilerIR rules for TEMP[1..31] clearing at the PC marker.

    TEMP[0] is reserved for the first-step JSR flag, so the caller must leave
    one blank hidden unit before lowering these rules to preserve legacy unit
    numbering.
    """

    return tuple(
        FFNRule.gated_write(
            name=f"l5_temp_clear_{k}_at_pc",
            conditions=(("MARK_PC", 1.0),),
            threshold=0.5,
            gate=f"TEMP+{k}",
            gate_weight=-1.0,
            writes=((f"TEMP+{k}", 2.0 / S),),
        )
        for k in range(1, 32)
    )


def _opcode_decode_all_step_pc_rules(S):
    """CompilerIR rules for all-step PC-marker opcode decode."""

    from ...embedding import Opcode

    all_step_opcodes = [
        (Opcode.BZ, 4, 0),
        (Opcode.BNZ, 5, 0),
        (Opcode.LEV, 8, 0),
        (Opcode.EXIT, 6, 2),
        (Opcode.JMP, 2, 0),
    ]
    op_names = _opcode_name_map()
    return tuple(
        FFNRule.constant_write(
            name=f"l5_all_step_decode_{op_names[op_val].lower()}_at_pc",
            conditions=(
                (f"OPCODE_BYTE_LO+{lo}", 1.0),
                (f"OPCODE_BYTE_HI+{hi}", 1.0),
                ("MARK_PC", 1.0),
            ),
            threshold=2.5,
            writes=((op_names[op_val], 10.0 / S),),
        )
        for op_val, lo, hi in all_step_opcodes
    )


def _lower_l5_opcode_rules(ffn, rules, BD, *, unit: int, S: float) -> int:
    dim_positions = Primitives.dim_positions_from_bd(
        BD,
        Primitives.ffn_rule_dim_names(rules),
    )
    return Primitives.lower_ffn_rules(
        ffn,
        rules,
        dim_positions,
        start_unit=unit,
        S=S,
    )


def _bake_opcode_decode_ffn(ffn, S, BD):
    """Declarative L5 FFN spec: opcode-byte one-hot decode."""

    unit = 0
    unit = _lower_l5_opcode_rules(
        ffn,
        _opcode_decode_main_rules(S),
        BD,
        unit=unit,
        S=S,
    )
    unit = _lower_l5_opcode_rules(
        ffn,
        _opcode_decode_first_step_rules(S),
        BD,
        unit=unit,
        S=S,
    )

    unit += 1  # TEMP[0] is reserved for first-step JSR; keep legacy blank unit.
    unit = _lower_l5_opcode_rules(
        ffn,
        _opcode_decode_temp_clear_rules(S),
        BD,
        unit=unit,
        S=S,
    )
    unit = _lower_l5_opcode_rules(
        ffn,
        _opcode_decode_all_step_pc_rules(S),
        BD,
        unit=unit,
        S=S,
    )


def make_opcode_decode_ffn_dep_anchor_op() -> Operation:
    """No-op companion for opcode_decode_ffn: declares identical reads/writes
    so the LayerCompiler's dep graph reserves a layer slot for it. The actual
    bake happens in `opcode_decode_ffn` (kind="block", layer_idx=5); this op's
    bake is a no-op.
    """
    def bake(ffn, dim_positions, S):
        # No-op: actual bake is in `opcode_decode_ffn` block op above.
        return

    return Operation(
        name="_opcode_decode_ffn_dep_anchor",
        phase=5,
        reads={"OPCODE_BYTE_LO", "OPCODE_BYTE_HI", "MARK_AX", "MARK_PC", "HAS_SE"},
        writes={"OP_LEA", "OP_IMM", "OP_JMP", "OP_JSR", "OP_BZ", "OP_BNZ",
                "OP_ENT", "OP_ADJ", "OP_LEV", "OP_LI", "OP_LC", "OP_SI",
                "OP_SC", "OP_PSH", "OP_OR", "OP_XOR", "OP_AND",
                "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
                "OP_SHL", "OP_SHR", "OP_ADD", "OP_SUB", "OP_MUL",
                "OP_DIV", "OP_MOD", "OP_EXIT", "OP_NOP",
                "OP_PUTCHAR", "OP_GETCHAR",
                "TEMP"},
        kind="ffn",
        bake_fn=bake,
        migrated=True,
        declarative_authority="topology_anchor",
        smoke_tests=set(),
        spec_section=None,
    )
