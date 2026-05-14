"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ..layer_compiler import Operation
from ..primitives import AO, AP, DeclarativeAttentionHeadSpec, Primitives
from .shared import _as_setdim_proxy


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
        from ...setup_helpers import _set_layer5_fetch
        attn = block.attn
        BD = _as_setdim_proxy(dim_positions)
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes.fill_(0.1)
            # Softmax-sharpness fix (head 5): s_target=125 is already strong
            # but ALiBi slope=0.1 over the synthetic Q->K distance of 4
            # contributes only 0.4 nats of positional separation. The
            # runner-up dim sits at the Q position (distance 0), so the
            # 0.4-nat gap is not enough to drown it (mass@target = 0.0).
            # Audit doc 87442ad recommends "raise ALiBi slope 0.1 -> ~1.0"
            # — implemented here. The other L5 heads keep slope=0.1 because
            # they attend to nearby positions (fetch operand bytes within
            # the current step); head 5 wants steeper recency to suppress
            # cross-step distractors.
            attn.alibi_slopes[5] = 1.0
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer5_fetch(attn, S, BD, HD)

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
        migrated=True,
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#how-bytecode-is-passed-to-the-network",
    )


def _band_projection_writes(slot_base: int, dim_base: int, weight: float = 1.0):
    return tuple(AP(slot_base + k, dim_base + k, weight) for k in range(16))


def _band_output_writes(dim_base: int, slot_base: int, weight: float = 1.0):
    return tuple(AO(dim_base + k, slot_base + k, weight) for k in range(16))


def _addr_key_match_writes(BD, weight: float):
    return (
        tuple(AP(k, BD.ADDR_KEY + k, weight) for k in range(16))
        + tuple(AP(16 + k, BD.ADDR_KEY + 16 + k, weight) for k in range(16))
        + (AP(32, BD.ADDR_KEY + 32, weight),)
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

    specs = [
        # Head 0: non-first-step immediate fetch at AX from TEMP=PC+1.
        DeclarativeAttentionHeadSpec(
            head_idx=0,
            q=(
                tuple(AP(k, BD.TEMP + k, L) for k in range(16))
                + tuple(AP(16 + k, BD.TEMP + 16 + k, L) for k in range(16))
                + (AP(32, BD.MARK_AX, L),)
                + ax_q_gate
                + (AP(34, BD.HAS_SE, 500.0), AP(34, BD.CONST, -500.0))
            ),
            k=_addr_key_match_writes(BD, L) + ax_k_gate + (AP(34, BD.CONST, 5.0),),
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
                tuple(AP(k, BD.EMBED_LO + k, L) for k in range(16))
                + tuple(AP(16 + k, BD.EMBED_HI + k, L) for k in range(16))
                + (AP(32, BD.MARK_AX, L),)
                + ax_q_gate
                + (AP(34, BD.HAS_SE, 500.0), AP(34, BD.CONST, -500.0))
            ),
            k=_addr_key_match_writes(BD, L) + ax_k_gate + (AP(34, BD.CONST, 5.0),),
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
                AP(0, BD.MARK_PC, L),
                AP(0, BD.HAS_SE, -L),
                AP(pc_lo, BD.CONST, L),
                AP(16 + pc_hi, BD.CONST, L),
                AP(32, BD.MARK_PC, L),
                *pc_q_gate,
                AP(34, BD.HAS_SE, -500.0),
            ),
            k=_addr_key_match_writes(BD, L) + pc_k_gate + (AP(34, BD.CONST, 5.0),),
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
                tuple(AP(k, BD.FETCH_LO + k, L) for k in range(16))
                + tuple(AP(16 + k, BD.FETCH_HI + k, L) for k in range(16))
                + (AP(32, BD.MARK_PC, L),)
                + pc_q_gate
            ),
            k=_addr_key_match_writes(BD, L) + pc_k_gate,
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
                AP(0, BD.MARK_AX, L),
                AP(0, BD.HAS_SE, -L),
                AP(pc_lo, BD.CONST, L),
                AP(16 + pc_hi, BD.CONST, L),
                AP(32, BD.MARK_AX, L),
                *ax_q_gate,
                AP(34, BD.HAS_SE, -500.0),
            ),
            k=_addr_key_match_writes(BD, L) + ax_k_gate + (AP(34, BD.CONST, 5.0),),
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
                tuple(AP(k, BD.EMBED_LO + k, L) for k in range(16))
                + tuple(AP(16 + k, BD.EMBED_HI + k, L) for k in range(16))
                + (AP(32, BD.MARK_PC, L),)
                + pc_q_gate
                + (AP(34, BD.HAS_SE, 500.0), AP(34, BD.CONST, -500.0))
            ),
            k=_addr_key_match_writes(BD, L) + pc_k_gate + (AP(34, BD.CONST, 5.0),),
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
        from ...vm_step import _set_opcode_decode_ffn
        _set_opcode_decode_ffn(block.ffn, S, _as_setdim_proxy(dim_positions))

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
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#how-bytecode-is-passed-to-the-network",
    )


def _bake_opcode_decode_ffn(ffn, S, BD):
    """Declarative L5 FFN spec: opcode-byte one-hot decode."""

    from ...embedding import Opcode

    unit = 0
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
    for op_val, lo, hi in opcodes:
        ffn.W_up.data[unit, BD.OPCODE_BYTE_LO + lo] = S
        ffn.W_up.data[unit, BD.OPCODE_BYTE_HI + hi] = S
        ffn.b_up.data[unit] = -S * 1.5
        ffn.W_gate.data[unit, BD.MARK_AX] = 1.0
        ffn.W_down.data[BD.opcode_dim(op_val), unit] = 10.0 / S
        unit += 1

    first_step_opcodes = [
        (2, 0, BD.OP_JMP),
        (3, 0, BD.TEMP + 0),
        (1, 0, BD.OP_IMM),
        (0, 0, BD.OP_LEA),
        (6, 2, BD.OP_EXIT),
        (7, 2, BD.OP_NOP),
        (9, 1, BD.OP_ADD),
        (10, 1, BD.OP_SUB),
        (11, 1, BD.OP_MUL),
        (12, 1, BD.OP_DIV),
        (13, 1, BD.OP_MOD),
        (14, 0, BD.OP_OR),
        (15, 0, BD.OP_XOR),
        (0, 1, BD.OP_AND),
        (1, 1, BD.OP_EQ),
        (3, 1, BD.OP_LT),
        (7, 1, BD.OP_SHL),
        (8, 1, BD.OP_SHR),
    ]
    for lo, hi, out_dim in first_step_opcodes:
        ffn.W_up.data[unit, BD.OPCODE_BYTE_LO + lo] = S
        ffn.W_up.data[unit, BD.OPCODE_BYTE_HI + hi] = S
        ffn.W_up.data[unit, BD.MARK_PC] = S
        ffn.W_up.data[unit, BD.HAS_SE] = -S
        ffn.b_up.data[unit] = -S * 2.5
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[out_dim, unit] = 10.0 / S
        unit += 1

    for k in range(32):
        if k == 0:
            unit += 1
            continue
        ffn.W_up.data[unit, BD.MARK_PC] = S
        ffn.b_up.data[unit] = -S * 0.5
        ffn.W_gate.data[unit, BD.TEMP + k] = -1.0
        ffn.W_down.data[BD.TEMP + k, unit] = 2.0 / S
        unit += 1

    for op_val, lo, hi in (
        (Opcode.BZ, 4, 0),
        (Opcode.BNZ, 5, 0),
        (Opcode.LEV, 8, 0),
        (Opcode.EXIT, 6, 2),
        (Opcode.JMP, 2, 0),
    ):
        ffn.W_up.data[unit, BD.OPCODE_BYTE_LO + lo] = S
        ffn.W_up.data[unit, BD.OPCODE_BYTE_HI + hi] = S
        ffn.W_up.data[unit, BD.MARK_PC] = S
        ffn.b_up.data[unit] = -S * 2.5
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.opcode_dim(op_val), unit] = 10.0 / S
        unit += 1


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
