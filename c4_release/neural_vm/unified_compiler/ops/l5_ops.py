"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ..layer_compiler import Operation
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
        from ...vm_step import _set_layer5_fetch
        attn = block.attn
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
        _set_layer5_fetch(attn, S, _as_setdim_proxy(dim_positions), HD)

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
        migrated=True,
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#how-bytecode-is-passed-to-the-network",
    )


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
        migrated=True,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#how-bytecode-is-passed-to-the-network",
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
        smoke_tests=set(),
        spec_section=None,
    )


