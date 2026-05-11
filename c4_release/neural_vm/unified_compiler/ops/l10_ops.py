"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ..layer_compiler import Operation
from .shared import _as_setdim_proxy
from .shared import _bake_post_op_into


def make_layer10_carry_relay_op() -> Operation:
    """L10 attention head 0: relay CARRY flags from AX marker to AX byte positions."""
    def bake(attn, dim_positions, S):
        from ...vm_step import _set_layer10_carry_relay
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer10_carry_relay(attn, S, _as_setdim_proxy(dim_positions), HD)

    return Operation(
        name="layer10_carry_relay",
        phase=10,
        reads={"MARK_AX", "IS_BYTE", "H1", "CARRY"},
        writes={"CARRY"},  # broadcast
        kind="attn",
        bake_fn=bake,
    )


def make_layer10_byte_passthrough_op() -> Operation:
    """L10 attention head 1: AX byte passthrough across steps."""
    def bake(attn, dim_positions, S):
        from ...vm_step import _set_layer10_byte_passthrough
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer10_byte_passthrough(attn, S, _as_setdim_proxy(dim_positions), HD)

    return Operation(
        name="layer10_byte_passthrough",
        phase=10,
        reads={"IS_BYTE", "HAS_SE", "OP_IMM", "TEMP",
               "H1", "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="attn",
        bake_fn=bake,
    )


def make_layer10_sp_byte_passthrough_op() -> Operation:
    """L10 attention head 2: SP byte passthrough."""
    def bake(attn, dim_positions, S):
        from ...vm_step import _set_layer10_sp_byte_passthrough
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer10_sp_byte_passthrough(attn, S, _as_setdim_proxy(dim_positions), HD)

    return Operation(
        name="layer10_sp_byte_passthrough",
        phase=10,
        reads={"IS_BYTE", "HAS_SE", "H1",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="attn",
        bake_fn=bake,
    )


def make_layer10_psh_stack0_passthrough_op() -> Operation:
    """L10 attention head 3: PSH STACK0 passthrough."""
    def bake(attn, dim_positions, S):
        from ...vm_step import _set_layer10_psh_stack0_passthrough
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer10_psh_stack0_passthrough(attn, S, _as_setdim_proxy(dim_positions), HD)

    return Operation(
        name="layer10_psh_stack0_passthrough",
        phase=10,
        reads={"MARK_STACK0", "OP_PSH", "AX_CARRY_LO", "AX_CARRY_HI",
               "OP_LI", "OP_LC", "OP_SI", "OP_SC"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="attn",
        bake_fn=bake,
    )


# -- L10 attention bake ops (migrated 2026-05-10) -----------------------------
#
# These five ``kind="block", layer_idx=10, migrated=True`` ops bake the five
# inline ``_set_layer10_*`` attention calls that used to live in
# ``set_vm_weights`` (both the ``alu_mode == 'lookup'`` and
# ``alu_mode == 'efficient'`` branches). The inline calls have been removed
# from both branches; these ops now own the bake. Phases 10.0-10.4 preserve
# the original ordering. The five ``layer10_*`` kind="attn" placeholders
# above are retained (no ``migrated=True``) as dep-graph anchors so the
# LayerCompiler topology does not shift downstream block assignments.
#
# All five target ``model.blocks[10].attn`` and run BEFORE legacy_bake (999),
# so the alibi_slopes mutations and the L10 FFN bake inside set_vm_weights
# still execute in their original order. The attn weight slots they write
# are NOT touched by legacy_bake after the inline removals.


def make_layer10_carry_relay_bake_op() -> Operation:
    """Bake ``_set_layer10_carry_relay`` into ``model.blocks[10].attn``.

    Was an inline call in ``set_vm_weights`` (both lookup and efficient
    branches): ``_set_layer10_carry_relay(attn10, S, BD, HD)``. Inline call
    removed; this op now owns the bake. Phase=10.0 preserves the original
    ordering relative to the four sibling L10 attn bake ops below.
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer10_carry_relay
        proxy = _as_setdim_proxy(dim_positions)
        attn = block.attn
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer10_carry_relay(attn, S, proxy, HD)

    return Operation(
        name="layer10_carry_relay_bake",
        phase=10.0,
        reads={"MARK_AX", "IS_BYTE", "H1", "CARRY", "CONST"},
        writes={"CARRY"},
        kind="block",
        bake_fn=bake,
        layer_idx=10,
        migrated=True,
    )


def make_layer10_byte_passthrough_bake_op() -> Operation:
    """Bake ``_set_layer10_byte_passthrough`` into ``model.blocks[10].attn``.

    Was an inline call in ``set_vm_weights`` (both branches):
    ``_set_layer10_byte_passthrough(attn10, S, BD, HD)``. Inline call
    removed; this op now owns the bake. Phase=10.1.
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer10_byte_passthrough
        proxy = _as_setdim_proxy(dim_positions)
        attn = block.attn
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer10_byte_passthrough(attn, S, proxy, HD)

    return Operation(
        name="layer10_byte_passthrough_bake",
        phase=10.1,
        reads={"IS_BYTE", "HAS_SE", "OP_IMM", "TEMP",
               "H1", "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
               "BYTE_INDEX_3", "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        bake_fn=bake,
        layer_idx=10,
        migrated=True,
    )


def make_layer10_sp_byte_passthrough_bake_op() -> Operation:
    """Bake ``_set_layer10_sp_byte_passthrough`` into ``model.blocks[10].attn``.

    Was an inline call in ``set_vm_weights`` (both branches):
    ``_set_layer10_sp_byte_passthrough(attn10, S, BD, HD)``. Inline call
    removed; this op now owns the bake. Phase=10.2.
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer10_sp_byte_passthrough
        proxy = _as_setdim_proxy(dim_positions)
        attn = block.attn
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer10_sp_byte_passthrough(attn, S, proxy, HD)

    return Operation(
        name="layer10_sp_byte_passthrough_bake",
        phase=10.2,
        reads={"IS_BYTE", "HAS_SE", "H1", "PSH_AT_SP", "CMP",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        bake_fn=bake,
        layer_idx=10,
        migrated=True,
    )


def make_layer10_psh_stack0_passthrough_bake_op() -> Operation:
    """Bake ``_set_layer10_psh_stack0_passthrough`` into ``model.blocks[10].attn``.

    Was an inline call in ``set_vm_weights`` (both branches):
    ``_set_layer10_psh_stack0_passthrough(attn10, S, BD, HD)``. Inline call
    removed; this op now owns the bake. Phase=10.3.
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer10_psh_stack0_passthrough
        proxy = _as_setdim_proxy(dim_positions)
        attn = block.attn
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer10_psh_stack0_passthrough(attn, S, proxy, HD)

    return Operation(
        name="layer10_psh_stack0_passthrough_bake",
        phase=10.3,
        reads={"MARK_STACK0", "IS_BYTE", "PSH_AT_SP", "H1", "H4",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        bake_fn=bake,
        layer_idx=10,
        migrated=True,
    )


def make_layer10_stack0_byte_relay_bake_op() -> Operation:
    """Bake ``_set_layer10_stack0_byte_relay`` into ``model.blocks[10].attn``.

    Was an inline call in ``set_vm_weights`` (lookup branch only):
    ``_set_layer10_stack0_byte_relay(attn10, S, BD, HD)``. Inline call
    removed; this op now owns the bake. Phase=10.4.
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer10_stack0_byte_relay
        proxy = _as_setdim_proxy(dim_positions)
        attn = block.attn
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer10_stack0_byte_relay(attn, S, proxy, HD)

    return Operation(
        name="layer10_stack0_byte_relay_bake",
        phase=10.4,
        reads={"IS_BYTE", "HAS_SE", "H1", "H4",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"ALU_LO", "ALU_HI"},
        kind="block",
        bake_fn=bake,
        layer_idx=10,
        migrated=True,
    )


def make_layer10_alu_op() -> Operation:
    """L10 FFN: AND/OR/XOR + DIV/MOD setup.

    Pinned to ``layer_idx=10`` via ``kind="block"``: the legacy
    ``set_vm_weights`` lookup branch targeted ``model.blocks[10].ffn``.
    Without pinning, dep-graph layer assignment could place this op on the
    wrong block. ``phase=10.2`` is before
    ``make_l10_post_op_attach_op`` (phase=10.7) and
    ``make_l10_alu_divmod_install_op`` (phase=10.8) so they don't conflict.

    Migrated 2026-05-10: the inline ``_set_layer10_alu(ffn10, S, BD)`` call
    in the lookup branch of ``set_vm_weights`` has been removed; this op
    now owns the bake. (Per Unit 9 diagnosis, this migration is SAFE so
    long as ``make_l10_post_op_attach_op`` is NOT modified.)
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer10_alu
        _set_layer10_alu(block.ffn, S, _as_setdim_proxy(dim_positions))

    return Operation(
        name="layer10_alu",
        phase=10.2,
        reads={"MARK_AX", "ALU_LO", "AX_CARRY_LO", "ALU_HI", "AX_CARRY_HI",
               "OP_OR", "OP_XOR", "OP_AND", "OP_DIV", "OP_MOD"},
        writes={"OUTPUT_LO", "OUTPUT_HI", "DIV_STAGING"},
        kind="block",
        bake_fn=bake,
        layer_idx=10,
        migrated=True,
    )


def make_layer10_stack0_byte_relay_op() -> Operation:
    """L10 attention: STACK0 byte values relay for AND/OR/XOR multi-byte."""
    def bake(attn, dim_positions, S):
        from ...vm_step import _set_layer10_stack0_byte_relay
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer10_stack0_byte_relay(attn, S, _as_setdim_proxy(dim_positions), HD)

    return Operation(
        name="layer10_stack0_byte_relay",
        phase=10,
        reads={"MARK_AX", "IS_BYTE", "H1", "MARK_STACK0", "STACK0_BYTE0",
               "TEMP", "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"ALU_LO", "ALU_HI"},
        kind="attn",
        bake_fn=bake,
    )


def make_l10_post_ops_combined() -> Operation:
    """Combined L10 post_ops: BinaryOpByteZeroing + 3x CarryPropagation +
    BitwiseBytePropagation + ComparisonCombine, baked sequentially into
    one FFN.

    Originally these were 6 separate post_ops on L10 in vm_step.py. Per Phase 0
    policy they belong in their own blocks, but for the migration we combine
    them additively into a single ffn at phase=10.5 so the compiler places
    them right after layer10_alu.
    """
    def bake(ffn, dim_positions, S):
        from ...vm_step import (
            BinaryOpByteZeroingPostOp,
            CarryPropagationPostOp,
            BitwiseBytePropagationPostOp,
            ComparisonCombine,
        )
        d_model = ffn.W_up.shape[1]
        offset = 0
        offset = _bake_post_op_into(ffn, BinaryOpByteZeroingPostOp(d_model, S), offset)
        offset = _bake_post_op_into(ffn, CarryPropagationPostOp(d_model, S, byte_idx=0, cascade=False), offset)
        offset = _bake_post_op_into(ffn, CarryPropagationPostOp(d_model, S, byte_idx=1, cascade=True), offset)
        offset = _bake_post_op_into(ffn, CarryPropagationPostOp(d_model, S, byte_idx=2, cascade=True), offset)
        offset = _bake_post_op_into(ffn, BitwiseBytePropagationPostOp(d_model, S), offset)
        offset = _bake_post_op_into(ffn, ComparisonCombine(d_model, S), offset)

    # phase=10.5 so it lands AFTER layer10_alu (phase=10) but BEFORE later layers
    # which depend on its OUTPUT_LO/HI updates. Note: float phases work because
    # phase comparison uses < / >.
    return Operation(
        name="l10_post_ops_combined",
        phase=10.5,
        reads={
            "MARK_AX", "MARK_PC", "IS_BYTE", "H1",
            "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
            "OP_SHL", "OP_SHR",
            "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
            "OP_OR", "OP_XOR", "OP_AND",
            "OP_LEA", "OP_IMM", "OP_JMP", "OP_JSR", "OP_BZ", "OP_BNZ",
            "OP_ENT", "OP_ADJ", "OP_LEV", "OP_LI", "OP_LC",
            "OP_SI", "OP_SC", "OP_PSH", "OP_EXIT", "OP_NOP",
            "OP_PUTCHAR", "OP_GETCHAR",
            "OUTPUT_LO", "OUTPUT_HI", "ALU_LO", "ALU_HI",
            "CARRY", "CMP", "TEMP",
            "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
        },
        writes={"OUTPUT_LO", "OUTPUT_HI", "CARRY"},
        kind="ffn",
        bake_fn=bake,
    )


def make_l10_post_op_attach_op(alu_mode: str = "lookup") -> Operation:
    """Block-level op: attach L10 post_op modules onto block.post_ops.

    Migrates the inline `model.blocks[10].post_ops.append(...)` calls in
    `set_vm_weights` for both lookup and efficient ALU modes into a compiler
    block op. The attached modules are the structural post-FFN passes that
    `_expand_wrapper_blocks` later splits into their own blocks.

    Modules attached (lookup mode):
      BinaryOpByteZeroingPostOp,
      CarryPropagationPostOp x3 (byte 0 no-cascade, bytes 1-2 cascade),
      BitwiseBytePropagationPostOp.
      (DIV/MOD post_op is appended by ``make_l10_alu_divmod_install_op``
      at phase=10.8 — see ``efficient_alu_divmod_split.FlattenedDivMod``.)

    Modules attached (efficient mode):
      BinaryOpByteZeroingPostOp,
      CarryPropagationPostOp x3,
      BitwiseBytePropagationPostOp,
      ComparisonCombine.
      (DIV/MOD post_op is appended by ``make_l10_alu_divmod_install_op``
      at phase=10.8 — see ``efficient_alu_divmod_split.FlattenedDivMod``.)

    The existing `make_l10_post_ops_combined` is unrelated: it bakes the
    LOGIC of the FFN-style post_ops into a single phase-10.5 FFN (a parallel
    representation), not the attached module list. Both can coexist.

    phase=10.7: runs after L10 FFN bake (phase=10) and the combined FFN
    (phase=10.5), but well before structural post-passes (1100+).
    """
    if alu_mode not in ("lookup", "efficient"):
        raise ValueError(
            f"alu_mode must be 'lookup' or 'efficient'; got {alu_mode!r}"
        )

    def bake(block, dim_positions, S):
        from ...vm_step import (
            BinaryOpByteZeroingPostOp,
            CarryPropagationPostOp,
            BitwiseBytePropagationPostOp,
            ComparisonCombine,
        )
        # Use the block's d_model when available; fall back to 512 to mirror
        # the previous inline behavior.
        d_model = 512
        if hasattr(block, "ffn") and hasattr(block.ffn, "W_up"):
            try:
                d_model = block.ffn.W_up.shape[1]
            except (AttributeError, IndexError):
                d_model = 512

        block.post_ops.append(BinaryOpByteZeroingPostOp(d_model=d_model, S=S))
        block.post_ops.append(
            CarryPropagationPostOp(d_model=d_model, S=S, byte_idx=0, cascade=False)
        )
        block.post_ops.append(
            CarryPropagationPostOp(d_model=d_model, S=S, byte_idx=1, cascade=True)
        )
        block.post_ops.append(
            CarryPropagationPostOp(d_model=d_model, S=S, byte_idx=2, cascade=True)
        )
        block.post_ops.append(BitwiseBytePropagationPostOp(d_model=d_model, S=S))
        if alu_mode == "efficient":
            # Pass the model's actual d_model so the underlying PureFFN's
            # Linear input dim matches the residual stream width. Without
            # this, ComparisonCombine builds a Linear(512, 18) which fails
            # forward when d_model != 512 (e.g., pin_io_only=True paths).
            block.post_ops.append(ComparisonCombine(d_model=d_model, S=S))
        # DIV/MOD post_op (FlattenedDivMod) appended by
        # ``make_l10_alu_divmod_install_op`` (phase=10.8). Both modes use the
        # same flattened composite — its forward is byte-identical to the
        # previous EfficientDivMod_Neural.

    return Operation(
        name="l10_post_op_attach",
        reads=set(),
        writes=set(),
        kind="block",
        bake_fn=bake,
        phase=10.7,
        layer_idx=10,
        migrated=True,
    )


