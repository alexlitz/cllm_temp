"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ..layer_compiler import Operation
from .shared import _as_setdim_proxy


def make_layer6_attn_op() -> Operation:
    """L6 attention: relay heads for IS_JMP, IS_EXIT, etc. at PC marker.

    Kept as a ``kind="attn"`` dep anchor (no ``migrated`` flag, no
    ``layer_idx``): its declared reads/writes preserve the LayerCompiler
    dep-graph topology that places downstream ops (e.g. ALU layers L8-L13)
    at the right model.blocks indices. The actual weight bake now happens
    in ``make_layer6_attn_bake_op`` (kind="model", phase=998.5, migrated=True);
    this op's bake_fn is preserved for the legacy pure_neural path (when
    ``legacy_bake`` is absent) but does no work in production where
    legacy_bake owns the dispatch.
    """
    def bake(attn, dim_positions, S):
        from ...vm_step import _set_layer6_attn
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer6_attn(attn, S, _as_setdim_proxy(dim_positions), HD)

    return Operation(
        name="layer6_attn",
        phase=6,
        reads={"OP_JMP", "OP_EXIT", "OP_JSR", "MARK_AX", "MARK_PC", "MARK_SP",
               "MARK_STACK0", "NEXT_SE", "FETCH_LO", "FETCH_HI",
               "PSH_AT_SP", "OP_PSH", "OP_ADJ", "OP_ENT", "OP_LEV",
               "AX_CARRY_LO", "AX_CARRY_HI"},
        writes={"CMP", "AX_CARRY_LO", "AX_CARRY_HI"},
        kind="attn",
        bake_fn=bake,
    )


def make_layer6_routing_ffn_op() -> Operation:
    """L6 FFN: per-opcode routing — write FETCH/AX_CARRY → OUTPUT, etc.

    Originally an inline call in `set_vm_weights`:
        `_set_layer6_routing_ffn(ffn6, S, BD)`

    Pinned to ``layer_idx=6`` via ``kind="block"`` so the bake hits the same
    transformer block (block[6].ffn) the legacy path used (mirrors the L4 FFN
    regression fix: dep-graph assignment alone can place a kind="ffn" op on a
    different block, leaving block[6] unbaked). ``migrated=True`` skips the
    legacy_bake path for this op; the inline call in ``set_vm_weights`` has
    been removed to avoid double-bake.

    Phase 6.5: runs AFTER the L6 attention ops (phases 6.0-6.2) and before
    other L6 FFN extension ops migrated to model-level phase=998.
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer6_routing_ffn
        _set_layer6_routing_ffn(block.ffn, S, _as_setdim_proxy(dim_positions))

    return Operation(
        name="layer6_routing_ffn",
        phase=6.5,
        reads={"OP_IMM", "OP_EXIT", "OP_JMP", "OP_NOP", "OP_LEA",
               "MARK_AX", "MARK_PC", "MARK_STACK0", "MARK_BP",
               "IS_BYTE", "FETCH_LO", "FETCH_HI",
               "AX_CARRY_LO", "AX_CARRY_HI", "CMP",
               "OUTPUT_LO", "OUTPUT_HI", "HAS_SE",
               "OPCODE_BASE", "OUTPUT_BYTE_LO", "OUTPUT_BYTE_HI",
               "TEMP", "DIV_STAGING"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        bake_fn=bake,
        layer_idx=6,
        migrated=True,
    )


def make_layer6_relay_heads_op() -> Operation:
    """L6 head 6/7: STACK0 ← AX relay for PSH.

    Kept as a ``kind="attn"`` dep anchor (no ``migrated`` flag, no
    ``layer_idx``): its declared reads/writes preserve the LayerCompiler
    dep-graph topology. The actual weight bake now happens in
    ``make_layer6_relay_heads_bake_op`` (kind="model", phase=998.6,
    migrated=True).
    """
    def bake(attn, dim_positions, S):
        from ...vm_step import _set_layer6_relay_heads
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer6_relay_heads(attn, S, _as_setdim_proxy(dim_positions), HD)

    return Operation(
        name="layer6_relay_heads",
        phase=6,
        reads={"MARK_STACK0", "MARK_AX", "AX_CARRY_LO", "AX_CARRY_HI"},
        writes={"ALU_LO", "ALU_HI"},
        kind="attn",
        bake_fn=bake,
    )


def make_layer6_attn_bake_op() -> Operation:
    """Bake ``_set_layer6_attn`` into ``model.blocks[6].attn``.

    Originally an inline call in ``set_vm_weights``:
        ``_set_layer6_attn(attn6, S, BD, HD)``

    Migrated as ``kind="model"`` with ``migrated=True``: the inline call
    has been removed, so the bake must happen here. Phase=998.5 so this
    op runs AFTER ``function_call_weights`` (998) but BEFORE
    ``legacy_bake`` (999), matching the legacy in-set_vm_weights ordering
    where _set_layer6_attn ran after function_call_weights.

    A model-level op (rather than ``kind="block"`` pinned to ``layer_idx=6``)
    is required because compile_full_vm dispatches block ops BEFORE all
    model ops, which would flip the override contract between this op's
    sibling ``layer6_relay_heads_bake`` (head 7 Q[MARK_STACK0]=50) and
    ``function_call_weights`` (head 7 Q[MARK_STACK0]=1050). Operating on
    ``model.blocks[6].attn`` directly sidesteps the dep-graph
    layer-assignment risk that motivates the L4 FFN regression's pinning.
    The dep anchor ``layer6_attn`` (kind="attn") above preserves the
    LayerCompiler topology so downstream ops (L8-L13 ALU) remain placed
    at their legacy block indices.
    """
    def bake(model, dim_positions, S):
        from ...vm_step import _set_layer6_attn
        attn = model.blocks[6].attn
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer6_attn(attn, S, _as_setdim_proxy(dim_positions), HD)

    return Operation(
        name="layer6_attn_bake",
        phase=998.5,
        reads=set(),
        writes=set(),
        kind="model",
        bake_fn=bake,
        migrated=True,
    )


def make_layer6_relay_heads_bake_op() -> Operation:
    """Bake ``_set_layer6_relay_heads`` into ``model.blocks[6].attn``.

    Originally an inline call in ``set_vm_weights``:
        ``_set_layer6_relay_heads(attn6, S, BD, HD)``

    Migrated as ``kind="model"`` with ``migrated=True``: the inline call
    has been removed. Phase=998.6 so this op runs AFTER
    ``function_call_weights`` (998) AND AFTER ``layer6_attn_bake`` (998.5)
    but BEFORE ``legacy_bake`` (999).

    The ordering matters: this op writes head 7 Q[MARK_STACK0]=50,
    Q[MARK_AX]=-50, K[MARK_AX]=50, which overlap with
    ``function_call_weights``'s head 7 writes (Q[MARK_STACK0]=1050,
    Q[CONST]=-1000, K[MARK_PC]=30, K[OP_JSR]=-20). Legacy
    in-set_vm_weights ordering had relay_heads run after function_call
    weights, so relay_heads's Q[MARK_STACK0]=50 overrides the JSR-side
    1050 (PSH semantics win). The phase ordering 998 < 998.6 < 999
    preserves that override contract; the dep anchor ``layer6_relay_heads``
    above preserves the LayerCompiler topology.
    """
    def bake(model, dim_positions, S):
        from ...vm_step import _set_layer6_relay_heads
        attn = model.blocks[6].attn
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer6_relay_heads(attn, S, _as_setdim_proxy(dim_positions), HD)

    return Operation(
        name="layer6_relay_heads_bake",
        phase=998.6,
        reads=set(),
        writes=set(),
        kind="model",
        bake_fn=bake,
        migrated=True,
    )


def make_layer6_bz_bnz_relay_bake_op() -> Operation:
    """Bake ``_set_bz_bnz_relay`` into ``model.blocks[6].attn`` (head 4).

    Originally an inline call in ``set_vm_weights``:
        ``_set_bz_bnz_relay(attn6, S, BD, HD)``

    Migrated as ``kind="model"`` with ``migrated=True``: the inline call
    has been removed. Phase=998.7 so this op runs AFTER
    ``layer6_attn_bake`` (998.5) and ``layer6_relay_heads_bake`` (998.6),
    BEFORE ``legacy_bake`` (999), matching the original in-set_vm_weights
    order. _set_bz_bnz_relay programs head 4's Q/K/V/O slots, left
    intentionally unprogrammed by ``_set_layer6_attn`` (head 4 is reserved
    for BZ/BNZ relay per the comment in _set_layer6_attn).

    No dep anchor needed: the legacy code never declared an op for this
    function, so its absence from the dep graph is the existing baseline.
    """
    def bake(model, dim_positions, S):
        from ...vm_step import _set_bz_bnz_relay
        attn = model.blocks[6].attn
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_bz_bnz_relay(attn, S, _as_setdim_proxy(dim_positions), HD)

    return Operation(
        name="layer6_bz_bnz_relay_bake",
        phase=998.7,
        reads=set(),
        writes=set(),
        kind="model",
        bake_fn=bake,
        migrated=True,
    )


def make_binary_pop_sp_increment_op() -> Operation:
    """L6 FFN extension: SP += 8 for binary-pop ops (ADD/SUB/etc.).

    Originally an inline call in `set_vm_weights`:
        `_set_binary_pop_sp_increment(ffn6, S, BD)`

    Operates on `model.blocks[6].ffn` (L6 FFN). Modeled as kind="model" so we
    can resolve `ffn6` from the model handle inside the bake_fn.

    Phase 998: runs just BEFORE legacy_bake (999) so that the L6 FFN units we
    program (starting at unit 2200) are present when `_right_size_ffns`
    (called at the end of legacy_bake) prunes dead units. Running at phase
    > 999 would write into already-rightsized FFN slots that no longer exist.
    """
    def bake(model, dim_positions, S):
        from ...vm_step import _set_binary_pop_sp_increment
        proxy = _as_setdim_proxy(dim_positions)
        _set_binary_pop_sp_increment(model.blocks[6].ffn, S, proxy)

    return Operation(
        name="binary_pop_sp_increment",
        reads=set(),
        writes=set(),
        kind="model",
        bake_fn=bake,
        phase=998,
        migrated=True,
    )


