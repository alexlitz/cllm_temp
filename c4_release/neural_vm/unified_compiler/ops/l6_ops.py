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


def make_putchar_think_protocol_op(
    enable_neural_io_think_protocol: bool = False,
) -> Operation:
    """L6 FFN: PUTCHAR THINK-tag I/O protocol — emit THINKING_END,
    output byte token, THINKING_START at end of PUTCHAR step.

    See ``c4_release/docs/NEURAL_IO_VIA_THINK_PROTOCOL_PLAN.md`` for the
    full design. This op implements the canonical neural-I/O mode from
    BLOG_SPEC.md:851: instead of the runner reading ``REG_AX`` byte 0
    off a just-completed PUTCHAR step (the current Phase-6 default),
    the model itself emits the protocol tokens
    ``THINKING_END, byte, THINKING_START`` between the MEM section and
    the STEP_END of a PUTCHAR step.

    Gated by ``enable_neural_io_think_protocol`` (default False) so the
    existing AX-readoff path remains the production default while the
    new bake is brought up. When False, the bake_fn is a no-op so the
    op stays registered for dep-graph stability — mirroring the
    ``make_tool_call_*_op`` and ``make_convo_io_*_op`` pattern.

    Phase 1 contract (this commit):
      - When enabled, AND ``IO_IS_PUTCHAR`` (set at AX marker by L5/L6
        FFN units 1500) with ``NEXT_SE`` and emit ``NEXT_THINKING_END``
        + suppress ``NEXT_SE`` (so the model writes THINKING_END
        instead of STEP_END at the end of a PUTCHAR step). This
        re-uses the convo-io state-machine pattern from
        ``_set_conversational_io_state_machine`` (setup_helpers.py:1789).
      - The L6 FFN ``_set_io_putchar_routing`` units (1500-1532) already
        write ``AX_CARRY → OUTPUT_LO/HI``; the same routing populates
        the next-byte-token slot so the model emits the actual byte
        right after THINKING_END.
      - Closing ``THINKING_START`` emission and the trailing STEP_END
        are deferred to a follow-up bake (see plan doc § B5).

    Phase 1 stub (this commit, enable_neural_io_think_protocol=False):
      - The bake_fn is a no-op. The op is registered for dep-graph
        stability and to expose the scaffolding so a follow-up worker
        can flip ``enable_neural_io_think_protocol=True`` and fill in
        the weight wiring without restructuring the migration chain.
      - The intended unit allocation (when enabled) is L6 FFN unit
        ~1402, immediately above ``_set_conversational_io_state_machine``
        units 1400-1401. This avoids overlap with the
        routing-FFN range (units 0-1033),
        ``_set_tool_call_detection`` (unit 1300),
        ``_set_conversational_io_state_machine`` (units 1400-1401),
        ``_set_io_putchar_routing`` (units 1500-1532), and the
        function-call-weights units (1700-2158).

    Phase 6.6 (pinned ``layer_idx=6``, ``kind="block"``): runs alongside
    other L6 block ops so the FFN unit writes survive ``_right_size_ffns``
    (phase=1200) trimming. Same phase as
    ``make_convo_io_state_machine_op`` since both extend the same L6
    FFN with adjacent unit ranges and reads/writes are disjoint.
    """
    def bake(block, dim_positions, S):
        # Phase 1 no-op stub. When ``enable_neural_io_think_protocol``
        # flips to True the body will route ``IO_IS_PUTCHAR`` through the
        # same state-machine entry as PRTF/READ (see
        # ``_set_conversational_io_state_machine`` at setup_helpers.py:1789
        # for the CMP[5]/CMP[6] pattern this will mirror) and write a new
        # L6 FFN unit at ~1402. The runner-side collector in run_vm.py is
        # already in place behind the same flag — the follow-up worker
        # only needs to fill in the weight writes here.
        return

    return Operation(
        name="putchar_think_protocol",
        phase=6.6,
        reads={"IO_IS_PUTCHAR", "NEXT_SE", "AX_CARRY_LO", "AX_CARRY_HI"},
        writes={"NEXT_THINKING_END", "NEXT_SE", "IO_STATE",
                "OUTPUT_BYTE_LO", "OUTPUT_BYTE_HI"},
        kind="block",
        bake_fn=bake,
        layer_idx=6,
        migrated=True,
    )


