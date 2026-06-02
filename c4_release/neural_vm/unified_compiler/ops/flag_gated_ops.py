"""Flag-gated factories (tool-call + conversational I/O). See ../migrated_ops.py for history."""

from ...dim_registry import dim_ref
from ..ir import CompilerIR, FFNRule
from ..layer_compiler import Operation
from ..primitives import AO, AP, DeclarativeAttentionHeadSpec, Primitives
from .shared import _as_setdim_proxy


# ---------------------------------------------------------------------------
# Tool-calling bakes (gated by enable_tool_calling)
#
# The 3 ops below replace inline calls in set_vm_weights that fire only when
# `enable_tool_calling=True`:
#   _set_tool_call_opcode_decode(ffn5, S, BD)  -> L5 FFN units 400-405
#   _set_tool_call_relay_head(attn6, S, BD, HD) -> L6 attn head 5
#   _set_tool_call_detection(ffn6, S, BD)      -> L6 FFN unit 1300
#
# All three are kind="model" at phase 998.8: they run AFTER io_putchar_routing
# (998), function_call_weights (998), the L6 attn/relay bakes (998.5/.6/.7)
# but BEFORE legacy_bake (999) and its right_size_ffns (phase 1200), so the
# FFN unit writes survive the rightsize pass. None of the units they touch
# (400-405 in L5 FFN, 1300 in L6 FFN, head 5 in L6 attn) conflict with the
# other ops dispatched at phase < 998.8.
#
# Each factory takes `enable_tool_calling=False`; when False the bake_fn is
# a no-op (the op is always registered to keep the registration list stable
# regardless of mode). The L6 attn's `alibi_slopes[5] = 5.0` mutation that
# used to live inline next to _set_tool_call_relay_head is folded into the
# relay-head bake_fn so the op is self-contained.
# ---------------------------------------------------------------------------

def make_tool_call_opcode_decode_op(enable_tool_calling: bool = False) -> Operation:
    """Bake L5 FFN tool-call opcode decoder (IO opcodes -> IO_IS_TOOL_CALL).

    Originally an inline call in `set_vm_weights` (inside `if enable_tool_calling:`):
        `_set_tool_call_opcode_decode(ffn5, S, BD)`

    Operates on `model.blocks[5].ffn`. Modeled as `kind="model"` so the bake_fn
    can resolve ffn5 from the model handle, matching `make_io_putchar_routing_op`.

    Phase 998.8: runs after the L5 main FFN bakes (phase ~5-6) and the L6
    bake ops (998.x), and before legacy_bake (999) so the unit writes survive
    `_right_size_ffns` (which runs inside legacy_bake).

    When `enable_tool_calling=False`, the bake_fn is a no-op so the op can be
    unconditionally registered in `all_core_ops()` without changing behavior.

    Migrated 2026-06-01 (Phase 6 Wave 3L): the imperative helper
    ``_set_tool_call_opcode_decode`` is replaced by an
    ``FFNRule``-driven lowerer (``_lower_tool_call_opcode_decode_ir``)
    — six ``FFNRule.gated_write`` units lowered through
    ``Primitives.lower_ffn_rules`` at pinned units 400..405.
    Byte-identical to the legacy helper.
    """
    def bake(model, dim_positions, S):
        if not enable_tool_calling:
            return
        _lower_tool_call_opcode_decode_ir(
            model.blocks[5].ffn, S, _as_setdim_proxy(dim_positions),
        )

    return Operation(
        name="tool_call_opcode_decode",
        reads=set(),
        writes=set(),
        kind="model",
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        phase=998.8,
        migrated=True,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#tool-use-mode",
    )


_TOOL_CALL_OPCODE_DECODE_START_UNIT = 400

# (OPCODE_BYTE_LO nibble, OPCODE_BYTE_HI nibble, opcode name).  Mirrors the
# ``io_opcodes`` table in ``_set_tool_call_opcode_decode``: OPEN=30 (0x1E),
# READ=31 (0x1F), CLOS=32 (0x20), PRTF=33 (0x21), GETCHAR=64 (0x40),
# PUTCHAR=65 (0x41).
_TOOL_CALL_OPCODE_TABLE = (
    (14, 1, "open"),
    (15, 1, "read"),
    (0, 2, "clos"),
    (1, 2, "prtf"),
    (0, 4, "getchar"),
    (1, 4, "putchar"),
)


def _tool_call_opcode_decode_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for ``_set_tool_call_opcode_decode``.

    Six gated-write units, one per I/O opcode: 2-way AND on
    ``OPCODE_BYTE_LO/HI`` nibbles, MARK_AX gate, writes ``IO_IS_TOOL_CALL``.
    Each unit's effective contribution is ``10.0/S`` so the combined flag
    is roughly 5.0 when an I/O opcode is active.

    Phase 8.D: the ``MARK_AX`` gate uses :func:`dim_ref` for the
    ``(marker, AX)`` semantic pair. Structural ``OPCODE_BYTE_LO/HI+{lo,hi}``
    condition reads stay as ``+N`` (per-nibble one-hot lookups, not
    role-meaningful byte positions). ``IO_IS_TOOL_CALL`` is a slot-level
    flag with no ``(category, role)`` binding so the write stays bare.
    """
    write_scale = 10.0 / S
    gate_ax = dim_ref("marker", "AX")
    return tuple(
        FFNRule.gated_write(
            name=f"tool_call_decode_{name}",
            conditions=(
                (f"OPCODE_BYTE_LO+{lo}", 1.0),
                (f"OPCODE_BYTE_HI+{hi}", 1.0),
            ),
            threshold=1.5,
            gate=gate_ax,
            writes=(("IO_IS_TOOL_CALL", write_scale),),
        )
        for lo, hi, name in _TOOL_CALL_OPCODE_TABLE
    )


def _lower_tool_call_opcode_decode_ir(ffn, S: float, BD) -> int:
    rules = _tool_call_opcode_decode_rules(S)
    dim_positions = Primitives.dim_positions_from_bd(
        BD,
        Primitives.ffn_rule_dim_names(rules),
    )
    return Primitives.lower_ffn_rules(
        ffn,
        rules,
        dim_positions,
        start_unit=_TOOL_CALL_OPCODE_DECODE_START_UNIT,
        S=S,
    )


def make_convo_io_opcode_decode_op(enable_conversational_io: bool = False) -> Operation:
    """Bake ``_set_conversational_io_opcode_decode`` into ``model.blocks[5].ffn``.

    Originally an inline call in ``set_vm_weights`` under
    ``if enable_conversational_io:``:
        ``_set_conversational_io_opcode_decode(ffn5, S, BD)``

    Decodes PRTF/READ opcodes at the AX marker and writes IO_IS_PRTF /
    IO_IS_READ flags (L5 FFN units 410-411). Gated by
    ``enable_conversational_io``: when the flag is False the bake_fn is a
    no-op so the op can remain registered for dep-graph stability. The op
    is pinned to ``layer_idx=5`` via ``kind="block"`` so the bake hits
    ``model.blocks[5].ffn`` (matching the legacy direct-FFN access). Block
    ops run BEFORE legacy_bake (phase 999) and right_size_ffns (phase 1200),
    so the FFN units we write survive rightsize trimming.

    Phase 5.6 places this AFTER ``opcode_decode_ffn`` (phase 5; same L5 FFN)
    so its base opcode decoder is in place before the convo_io extension
    units are added, matching the legacy in-set_vm_weights ordering.
    """
    def bake(block, dim_positions, S):
        if not enable_conversational_io:
            return
        _lower_convo_io_opcode_decode_ir(
            block.ffn,
            S,
            _as_setdim_proxy(dim_positions),
        )

    return Operation(
        name="convo_io_opcode_decode",
        phase=5.6,
        reads=set(),
        writes=set(),
        kind="block",
        # Phase 8.A.4: dropped ``layer_idx=5`` in favour of
        # ``target_op_name``. Binds to whichever layer the compiler
        # placed ``_layer5_fetch_dep_anchor`` (the L5 dep anchor).
        target_op_name="_layer5_fetch_dep_anchor",
        migrated=True,
        declarative_bake_fn=bake,
        # When ``enable_conversational_io=False`` the bake_fn is a true
        # no-op — declarations-only / symbolic tooling treats the op as
        # vacuous, so we expose ``compiler_ir`` only when the flag is on.
        # The rules cover the 2 FFN units the helper writes at the L5
        # FFN's 410-411 pin offsets; the IR itself is offset-agnostic
        # (``start_unit=`` is supplied by ``_lower_convo_io_opcode_decode_ir``
        # at bake time, and by the verifier / symbolic runner at index 0
        # for declaration-semantics + per-rule lowering-contract checks).
        compiler_ir=(
            _convo_io_opcode_decode_ir()
            if enable_conversational_io
            else None
        ),
        declarative_authority="spec_generated",
        ffn_units_used=412 if enable_conversational_io else None,
        # B12 backfill: docstring above names ``opcode_decode_ffn`` (phase
        # 5, same L5 FFN) as the required predecessor — its base opcode
        # decoder must be in place before this extension writes units
        # 410-411. Encoded as a B10 op-name reference so the dynamic
        # scheduler honours the dep edge even though reads/writes are
        # empty (bake body flag-gated on ``enable_conversational_io``).
        requires={"after": "opcode_decode_ffn"},
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#printing-and-reading-input",
    )


def _convo_io_opcode_decode_ir(S: float = 100.0) -> CompilerIR:
    """Build the declarative ``CompilerIR`` for ``convo_io_opcode_decode``.

    Exposed via ``compiler_ir=`` (only when ``enable_conversational_io``
    is True) so symbolic / decl-verifier tooling sees the same FFNRule
    declarations the bake lowers. The production bake itself stays in
    :func:`_lower_convo_io_opcode_decode_ir` because it pins
    ``start_unit=410`` while ``CompilerIR.lower_ffn`` lowers starting at
    ``start_unit=0`` — both produce the same per-rule weights at their
    respective unit offsets.
    """

    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_convo_io_opcode_decode_rules(S))
    return ir


def _convo_io_opcode_decode_rules(S: float) -> tuple[FFNRule, ...]:
    """Two gated-write units detecting PRTF / READ opcodes at MARK_AX.

    Phase 8.D: the ``MARK_AX`` gate uses :func:`dim_ref` for the
    ``(marker, AX)`` semantic pair. Structural ``OPCODE_BYTE_LO/HI+{n}``
    condition reads stay as ``+N`` (per-nibble one-hot lookups, not
    role-meaningful byte positions). ``IO_IS_PRTF`` / ``IO_IS_READ`` /
    ``ACTIVE_OPCODE_*`` are slot-level flags with no ``(category, role)``
    binding so the writes stay bare.
    """
    write_io = 10.0 / S
    write_active = 2.0 / S
    gate_ax = dim_ref("marker", "AX")
    return (
        FFNRule.gated_write(
            name="convo_io_decode_prtf",
            conditions=(
                ("OPCODE_BYTE_LO+1", 1.0),
                ("OPCODE_BYTE_HI+2", 1.0),
            ),
            threshold=1.5,
            gate=gate_ax,
            writes=(
                ("IO_IS_PRTF", write_io),
                ("ACTIVE_OPCODE_PRTF", write_active),
            ),
        ),
        FFNRule.gated_write(
            name="convo_io_decode_read",
            conditions=(
                ("OPCODE_BYTE_LO+15", 1.0),
                ("OPCODE_BYTE_HI+1", 1.0),
            ),
            threshold=1.5,
            gate=gate_ax,
            writes=(
                ("IO_IS_READ", write_io),
                ("ACTIVE_OPCODE_READ", write_active),
            ),
        ),
    )


def _lower_convo_io_opcode_decode_ir(ffn, S: float, BD) -> int:
    rules = _convo_io_opcode_decode_rules(S)
    dim_positions = Primitives.dim_positions_from_bd(
        BD,
        Primitives.ffn_rule_dim_names(rules),
    )
    return Primitives.lower_ffn_rules(
        ffn,
        rules,
        dim_positions,
        start_unit=410,
        S=S,
    )


def make_tool_call_relay_head_op(enable_tool_calling: bool = False) -> Operation:
    """Bake L6 attention head 5 for tool-call relay (IO_IS_TOOL_CALL AX -> SE).

    Originally two inline statements in `set_vm_weights` (inside
    `if enable_tool_calling:`):
        `attn6.alibi_slopes[5] = 5.0`
        `_set_tool_call_relay_head(attn6, S, BD, HD)`

    Both folded into this single op so the bake is self-contained: the
    alibi_slopes mutation lives next to the head bake that depends on it.

    Operates on `model.blocks[6].attn`. Modeled as `kind="model"` so the
    bake_fn can resolve attn6 from the model handle (and look up HD), matching
    `make_layer6_attn_bake_op`.

    Phase 998.8: runs after the L6 attn bakes (`layer6_attn_bake` 998.5,
    `layer6_relay_heads_bake` 998.6, `layer6_bz_bnz_relay_bake` 998.7) which
    fill_(0.0) and then set heads 0-4/6-7. Head 5 is left at 0.0 by those
    earlier bakes — this op sets head 5's alibi slope and its Q/K/V/O slots.
    Runs before legacy_bake (999).

    When `enable_tool_calling=False`, the bake_fn is a no-op.

    Migrated 2026-06-01 (Phase 6 Wave 3L): the imperative helper
    ``_set_tool_call_relay_head`` is replaced by a
    ``DeclarativeAttentionHeadSpec`` lowered through
    ``Primitives.generate_attention_head`` (matching the
    ``make_convo_io_relay_heads_op`` pattern). Byte-identical to the
    legacy helper. The ``alibi_slopes[5] = 5.0`` mutation stays inline
    in the bake_fn (alibi slopes are not part of the declarative
    attention IR).
    """
    def bake(model, dim_positions, S):
        del S
        if not enable_tool_calling:
            return
        attn = model.blocks[6].attn
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes[5] = 5.0  # steep ALiBi for head 5
        HD = attn.W_q.shape[0] // attn.num_heads
        Primitives.generate_attention_head(
            attn,
            _tool_call_relay_head_spec(_as_setdim_proxy(dim_positions)),
            HD,
        )

    return Operation(
        name="tool_call_relay_head",
        reads=set(),
        writes=set(),
        kind="model",
        declarative_bake_fn=bake,
        compiler_ir_factory=(
            _tool_call_relay_head_ir if enable_tool_calling else None
        ),
        declarative_authority="spec_generated",
        phase=998.8,
        migrated=True,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#tool-use-mode",
    )


def _tool_call_relay_head_ir(dim_positions, HD) -> CompilerIR:
    del HD
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.append(_tool_call_relay_head_spec(proxy))
    return ir


def _tool_call_relay_head_spec(BD) -> DeclarativeAttentionHeadSpec:
    """Declarative L6 attention head 5: relay IO_IS_TOOL_CALL AX -> SE.

    Mirrors ``_set_tool_call_relay_head`` exactly:
      - Q: ``NEXT_SE`` × L (query at SE position), ``MARK_AX`` × -L (block at AX)
      - K: ``MARK_AX`` × L (attend to AX marker)
      - V slot 1: copy ``IO_IS_TOOL_CALL`` (×1.0)
      - O: write slot-1 value into ``CMP+2`` (×1.0, the IS_TOOL_CALL relay)
    """
    L = 50.0
    return DeclarativeAttentionHeadSpec(
        head_idx=5,
        q=(
            AP(0, BD.NEXT_SE, L),
            AP(0, BD.MARK_AX, -L),
        ),
        k=(AP(0, BD.MARK_AX, L),),
        v=(AP(1, BD.IO_IS_TOOL_CALL, 1.0),),
        o=(AO(BD.CMP + 2, 1, 1.0),),
    )


def make_convo_io_relay_heads_op(enable_conversational_io: bool = False) -> Operation:
    """Bake ``_set_conversational_io_relay_heads`` into ``model.blocks[6].attn``.

    Originally an inline call in ``set_vm_weights`` under
    ``if enable_conversational_io:`` (with two ALiBi slope mutations folded
    in here for self-containment):
        ``attn6.alibi_slopes[4] = 5.0  # PRTF relay``
        ``attn6.alibi_slopes[5] = 5.0  # READ relay``
        ``_set_conversational_io_relay_heads(attn6, S, BD, HD)``

    Programs L6 attention heads 4-5 to relay IO_IS_PRTF / IO_IS_READ from
    the AX marker → SE position. Head 4 also shares Q/K rows with
    ``_set_bz_bnz_relay`` (programmed at phase 998.7) but writes
    non-overlapping dim columns (Q[NEXT_SE/ACTIVE_OPCODE_PRTF] vs
    Q[MARK_PC/OP_BZ/OP_BNZ]; V[base+37] vs V[base+1..36]; O[CMP+5,base+37]
    vs O[CMP+5,base+4]).

    Phase 999.5 (``kind="model"``): MUST run AFTER ``legacy_bake`` (999)
    because legacy_bake's L6 setup begins with
    ``attn6.alibi_slopes.fill_(0.0)`` followed by per-head slope assignments
    for heads 0-4 / 6-7. A block-op phase (which would run before the model
    ops including legacy_bake) sees its alibi slopes for heads 4-5 clobbered
    by that fill_. Running here, after legacy_bake's slope setup, preserves
    the ``alibi_slopes[4]=5.0, [5]=5.0`` writes. The Q/K/V/O writes are also
    additive into legacy-untouched cells (legacy_bake no longer programs L6
    attn Q/K/V/O — those came from ``make_layer6_attn_bake_op`` at 998.5).

    Gated by ``enable_conversational_io``: bake_fn is a no-op when False.
    """
    def bake(model, dim_positions, S):
        del S
        if not enable_conversational_io:
            return
        attn = model.blocks[6].attn
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes[4] = 5.0  # PRTF relay
            attn.alibi_slopes[5] = 5.0  # READ relay
        HD = attn.W_q.shape[0] // attn.num_heads
        Primitives.generate_attention_heads(
            attn,
            _convo_io_relay_head_specs(_as_setdim_proxy(dim_positions)),
            HD,
        )

    return Operation(
        name="convo_io_relay_heads",
        phase=999.5,
        reads=set(),
        writes=set(),
        kind="model",
        declarative_bake_fn=bake,
        compiler_ir_factory=(
            _convo_io_relay_heads_ir
            if enable_conversational_io else None
        ),
        declarative_authority="spec_generated",
        migrated=True,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#printing-and-reading-input",
    )


def _convo_io_relay_heads_ir(dim_positions, HD) -> CompilerIR:
    del HD
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.extend(_convo_io_relay_head_specs(proxy))
    return ir


def _convo_io_relay_head_specs(BD) -> tuple[DeclarativeAttentionHeadSpec, ...]:
    L = 50.0
    return (
        DeclarativeAttentionHeadSpec(
            head_idx=4,
            q=(
                AP(0, BD.NEXT_SE, L),
                AP(0, BD.MARK_AX, -L),
                AP(0, BD.ACTIVE_OPCODE_PRTF, L * 1.5),
            ),
            k=(AP(0, BD.MARK_AX, L),),
            v=(AP(37, BD.IO_IS_PRTF, 1.0),),
            o=(AO(BD.CMP + 5, 37, 1.0),),
        ),
        DeclarativeAttentionHeadSpec(
            head_idx=5,
            q=(
                AP(0, BD.NEXT_SE, L),
                AP(0, BD.MARK_AX, -L),
            ),
            k=(AP(0, BD.MARK_AX, L),),
            v=(AP(1, BD.IO_IS_READ, 1.0),),
            o=(AO(BD.CMP + 6, 1, 1.0),),
        ),
    )


def make_tool_call_detection_op(enable_tool_calling: bool = False) -> Operation:
    """Bake L6 FFN tool-call detection unit (CMP[2] AND NEXT_SE -> NEXT_TOOL_CALL).

    Originally an inline call in `set_vm_weights` (inside `if enable_tool_calling:`):
        `_set_tool_call_detection(ffn6, S, BD)`

    Operates on `model.blocks[6].ffn` unit 1300. Modeled as `kind="model"` so
    the bake_fn can resolve ffn6 from the model handle, matching
    `make_io_putchar_routing_op`.

    Phase 998.8: runs after the L6 FFN main bakes (phase 6.5) and before
    legacy_bake (999) so unit 1300 survives `_right_size_ffns`.

    When `enable_tool_calling=False`, the bake_fn is a no-op.

    Migrated 2026-06-01 (Phase 6 Wave 3L): the imperative helper
    ``_set_tool_call_detection`` is replaced by an ``FFNRule``-driven
    lowerer (``_lower_tool_call_detection_ir``) — single
    ``FFNRule.constant_write`` lowered through
    ``Primitives.lower_ffn_rules`` at pinned unit 1300. Byte-identical to
    the legacy helper (validated via
    ``compare_symbolic_to_lowered_ffn``).
    """
    def bake(model, dim_positions, S):
        if not enable_tool_calling:
            return
        _lower_tool_call_detection_ir(
            model.blocks[6].ffn, S, _as_setdim_proxy(dim_positions),
        )

    return Operation(
        name="tool_call_detection",
        reads=set(),
        writes=set(),
        kind="model",
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        phase=998.8,
        migrated=True,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#tool-use-mode",
    )


_TOOL_CALL_DETECTION_START_UNIT = 1300


def _tool_call_detection_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for ``_set_tool_call_detection`` (L6 FFN unit 1300).

    Single ``constant_write`` unit: ``CMP[2] + NEXT_SE >= 3.0`` fires and
    emits ``NEXT_TOOL_CALL`` (+2/S) while suppressing ``NEXT_SE`` (-2/S).
    ``b_gate=1.0`` is supplied via :class:`FFNRule`'s default
    ``gate_bias`` for ``constant_write``.
    """
    write_scale = 2.0 / S
    return (
        FFNRule.constant_write(
            name="tool_call_detection",
            conditions=(
                ("CMP+2", 1.0),
                ("NEXT_SE", 1.0),
            ),
            threshold=3.0,
            writes=(
                ("NEXT_TOOL_CALL", write_scale),
                ("NEXT_SE", -write_scale),
            ),
        ),
    )


def _lower_tool_call_detection_ir(ffn, S: float, BD) -> int:
    rules = _tool_call_detection_rules(S)
    dim_positions = Primitives.dim_positions_from_bd(
        BD,
        Primitives.ffn_rule_dim_names(rules),
    )
    return Primitives.lower_ffn_rules(
        ffn,
        rules,
        dim_positions,
        start_unit=_TOOL_CALL_DETECTION_START_UNIT,
        S=S,
    )


def make_convo_io_state_machine_op(enable_conversational_io: bool = False) -> Operation:
    """Bake ``_set_conversational_io_state_machine`` into ``model.blocks[6].ffn``.

    Originally an inline call in ``set_vm_weights`` under
    ``if enable_conversational_io:``:
        ``_set_conversational_io_state_machine(ffn6, S, BD)``

    Programs the L6 FFN state-machine entry: CMP[5]/CMP[6] AND NEXT_SE →
    emit NEXT_THINKING_END, suppress NEXT_SE, set IO_STATE = 1. Writes
    L6 FFN units 1400-1401 (separate from the routing-FFN units 0-1033
    and the function-call-weights units 1700-2158).

    Gated by ``enable_conversational_io``: bake_fn is a no-op when False.
    Pinned to ``layer_idx=6`` via ``kind="block"`` so the bake hits
    ``model.blocks[6].ffn``. Block ops run BEFORE right_size_ffns
    (phase 1200), so the FFN units we write survive rightsize trimming.

    Phase 6.6 places this AFTER ``layer6_routing_ffn`` (phase 6.5) so the
    convo-io extension units are appended after the base routing FFN, and
    BEFORE the model-level FFN bakes at phase 998 (function_call_weights,
    io_putchar_routing) which write higher-numbered units.
    """
    def bake(block, dim_positions, S):
        if not enable_conversational_io:
            return
        _lower_convo_io_state_machine_ir(
            block.ffn,
            S,
            _as_setdim_proxy(dim_positions),
        )

    return Operation(
        name="convo_io_state_machine",
        phase=6.6,
        reads=set(),
        writes=set(),
        kind="block",
        # Phase 8.A.4: dropped ``layer_idx=6`` in favour of
        # ``target_op_name``. Binds to whichever layer the compiler
        # placed ``layer6_attn`` (an L6 attn op anchor).
        target_op_name="layer6_attn",
        migrated=True,
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        compiler_ir=make_convo_io_state_machine_ir(),
        ffn_units_used=1402 if enable_conversational_io else None,
        # B12 backfill: docstring above names ``layer6_routing_ffn`` (phase
        # 6.5, same L6 FFN) as the required predecessor — the base routing
        # FFN must be in place before this extension appends units
        # 1400-1401. Encoded as a B10 op-name reference so the dynamic
        # scheduler honours the dep edge despite empty reads/writes.
        requires={"after": "layer6_routing_ffn"},
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#printing-and-reading-input",
    )


def _convo_io_state_machine_rules(S: float) -> tuple[FFNRule, ...]:
    write_scale = 2.0 / S
    rules = []
    for flag_name, flag_dim in (
        ("prtf", "CMP+5"),
        ("read", "CMP+6"),
    ):
        rules.append(FFNRule.gated_write(
            name=f"convo_io_state_machine_{flag_name}",
            conditions=((flag_dim, 1.0),),
            threshold=0.5,
            gate="NEXT_SE",
            gate_weight=10.0,
            writes=(
                ("NEXT_THINKING_END", write_scale),
                ("NEXT_SE", -write_scale),
                ("IO_STATE", write_scale),
            ),
        ))
    return tuple(rules)


def _lower_convo_io_state_machine_ir(ffn, S: float, BD) -> int:
    rules = _convo_io_state_machine_rules(S)
    dim_positions = Primitives.dim_positions_from_bd(
        BD,
        Primitives.ffn_rule_dim_names(rules),
    )
    return Primitives.lower_ffn_rules(
        ffn,
        rules,
        dim_positions,
        start_unit=1400,
        S=S,
    )


def make_convo_io_state_machine_ir(S: float = 100.0) -> CompilerIR:
    """Declarative CompilerIR for the L6 convo-IO state machine band.

    Mirrors ``_lower_convo_io_state_machine_ir`` so the verifier, scope
    checker, and dominance auditor can read the rule set directly. The
    actual bake stays in ``make_convo_io_state_machine_op`` because the
    band is pinned to L6 FFN units 1400..1401, which the generic
    ``_dispatch_operation_ir`` (start_unit=0) cannot replicate.
    """

    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_convo_io_state_machine_rules(S))
    return ir


# ---------------------------------------------------------------------------
# Conversational I/O tail bakes (L10 + L15)
# ---------------------------------------------------------------------------
#
# These ops migrate the two inline ``_set_null_terminator_detection`` and
# ``_set_conversational_io_output_routing`` calls out of ``set_vm_weights``.
# Both are gated by ``enable_conversational_io`` — when the flag is off the
# bake_fn is a no-op so the op is safe to register unconditionally. The flag
# is plumbed through ``all_core_ops`` (and from ``compile_full_vm``).
#
# L10 null terminator: writes ffn10 units starting at 1864 (above the
# ~1854 units used by ``_set_layer10_alu``). Pinned to layer_idx=10 via
# kind="block". The bake is skipped in efficient mode because ffn10 is
# replaced with an ``ALUAndOrXor`` module (a ``PureNeuralALU`` subclass
# without the ``W_up``/``W_gate``/``W_down``/``b_up`` PureFFN interface
# the helper expects). The lookup-branch nesting in the legacy call was
# necessary, not incidental.
#
# L15 output routing: writes ffn15 units starting at 1200 (well above the
# ~32 units used by ``_set_nibble_copy_ffn``). Pinned to layer_idx=15 via
# kind="block". ffn15 is a ``PureFFN`` in both alu_modes, so this op
# always bakes when convo-io is enabled.

def make_null_terminator_detection_op(
    enable_conversational_io: bool = False,
    alu_mode: str = "lookup",
) -> Operation:
    """L10 FFN: null-terminator detection for conversational I/O.

    Originally an inline call in ``set_vm_weights`` (nested inside the
    ``alu_mode == 'lookup'`` branch under ``if enable_conversational_io:``):
        ``_set_null_terminator_detection(ffn10, S, BD)``

    The helper writes ``ffn10`` units starting at 1864 (above the ~1854
    units used by ``_set_layer10_alu``). Detects ``OUTPUT_BYTE == 0`` while
    ``IO_IN_OUTPUT_MODE`` is active and sets ``IO_OUTPUT_COMPLETE`` +
    clears ``IO_IN_OUTPUT_MODE`` + emits ``NEXT_THINKING_START``.

    Pinned to ``layer_idx=10`` via ``kind="block"``. Runs in the block-op
    dispatch loop BEFORE legacy_bake (phase=999), so ``_set_layer10_alu``
    has not yet populated units 0-1853 when this op writes to units 1864+;
    the two ranges don't overlap so the order doesn't matter. The phase
    (10.6) only matters relative to other block ops at layer_idx=10 (none
    today). All written units survive ``right_size_ffns`` (phase=1200).

    No-op when ``enable_conversational_io=False`` or when
    ``alu_mode != 'lookup'`` (in efficient mode ffn10 is an
    ``ALUAndOrXor`` and lacks the PureFFN ``W_*``/``b_*`` interface the
    helper expects).
    """
    if enable_conversational_io and alu_mode == "lookup":
        def bake(block, dim_positions, S):
            from ...vm_step import _set_null_terminator_detection
            _set_null_terminator_detection(
                block.ffn, S, _as_setdim_proxy(dim_positions)
            )
    else:
        def bake(block, dim_positions, S):
            return

    return Operation(
        name="null_terminator_detection",
        phase=10.6,
        reads={"OUTPUT_BYTE_LO", "OUTPUT_BYTE_HI", "IO_IN_OUTPUT_MODE"},
        writes={"IO_OUTPUT_COMPLETE", "IO_IN_OUTPUT_MODE",
                "NEXT_THINKING_START"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake
        if not (enable_conversational_io and alu_mode == "lookup")
        else None,
        # Phase 8.G.6: drop ``layer_idx=10`` literal; bind to the L10
        # attn anchor ``layer10_carry_relay`` so the block op resolves to
        # whichever layer the compiler picks for the L10 carry-relay attn.
        target_op_name="layer10_carry_relay",
        migrated=True,
        # When ``enable_conversational_io=True`` AND ``alu_mode='lookup'``,
        # the helper writes L10 FFN unit 1864 (single unit). That sits
        # above ``layer10_alu``'s 0-1845 cluster, so the L10 FFN must be
        # sized to 1865 hidden units in this mode. The compiler takes the
        # per-layer max across this op + ``layer10_alu`` (1846), so the
        # net dynamic width becomes 1865 here.
        ffn_units_used=(
            1865 if (enable_conversational_io and alu_mode == "lookup")
            else None
        ),
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def make_convo_io_step_resume_op(
    enable_conversational_io: bool = False,
    enable: bool = False,
) -> Operation:
    """L3 FFN: V18 Phase 1 step-resumption bake (3a in the V18 plan).

    Wires the L3 FFN unit 1035 that fires on ``LAST_WAS_THINKING_START``:
    sets ``NEXT_PC = 1`` (so the head emits ``Token.REG_PC`` for the next
    token, starting a fresh VM step) and clears ``IO_STATE`` /
    ``IO_IN_OUTPUT_MODE``. Closes the "missing transition" in
    ``_set_conversational_io_state_machine``'s state-machine docstring
    (V18_CONVO_IO_NEURAL_PLAN.md §3a).

    Double-gated: both ``enable_conversational_io`` AND ``enable`` must be
    True for the bake to fire. The two-gate design lets this op stay
    registered in ``all_core_ops()`` (so the dep graph is stable) and
    safely defaults to a no-op for the Phase 1 landing. Flip ``enable=True``
    once the parity test in
    ``tests/test_v18_convo_io_neural_bakes.py`` passes end-to-end.

    Pinned to ``layer_idx=3`` via ``kind="block"`` so the bake hits
    ``model.blocks[3].ffn`` (matching the legacy direct-FFN access for
    convo-IO state init). Phase 3.2 places this AFTER
    ``layer3_convo_io_state_init`` (phase 3.1) — that op writes unit 1034,
    this one writes unit 1035, so the explicit phase ordering guarantees
    no collision regardless of dispatch order tweaks.
    """
    def bake(block, dim_positions, S):
        if not (enable_conversational_io and enable):
            return
        _lower_convo_io_step_resume_ir(
            block.ffn,
            S,
            _as_setdim_proxy(dim_positions),
        )

    return Operation(
        name="convo_io_step_resume",
        phase=3.2,
        reads=set(),
        writes=set(),
        kind="block",
        # Phase 8.A.4: dropped ``layer_idx=3`` in favour of
        # ``target_op_name``. Binds to whichever layer the compiler
        # placed ``layer3_carry_forward_attn`` (the L3 attn op).
        target_op_name="layer3_carry_forward_attn",
        declarative_bake_fn=bake,
        compiler_ir=_convo_io_step_resume_ir(),
        declarative_authority="spec_generated",
        migrated=True,
        ffn_units_used=1036 if (enable_conversational_io and enable) else None,
        # B12 backfill: docstring above names ``layer3_convo_io_state_init``
        # (phase 3.1, same L3 FFN) as the required predecessor — that op
        # writes unit 1034, this one writes unit 1035, so the upstream
        # state-init unit must be in place first. Encoded as a B10
        # op-name reference so the dynamic scheduler honours the dep
        # edge despite empty reads/writes.
        requires={"after": "layer3_convo_io_state_init"},
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#printing-and-reading-input",
    )


def _convo_io_step_resume_rules(S: float) -> tuple[FFNRule, ...]:
    write_scale = 2.0 / S
    return (
        FFNRule.constant_write(
            name="convo_io_step_resume",
            conditions=(("LAST_WAS_THINKING_START", 1.0),),
            threshold=0.5,
            writes=(
                ("NEXT_PC", write_scale),
                ("IO_STATE", -write_scale),
                ("IO_IN_OUTPUT_MODE", -write_scale),
            ),
            scope="LAST_WAS_THINKING_START",
        ),
    )


def _convo_io_step_resume_ir(S: float = 100.0) -> CompilerIR:
    """Build a :class:`CompilerIR` exposing the convo-IO step-resume rule.

    Pinned to :attr:`Operation.compiler_ir` so the declarative verifier
    sees the single ``convo_io_step_resume`` rule even though the bake
    body lowers it at start_unit=1035 (above the L3 main rule range).
    The IR-level lowering uses start_unit=0; the actual bake body uses
    the production offset via :func:`_lower_convo_io_step_resume_ir`.
    """
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_convo_io_step_resume_rules(S))
    return ir


def _lower_convo_io_step_resume_ir(ffn, S: float, BD) -> int:
    rules = _convo_io_step_resume_rules(S)
    dim_positions = Primitives.dim_positions_from_bd(
        BD,
        Primitives.ffn_rule_dim_names(rules),
    )
    return Primitives.lower_ffn_rules(
        ffn,
        rules,
        dim_positions,
        start_unit=1035,
        S=S,
    )


def make_convo_io_pc_sp_latch_op(
    enable_conversational_io: bool = False,
    enable: bool = False,
) -> Operation:
    """L6 FFN: V18 Phase 1 PC/SP latch replay band (3b in the V18 plan).

    Wires L6 FFN units 1402-1465 (64 units total) that, on the
    LAST_WAS_THINKING_START edge, drive ``OUTPUT_LO/HI`` from the
    staged PC and SP nibbles (so the resumed step's REG_PC / REG_SP
    value bytes carry the post-PRTF advanced PC and popped SP rather
    than relying on the runner's ``_inject_synthetic_step`` to supply
    them).

    Unit allocation is non-overlapping with existing baked ranges:
      - 1400-1401: existing ``_set_conversational_io_state_machine``
      - 1402-1465: this op
      - 1500-1532: V9 PUTCHAR routing (PHASE_6_PUTCHAR_BAKE_SPEC.md)
      - 1700-2158: function_call_weights
      - 2200+:     binary_pop_sp_increment

    Double-gated: both ``enable_conversational_io`` AND ``enable`` must be
    True for the bake to fire. Defaults to a no-op for the Phase 1 landing
    — flip ``enable=True`` once the companion *capture*-side attention
    bake (Phase 1b) lands and the parity test passes end-to-end.

    Pinned to ``layer_idx=6`` via ``kind="block"`` so the bake hits
    ``model.blocks[6].ffn``. Phase 6.7 places this AFTER
    ``convo_io_state_machine`` (phase 6.6) so the state-machine units
    (1400-1401) are in place before this op writes units 1402+. Both
    run BEFORE ``right_size_ffns`` (phase 1200) so all units survive.
    """
    def bake(block, dim_positions, S):
        if not (enable_conversational_io and enable):
            return
        _lower_convo_io_pc_sp_latch_ir(
            block.ffn,
            S,
            _as_setdim_proxy(dim_positions),
        )

    return Operation(
        name="convo_io_pc_sp_latch",
        phase=6.7,
        reads=set(),
        writes=set(),
        kind="block",
        # Phase 8.A.4: dropped ``layer_idx=6`` in favour of
        # ``target_op_name``. Binds to whichever layer the compiler
        # placed ``layer6_attn`` (an L6 attn op anchor).
        target_op_name="layer6_attn",
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        compiler_ir=make_convo_io_pc_sp_latch_ir(),
        migrated=True,
        ffn_units_used=1466 if (enable_conversational_io and enable) else None,
        # B12 backfill: docstring above names ``convo_io_state_machine``
        # (phase 6.6, same L6 FFN) as the required predecessor — its
        # state-machine units 1400-1401 must be in place before this op
        # writes units 1402+. Encoded as a B10 op-name reference so the
        # dynamic scheduler honours the dep edge despite empty
        # reads/writes.
        requires={"after": "convo_io_state_machine"},
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#printing-and-reading-input",
    )


def _convo_io_pc_sp_latch_rules(S: float) -> tuple[FFNRule, ...]:
    write_scale = 2.0 / S
    rules = []
    for src_lo, src_hi in (
        ("POST_PRTF_PC_LO", "POST_PRTF_PC_HI"),
        ("POST_PRTF_SP_LO", "POST_PRTF_SP_HI"),
    ):
        for source_base, output_base in (
            (src_lo, "OUTPUT_LO"),
            (src_hi, "OUTPUT_HI_THIS_STEP"),
        ):
            for k in range(16):
                rules.append(FFNRule.gated_write(
                    name=f"convo_io_latch_{source_base.lower()}_{k}",
                    conditions=(("LAST_WAS_THINKING_START", 1.0),),
                    threshold=0.5,
                    gate=f"{source_base}+{k}",
                    writes=((f"{output_base}+{k}", write_scale),),
                ))
    return tuple(rules)


def _lower_convo_io_pc_sp_latch_ir(ffn, S: float, BD) -> int:
    rules = _convo_io_pc_sp_latch_rules(S)
    dim_positions = Primitives.dim_positions_from_bd(
        BD,
        Primitives.ffn_rule_dim_names(rules),
    )
    return Primitives.lower_ffn_rules(
        ffn,
        rules,
        dim_positions,
        start_unit=1402,
        S=S,
    )


def make_convo_io_pc_sp_latch_ir(S: float = 100.0) -> CompilerIR:
    """Declarative CompilerIR for the L6 convo-IO PC/SP latch band.

    Mirrors ``_lower_convo_io_pc_sp_latch_ir`` so the verifier, scope
    checker, and dominance auditor can read the rule set directly. The
    actual bake stays in ``make_convo_io_pc_sp_latch_op`` because the band
    is pinned to L6 FFN units 1402..1465, which the generic
    ``_dispatch_operation_ir`` (start_unit=0) cannot replicate.
    """

    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_convo_io_pc_sp_latch_rules(S))
    return ir


def make_convo_io_prtf_capture_op(
    enable_conversational_io: bool = False,
    enable: bool = False,
) -> Operation:
    """L7 FFN: V18 Phase 1b PRTF AX-marker capture bake (3c in the V18 plan).

    Wires L7 FFN units 800-863 (64 units total) that, at the PRTF AX
    marker (``ACTIVE_OPCODE_PRTF`` AND ``MARK_AX``), decompose the current
    step's PC and SP byte-0 nibbles and stage them into the dedicated
    ``POST_PRTF_PC_LO/HI`` and ``POST_PRTF_SP_LO/HI`` cache dims that the
    3b replay band (``convo_io_pc_sp_latch`` at L6 FFN units 1402-1465)
    reads at the resumed step's REG_PC / REG_SP value-byte positions.

    Sources at the AX marker after L4:
      - PC byte 0 nibbles: ``EMBED_LO/HI`` (set by L4 head 0 ``_set_layer4_pc_relay``).
      - SP byte 0 nibbles: ``ADDR_B0_HI`` lo / ``ADDR_B1_HI`` hi (set by L4 heads 2-3
        ``make_layer4_sp_to_addr_key_op``).

    The cache dims alias dead slots at the PRTF AX marker:
      - POST_PRTF_PC_LO/HI alias AX_FULL_LO/HI (PRTF never PSHes AX).
      - POST_PRTF_SP_LO/HI alias AX_CARRY_LO/HI (PRTF is not an ALU op).

    Double-gated: both ``enable_conversational_io`` AND ``enable`` must
    be True for the bake to fire. Defaults to a no-op for the Phase 1b
    landing — flip ``enable=True`` once the parity test in
    ``tests/test_v18_convo_io_neural_bakes.py`` passes end-to-end and
    Phase 2 (deletion of the V18 handler block at ``run_vm.py:534-583``)
    is ready to land.

    Pinned to ``layer_idx=7`` via ``kind="block"`` so the bake hits
    ``model.blocks[7].ffn``. Phase 7.6 places this AFTER the L7 FFN main
    bakes (phase 7.x) so the unit writes layer cleanly. Runs BEFORE
    ``right_size_ffns`` (phase 1200) so all units survive.

    Unit allocation (kept disjoint from existing baked ranges):
      - 0-99:    typical L7 FFN main bake range
      - 800-863: this op (capture-side band)
    """
    def bake(block, dim_positions, S):
        if not (enable_conversational_io and enable):
            return
        _lower_convo_io_prtf_capture_ir(
            block.ffn,
            S,
            _as_setdim_proxy(dim_positions),
        )

    return Operation(
        name="convo_io_prtf_capture",
        phase=7.6,
        reads=set(),
        writes=set(),
        kind="block",
        # Phase 8.A.4: dropped ``layer_idx=7`` in favour of
        # ``target_op_name``. Binds to whichever layer the compiler
        # placed ``nibble_copy_ffn`` (an L7 FFN op anchor).
        target_op_name="nibble_copy_ffn",
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        migrated=True,
        ffn_units_used=864 if (enable_conversational_io and enable) else None,
        # B12 backfill: docstring above names the L7 FFN main bakes as
        # the required predecessors so the capture-side units 800-863
        # layer cleanly on top. ``layer7_operand_gather`` is the L7
        # block op that owns the main-band weights; pin after it via a
        # B10 op-name reference so the dynamic scheduler honours the
        # dep edge despite empty reads/writes.
        requires={"after": "layer7_operand_gather"},
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#printing-and-reading-input",
    )


def _convo_io_prtf_capture_rules(S: float) -> tuple[FFNRule, ...]:
    write_scale = 2.0 / S
    rules = []
    for src_lo, src_hi, dst_lo, dst_hi in (
        ("EMBED_LO", "EMBED_HI", "POST_PRTF_PC_LO", "POST_PRTF_PC_HI"),
        ("ADDR_B0_HI", "ADDR_B1_HI", "POST_PRTF_SP_LO", "POST_PRTF_SP_HI"),
    ):
        for source_base, dest_base in (
            (src_lo, dst_lo),
            (src_hi, dst_hi),
        ):
            for k in range(16):
                rules.append(FFNRule.gated_write(
                    name=f"convo_io_capture_{dest_base.lower()}_{k}",
                    conditions=(
                        ("ACTIVE_OPCODE_PRTF", 1.0),
                        ("MARK_AX", 1.0),
                    ),
                    threshold=1.5,
                    gate=f"{source_base}+{k}",
                    writes=((f"{dest_base}+{k}", write_scale),),
                ))
    return tuple(rules)


def _lower_convo_io_prtf_capture_ir(ffn, S: float, BD) -> int:
    rules = _convo_io_prtf_capture_rules(S)
    dim_positions = Primitives.dim_positions_from_bd(
        BD,
        Primitives.ffn_rule_dim_names(rules),
    )
    return Primitives.lower_ffn_rules(
        ffn,
        rules,
        dim_positions,
        start_unit=800,
        S=S,
    )


def make_convo_io_prtf_transport_op(
    enable_conversational_io: bool = False,
    enable: bool = False,
) -> Operation:
    """L4 attn head 4: V18 Phase 1c PRTF AX-marker transport bake (3d in plan).

    Closes the variable-length transport gap (V18_CONVO_IO_NEURAL_PLAN.md §3,
    Phase 1c). Pairs with the capture bake (3c, ``convo_io_prtf_capture`` at
    L7 FFN units 800-863) and the replay bake (3b, ``convo_io_pc_sp_latch`` at
    L6 FFN units 1402-1465). The capture stages PC/SP nibbles into
    POST_PRTF_PC/SP cache dims at the PRTF AX marker; this op's attention
    head reads those nibbles back at the post-THINKING_START position so the
    replay band has them available at the same position.

    Why an attention head
    ---------------------
    POST_PRTF_PC/SP are residual-stream dims, attached to a single token
    position. Across the variable-length output-byte interlude
    (THINKING_END → N output bytes → THINKING_START → next step's REG_PC)
    the residual does NOT propagate from the capture position to the replay
    position. An attention head with K-gating on (ACTIVE_OPCODE_PRTF AND
    MARK_AX) and ALiBi recency bias finds the most recent PRTF AX marker
    and copies the captured nibbles forward.

    Head selection
    --------------
    L4 attn has heads 0 (PC relay), 2-3 (SP-to-ADDR_KEY) baked; heads 1,
    4, 5, 6, 7 are free. We use head 4 here. Phase 4.6 places this AFTER
    ``layer4_pc_relay`` (phase 4) and ``layer4_sp_to_addr_key`` (phase 4.5)
    so the per-block fill_(0.5) of alibi slopes has settled before we
    override slope[4] = 0.1.

    Double-gated: both ``enable_conversational_io`` AND ``enable`` must be
    True for the bake to fire. Defaults to a no-op for the Phase 1c
    landing — flip ``enable=True`` once the end-to-end neural convo-IO
    loop is validated together with the capture (3c) and replay (3b)
    bakes.

    Pinned to ``layer_idx=4`` via ``kind="block"`` so the bake hits
    ``model.blocks[4].attn``.

    No collision with existing L4 attn writes:
      - Head 0 writes EMBED_LO/HI at AX marker (PC relay).
      - Heads 2-3 write ADDR_B0_HI / ADDR_B1_HI / ADDR_B2_HI at AX marker
        (SP→ADDR_KEY staging).
      - Head 4 (this op) writes POST_PRTF_PC_LO/HI (= AX_FULL_LO/HI, dims
        471/487) and POST_PRTF_SP_LO/HI (= AX_CARRY_LO/HI, dims 328/344)
        at the post-THINKING_START position. None of the other L4 attn
        heads write to those dims at the post-THINKING_START position;
        the dim aliases (AX_FULL/AX_CARRY) are dead at this position
        because the post-THINKING_START token is not a STACK0 marker (no
        AX_FULL write) and not an ALU op (no AX_CARRY write).
    """
    def bake(block, dim_positions, S):
        del S
        if not (enable_conversational_io and enable):
            return
        attn = block.attn
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            # Override head 4's slope to be shallow so the head can reach
            # back across the variable-length output-byte interlude
            # (typically ~38..78 tokens between resume and PRTF AX marker).
            # See _set_convo_io_prtf_transport docstring for score-budget
            # analysis.
            attn.alibi_slopes[4] = 0.1
        HD = attn.W_q.shape[0] // attn.num_heads
        Primitives.generate_attention_head(
            attn,
            _convo_io_prtf_transport_spec(_as_setdim_proxy(dim_positions)),
            HD,
        )

    return Operation(
        name="convo_io_prtf_transport",
        phase=4.6,
        reads=set(),
        writes=set(),
        kind="block",
        # Phase 8.A.4: dropped ``layer_idx=4`` in favour of
        # ``target_op_name``. Binds to whichever layer the compiler
        # placed ``_layer3_ffn_dep_anchor`` (the L4 ffn dep anchor;
        # the dep graph places it at the layer the L3 ffn extends to).
        target_op_name="_layer3_ffn_dep_anchor",
        declarative_bake_fn=bake,
        compiler_ir_factory=(
            _convo_io_prtf_transport_ir
            if (enable_conversational_io and enable) else None
        ),
        declarative_authority="spec_generated",
        migrated=True,
        # B12 backfill: docstring above names ``layer4_pc_relay`` (phase
        # 4) and ``layer4_sp_to_addr_key`` (phase 4.5) as the required
        # predecessors so the per-block ``fill_(0.5)`` of alibi slopes
        # has settled before this op overrides ``slope[4] = 0.1``. Pin
        # via a B10 op-name reference so the dynamic scheduler honours
        # the dep edge despite empty reads/writes.
        #
        # Phase 7.A.1: ``layer4_sp_to_addr_key`` is itself a kind="block"
        # op pinned to layer_idx=4 (phase 4.5), so this op (layer_idx=4,
        # phase 4.6) really wants to co-place at L4 with a per-block
        # phase tiebreaker -- NOT push to L5. Use
        # ``requires["same_layer_as"]`` to express the co-placement
        # constraint to analyze_scheduler.py and LayerCompiler equally;
        # the per-block phase ordering (4.5 < 4.6) handles within-block
        # sequencing of alibi-slope writes.
        #
        # Phase 7.A.2 backfill: declare the strict-after edges for both
        # head 0/1 (``layer4_pc_relay``) and head 2/3
        # (``layer4_sp_to_addr_key``) as named in the docstring above.
        # The two predecessors share the L4 attention-head allocator with
        # this op (this is head 4 in the same builder), and the
        # ``alibi_slopes[4] = 0.1`` override here MUST run after the
        # per-block ``alibi_slopes.fill_(0.5)`` plus the relay's slope
        # writes -- the ordering is currently maintained by the
        # per-block phase tiebreaker (phase 4 < 4.5 < 4.6) but is
        # otherwise invisible to the dim-only analyzer because this op
        # has empty reads/writes. ``layer4_pc_relay`` is itself currently
        # a cycle member; the ``same_layer_as`` edge previously already
        # pulled this op into the same SCC transitively via
        # ``layer4_sp_to_addr_key``, so declaring the explicit
        # ``after`` edges is documentation-only for cycle membership and
        # makes the dep-graph honest about the load-bearing order.
        requires={
            "after": ["layer4_pc_relay", "layer4_sp_to_addr_key"],
            "same_layer_as": "layer4_sp_to_addr_key",
        },
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#printing-and-reading-input",
    )


def _convo_io_prtf_transport_ir(dim_positions, HD) -> CompilerIR:
    del HD
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.append(_convo_io_prtf_transport_spec(proxy))
    return ir


def _convo_io_prtf_transport_spec(BD) -> DeclarativeAttentionHeadSpec:
    L = 50.0
    v = []
    o = []
    for slot_offset, dim_base, count in (
        (1, BD.POST_PRTF_PC_LO, 16),
        (17, BD.POST_PRTF_PC_HI, 16),
        (33, BD.POST_PRTF_SP_LO, 16),
        (49, BD.POST_PRTF_SP_HI, 15),
    ):
        for k in range(count):
            v.append(AP(slot_offset + k, dim_base + k, 1.0))
            o.append(AO(dim_base + k, slot_offset + k, 1.0))
    return DeclarativeAttentionHeadSpec(
        head_idx=4,
        q=(
            AP(0, BD.LAST_WAS_THINKING_START, L),
            AP(0, BD.CONST, -L * 0.5),
        ),
        k=(
            AP(0, BD.ACTIVE_OPCODE_PRTF, L),
            AP(0, BD.MARK_AX, L),
            AP(0, BD.CONST, -L),
        ),
        v=tuple(v),
        o=tuple(o),
    )


_CONVERSATIONAL_IO_OUTPUT_ROUTING_START_UNIT = 1200


def _conversational_io_output_routing_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for ``_set_conversational_io_output_routing``.

    32 gated-write units (16 ``OUTPUT_LO`` + 16 ``OUTPUT_HI`` nibbles).
    Each unit gates on ``IO_IN_OUTPUT_MODE`` (threshold 0.5) and copies one
    ``OUTPUT_BYTE_LO/HI+k`` nibble through to the matching ``OUTPUT_LO/HI+k``
    cell. The imperative form (``setup_helpers_l15``) writes:

        W_up[unit, IO_IN_OUTPUT_MODE]      = S
        b_up[unit]                         = -S * 0.5
        W_gate[unit, OUTPUT_BYTE_LO + k]   = 1.0
        W_down[OUTPUT_LO + k, unit]        = 2.0 / S

    which is byte-identical to ``FFNRule.gated_write`` with
    ``conditions=(("IO_IN_OUTPUT_MODE", 1.0),)``, ``threshold=0.5``,
    ``gate=f"OUTPUT_BYTE_LO+{k}"``, ``gate_weight=1.0``, ``gate_bias=0.0``,
    and ``writes=((f"OUTPUT_LO+{k}", 2.0 / S),)``: ``lower_ffn`` accumulates
    ``W_up += S * 1.0``, ``b_up = -S * 0.5``, ``W_gate += 1.0``,
    ``b_gate = 0.0``, and ``W_down += (2.0 / S) * 1.0``.

    The LO and HI bands are interleaved per the legacy unit ordering
    (lo[0], hi[0], lo[1], hi[1], ...) so the lowered band stays at
    units 1200..1231 byte-for-byte against the imperative helper.
    """
    rules: list[FFNRule] = []
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"convo_io_output_routing_lo_{k}",
            conditions=(("IO_IN_OUTPUT_MODE", 1.0),),
            threshold=0.5,
            gate=f"OUTPUT_BYTE_LO+{k}",
            gate_weight=1.0,
            gate_bias=0.0,
            writes=((f"OUTPUT_LO+{k}", 2.0 / S),),
        ))
        rules.append(FFNRule.gated_write(
            name=f"convo_io_output_routing_hi_{k}",
            conditions=(("IO_IN_OUTPUT_MODE", 1.0),),
            threshold=0.5,
            gate=f"OUTPUT_BYTE_HI+{k}",
            gate_weight=1.0,
            gate_bias=0.0,
            writes=((f"OUTPUT_HI_THIS_STEP+{k}", 2.0 / S),),
        ))
    return tuple(rules)


def _conversational_io_output_routing_ir(S: float = 100.0) -> CompilerIR:
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_conversational_io_output_routing_rules(S))
    return ir


def _lower_conversational_io_output_routing_ir(ffn, S: float, BD) -> int:
    """Lower the convo-IO output-routing FFN rules at pinned unit 1200."""

    rules = _conversational_io_output_routing_rules(S)
    dim_positions = Primitives.dim_positions_from_bd(
        BD,
        Primitives.ffn_rule_dim_names(rules),
    )
    return Primitives.lower_ffn_rules(
        ffn,
        rules,
        dim_positions,
        start_unit=_CONVERSATIONAL_IO_OUTPUT_ROUTING_START_UNIT,
        S=S,
    )


def make_conversational_io_output_routing_op(
    enable_conversational_io: bool = False,
) -> Operation:
    """L15 FFN: route ``OUTPUT_BYTE_LO/HI`` -> ``OUTPUT_LO/HI`` in output mode.

    Originally an inline call in ``set_vm_weights`` (gated by
    ``if enable_conversational_io:`` at the top level after the
    ``alu_mode`` branches):
        ``_set_conversational_io_output_routing(ffn15, S, BD)``

    The helper writes ``ffn15`` units starting at 1200 (well above the
    ~32 units used by ``_set_nibble_copy_ffn``), copying each
    ``OUTPUT_BYTE`` nibble to the corresponding ``OUTPUT`` nibble when
    ``IO_IN_OUTPUT_MODE`` is set. This emits the fetched format-string
    byte through the output head.

    Pinned to ``layer_idx=15`` via ``kind="block"``. Phase=15.1 so it
    runs AFTER ``layer15_nibble_copy`` (phase=15, also layer_idx=15) in
    the block-op dispatch order. Both run BEFORE legacy_bake (phase=999)
    and BEFORE ``right_size_ffns`` (phase=1200) prunes dead units. ffn15
    is a ``PureFFN`` in both alu_modes, so this bake is alu_mode-agnostic.

    No-op when ``enable_conversational_io=False``.

    Phase 8.C migration: the 32-unit imperative ``_set_conversational_io_
    output_routing`` body is replaced by a CompilerIR built from
    :func:`_conversational_io_output_routing_rules` and lowered through
    ``Primitives.lower_ffn_rules`` at pinned unit 1200. Byte-identical to
    the legacy helper (one ``gated_write`` per OUTPUT_LO/HI nibble lane,
    same S-scaled weights).
    """
    if enable_conversational_io:
        def bake(block, dim_positions, S):
            _lower_conversational_io_output_routing_ir(
                block.ffn, S, _as_setdim_proxy(dim_positions)
            )
    else:
        def bake(block, dim_positions, S):
            return

    return Operation(
        name="conversational_io_output_routing",
        phase=15.1,
        reads={"IO_IN_OUTPUT_MODE", "OUTPUT_BYTE_LO", "OUTPUT_BYTE_HI"},
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake if not enable_conversational_io else None,
        # Phase 8.A.4: dropped ``layer_idx=15`` in favour of
        # ``target_op_name``. Binds to whichever layer the compiler
        # placed ``layer15_memory_lookup`` (the L15 attn op).
        target_op_name="layer15_memory_lookup",
        migrated=True,
        # ``_set_conversational_io_output_routing`` writes units 1200..1231
        # (16 LO + 16 HI = 32 units; see setup_helpers.py:1777). The op is
        # registered unconditionally, but the bake body is a no-op when
        # ``enable_conversational_io`` is False — so we only request the
        # 1232-unit allocation in that mode. In lookup mode the L15 FFN
        # falls back to ``layer15_nibble_copy``'s 40 units.
        ffn_units_used=1232 if enable_conversational_io else None,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#printing-and-reading-input",
    )
