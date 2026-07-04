"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ...attention_head_allocator import AttentionHeadAllocator
from ...dim_registry import dim_ref
from ...ffn_unit_allocator import FFNUnitAllocator
from ..building_blocks_dsl import byte_clear_rules, multi_way_and_rule
from ..ir import CompilerIR, FFNRule
from ..isa_semantics_dsl import (
    MarkerBroadcastBand,
    MarkerBroadcastSpec,
    RegisterDeltaSpec,
    RuntimeAddDelta,
    control_op,
    marker_broadcast,
)
from ..wide_alu_dsl import (
    amplified_nibble_adder_rules,
    nibble_alu_lane_rules,
    nibble_compare_lane_rules,
)
from ..layer_compiler import Operation
from ..primitives import AO, AP, DeclarativeAttentionHeadSpec, Primitives
from .shared import (  # noqa: F401
    _as_setdim_proxy,
    _empty_compiler_ir_factory,
)


# Wave B Cluster 3: operand dims that the Wave A step_end_operand_relay
# broadcasts from MARK_AX into MARK_SE_ONLY for the L9 CMP factory.
# After the relay the CMP factory's MARK_AX-gated rules can fire one
# row later (at STEP_END) byte-identically.
_CMP_RELAYED: tuple[str, ...] = (
    "CMP",
    "ALU_LO", "ALU_HI",
    "AX_CARRY_LO", "AX_CARRY_HI",
    "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
)

# Wave B Cluster 3 row 5: the LEV BP+8 shift gates on MARK_PC (not
# MARK_AX). Wave A relays the LEV opcode flag and the address-byte
# carriers into MARK_SE_ONLY; the helper extension swaps MARK_PC for
# MARK_SE_ONLY when called with from_marker="MARK_PC".
_BP_PLUS8_RELAYED: tuple[str, ...] = (
    "BP_FRAME_BYTE0", "BP_FRAME_BYTE1",
    "OP_ENT", "OP_LEV",
)


# === L9 attention-head layout (pinned indices) ======================
#
# Three migrated ops claim heads on the L9 attention block today:
#
#   * ``layer9_lev_addr_relay`` (phase=9.0)      -> head 0
#   * ``layer9_lev_bp_to_pc_relay`` (phase=9.1)  -> head 1
#   * ``layer9_alibi_mem_attn`` (phase=9.2)      -> head 2
#
# ``layer9_alibi_mem_attn`` is conditionally enabled at runtime via
# ``enable=False``; we still claim its head index in the allocator so
# the layout is structurally stable across builds (the bake itself
# early-returns when the gate is off, so no weights move either way).
# Pre-migration each call site picked its ``head_idx`` as a bare integer
# literal -- ``head_idx=0`` / ``head_idx=1`` in the relay
# :class:`DeclarativeAttentionHeadSpec` calls and ``head = 2`` in the
# ALiBi bake -- which made adding a future L9 head fragile (the author
# had to remember which slots were already taken). With the allocator
# the handoff is structural: each bake instantiates its OWN allocator
# pre-loaded with the full L9 head layout (pinned to existing slots),
# stashes it on ``attn._l9_head_allocator`` for downstream inspection,
# and resolves its own head index by name. A future L9 attention op
# can claim a free head via ``allocator.alloc(name, layer_idx=9)`` --
# with no ``pin=`` -- without touching this table.
#
# NOTE: ``format_string_fetch_head`` (gated by ``enable_conversational_io``)
# also writes head 0 in the conversational-I/O path, intentionally
# clobbering ``layer9_lev_addr_relay`` slopes/weights. That aliasing
# pre-dates the allocator and is not modeled here; the allocator
# forbids head aliasing, so the convo-I/O head stays outside this
# layout until a follow-up reconciles the two ops onto distinct slots.
_ALU_HEAD_LAYOUT = (
    # (op-name key,)  -- no pinned head_idx; allocator first-fits in
    # declaration order, landing at 0 / 1 / 2 / 3 byte-identically
    # (Phase 8.B retry attn pin drop).
    ("layer9_lev_addr_relay",),
    ("layer9_lev_bp_to_pc_relay",),
    ("layer9_alibi_mem_attn",),
    # Wave A v2 (2026-06-10): register-tagged STEP_END operand relay.
    # Q@MARK_SE_ONLY, K@MARK_AX (within-step), V copies raw
    # ALU_LO/HI / AX_CARRY_LO/HI / CMP / OP_<cmp> into the SE_-tagged
    # mirror dims so the migrated L9 CMP rules (which now gate on
    # MARK_SE_ONLY -- commit 62b64449) have the operand state at the
    # SE row without colliding with the raw band readers downstream.
    # See docs/STEP_END_COMPUTE_ARCHITECTURE_2026_06_10.md (Wave A) and
    # memory note ``project_wave_b_cmp_needs_l9_internal_relay.md``.
    ("layer9_step_end_operand_relay",),
)


def _allocate_alu_attention_heads() -> AttentionHeadAllocator:
    """Build a per-bake :class:`AttentionHeadAllocator` for L9 heads.

    Phase 8.B retry auto-fit: the allocator is built in
    ``dynamic_first_fit`` mode without ``pin=`` hints. Each entry is
    allocated in declaration order; first-fit picks the lowest free
    head each call, so the 3 declarations land at indices 0 / 1 / 2 --
    byte-identical to the legacy pinned offsets. The alibi-slope
    overrides in each bake key off the allocator-resolved indices, so
    the LEV/MEM relay slopes follow their semantic heads across any
    future allocator reshuffle.
    """
    allocator = AttentionHeadAllocator(strategy="dynamic_first_fit")
    for (name,) in _ALU_HEAD_LAYOUT:
        allocator.alloc(name, layer_idx=9)
    return allocator


def _alu_head_idx(op_name: str) -> int:
    """Return the L9 ``head_idx`` for ``op_name`` under the current
    layout / allocator strategy.

    Replays :func:`_allocate_alu_attention_heads` (cheap; the L9
    pool has 3 declared heads) and returns the named head's resolved
    ``head_idx``. The static-lookup contract held before the pin drop,
    but with ``dynamic_first_fit`` the layout decision is "compute the
    assignment by replaying the allocator", so the helper now does
    exactly that. Bakes that need to thread the same allocator
    through multiple call sites should pass the allocator instance
    directly rather than re-replaying for every lookup.
    """
    allocator = _allocate_alu_attention_heads()
    for rec in allocator.heads():
        if rec.op_name == op_name:
            return rec.head_idx
    raise KeyError(f"_alu_head_idx: unknown L9 attention op {op_name!r}")


# === L9 FFN unit layout (auto-fit; legacy offsets retained as docs) ==
#
# The ``layer9_alu`` op owns the entire L9 FFN. The actual weight writes
# happen inside ``vm_step._set_layer9_alu`` + ``_set_layer9_marker_suppress``,
# which use a local ``unit = 0`` counter that increments through 3405
# sub-stages.
#
# Phase 7.B.4: every entry below is auto-placed by
# :class:`FFNUnitAllocator` first-fit. Because the layout is fully
# contiguous in declaration order (every entry starts exactly where
# the previous one ended), first-fit reproduces the legacy pinned
# offsets bit-for-bit. The underlying ``_set_layer9_alu`` helper /
# ``Primitives.lower_ffn_rules`` cursor positions the actual weight
# writes, so byte-identity with the legacy bake is preserved
# regardless of allocator order. The ``legacy_start`` column is
# documentation only.
#
# The offsets below mirror the unit-counter walk in
# ``vm_step._set_layer9_alu`` (carry/borrow doubled inner loops) followed by
# ``_set_layer9_marker_suppress`` (7 NEXT_* dims). Changing any helper's
# unit count requires updating this table in lock-step.
_L9_ALU_UNIT_LAYOUT = (
    # (sub-stage name, legacy_start (docs only), n_units)
    ("layer9_alu.add_hi_nibble",         0, 512),  # ADD hi nibble (carry x 256)
    ("layer9_alu.lea_hi_nibble",       512, 512),  # LEA hi nibble (carry x 256)
    ("layer9_alu.adj_hi_nibble",      1024, 512),  # ADJ hi nibble (carry x 256)
    ("layer9_alu.sub_hi_nibble",      1536, 512),  # SUB hi nibble (borrow x 256)
    ("layer9_alu.ent_hi_nibble",      2048, 512),  # ENT hi nibble (borrow x 256)
    ("layer9_alu.hi_eq",              2560,  16),  # CMP+1 hi-eq
    ("layer9_alu.lo_eq",              2576,  16),  # CMP+2 lo-eq
    ("layer9_alu.hi_lt",              2592, 120),  # CMP+0 hi-lt
    ("layer9_alu.lo_lt",              2712, 120),  # CMP+3 lo-lt
    ("layer9_alu.add_carry_out",      2832, 256),  # CARRY+1 add carry-out
    ("layer9_alu.sub_borrow_out",     3088, 256),  # CARRY+2 sub borrow-out
    ("layer9_alu.alu_lo_clear",       3344,  16),  # ALU_LO clear at non-ALU op
    ("layer9_alu.alu_hi_clear",       3360,  16),  # ALU_HI clear at non-ALU op
    ("layer9_alu.bp_plus8_shift",     3376,  16),  # ADDR_B0_LO BP+8 lo-nibble shift
    ("layer9_alu.addr_b1_lo_set",     3392,   1),  # ADDR_B1_LO+15
    ("layer9_alu.addr_b1_hi_set",     3393,   1),  # ADDR_B1_HI+15
    ("layer9_alu.cascade_b0_hi",      3394,   1),  # cascade ADDR_B0_HI[15]->[0]
    ("layer9_alu.cascade_b1_lo",      3395,   1),  # cascade ADDR_B1_LO[15]->[0]
    ("layer9_alu.cascade_b1_hi",      3396,   1),  # cascade ADDR_B1_HI[15]->[0]
    ("layer9_alu.cascade_b2_lo",      3397,   1),  # cascade ADDR_B2_LO+1
    ("layer9_alu.marker_suppress",    3398,   7),  # _set_layer9_marker_suppress NEXT_*
)


def _allocate_alu_units() -> FFNUnitAllocator:
    """Build a per-bake :class:`FFNUnitAllocator` with all L9 ALU sub-stages.

    Phase 7.B.4: ``pin=`` is dropped from every entry in
    :data:`_L9_ALU_UNIT_LAYOUT`. The allocator's default first-fit
    strategy walks the layout in declaration order and lands each
    sub-stage at the lowest free gap large enough to hold it. Because
    the layout is fully contiguous (every entry starts exactly where
    the previous one ended), first-fit reproduces the legacy pinned
    offsets bit-for-bit. The underlying ``vm_step._set_layer9_alu``
    helper -- which writes via its own monotonic ``unit = 0`` counter --
    lands on exactly the same hidden-unit indices regardless of
    allocator order, so byte-identity with the legacy bake is
    preserved. The allocator's role is bookkeeping: the layout declares
    ranges by name, the helper writes the weights. A future refactor
    can split the monolithic helper into per-range bake functions that
    consume ``allocator.alloc(...)`` directly.

    Returns the allocator so callers can inspect or extend it (e.g. a
    future L9 op claims a free range past unit 3405).
    """
    allocator = FFNUnitAllocator()
    for name, _legacy_start, n_units in _L9_ALU_UNIT_LAYOUT:
        allocator.alloc(name, n_units)
    return allocator


# ====================================================================
# L9 ALU sub-stage rule helpers (Phase 6 Wave 4C migration).
#
# Each helper returns a tuple of :class:`FFNRule` instances that exactly
# reproduces the corresponding sub-stage's hidden-unit writes in
# :func:`vm_step._set_layer9_alu` (or
# :func:`vm_step._set_layer9_marker_suppress` for the trailing suppress
# range). The rules use ``"NAME+offset"`` string dim references so they
# can be lowered through :func:`Primitives.lower_ffn_rules` against the
# compiler-allocated ``dim_positions`` map.
#
# The unit ordering inside each helper matches the imperative loop
# nesting (outermost ``carry_in``/``borrow_in`` -> ``a`` -> ``b``) so a
# pinned ``start_unit`` lowering lands byte-for-byte where the legacy
# helper used to write.
# ====================================================================


# Markers the LEA/ADJ/ENT high-nibble units block when the AX-marker
# amplification is high. Mirrors ``_block_non_ax_marker_sites`` inside
# :func:`vm_step._set_layer9_alu`.
_NON_AX_BLOCKERS: tuple[str, ...] = (
    "MARK_PC",
    "MARK_SP",
    "MARK_BP",
    "MARK_STACK0",
    "MARK_MEM",
    "MARK_SE",
    "IS_BYTE",
)


def _add_hi_nibble_rules(S: float) -> tuple[FFNRule, ...]:
    """ADD hi-nibble cross-product (512 units).

    Mirrors the first loop in :func:`vm_step._set_layer9_alu` -- 4-way
    AND at the AX marker over ``ALU_HI[a]`` (operand A high nibble) and
    ``AX_CARRY_HI[b]`` (operand B high nibble) with the inter-byte
    ``CARRY[0]`` either repelled (carry_in=0, threshold 2.5) or required
    (carry_in=1, threshold 4.5). Gated by ``OP_ADD``; writes
    ``OUTPUT_HI_THIS_STEP+result`` where ``result = (a + b + carry_in) %
    16``.

    Phase 7.E.3: gate uses :func:`dim_ref` for the
    ``(opcode_flag, ADD)`` semantic pair. The ``OUTPUT_HI_THIS_STEP+result``
    write stays structural (result is a value-bus lookup index, not a
    role-meaningful byte position).

    Phase 7.E (sem-dim follow-up): the ``CARRY+0`` carry-in read
    resolves through ``dim_ref("carry", "alu", 0)`` -- byte 0 of the
    inter-byte ALU carry cascade (the same semantic position that
    ``_layer8_alu_add_carry_rules`` *writes*). Byte-identical: the
    helper returns the legacy ``"CARRY+0"`` string verbatim.

    DERIVED (2026-07): the hi-nibble ADD lookup with carry-IN from the
    L8-generated CARRY+0 (`result=(a+b+cin)%16`) is a computed-lookup
    member of the ALU derivable regime (docs/semantic_spec_ALU.md §G2 --
    the inter-nibble carry cascade). The per-value loop is deleted and
    routed through ``wide_alu_dsl.nibble_alu_lane_rules``: byte 0's high
    nibble reads the carry-in dim (thr 2.5 no-carry / 4.5 with-carry,
    +/-2.0 CARRY+0 discrimination). Byte-identity gated by
    ``tools/verify_l8l9_addsub_derived.py`` (and the golden hash).
    """
    return nibble_alu_lane_rules(
        op="add", emit="result",
        operand_a_band="ALU_HI", operand_b_band="AX_CARRY_HI",
        marker_gate="MARK_AX", marker_weight=1.0, mark_pc_weight=-2.0,
        gate=dim_ref("opcode_flag", "ADD"),
        threshold_no_carry=2.5, threshold_with_carry=4.5,
        carry_in_dim=dim_ref("carry", "alu", 0), carry_in_weight=2.0,
        result_band="OUTPUT_HI_THIS_STEP", write_scale=2.0 / S,
        name_fn=lambda c, a, b: f"l9_add_hi_c{c}_a{a}_b{b}",
    )


def _lea_hi_nibble_rules(S: float) -> tuple[FFNRule, ...]:
    """LEA hi-nibble cross-product (512 units).

    Mirrors the second loop in :func:`vm_step._set_layer9_alu` -- amplified
    ``MARK_AX`` AND with operand ``ALU_HI[a]`` and the FETCH-side
    ``FETCH_HI[b]`` immediate. Carry discrimination via ``CARRY[0]`` at the
    ``S * 8.0`` strength to maintain selectivity against the
    ``S * 40``-scale main score. The seven non-AX marker dims are repelled
    at ``-S * 1000`` so PC/SP/BP/STACK0/MEM/SE/byte rows never fire even
    when LEA's FETCH gate is high. Gated by ``OP_LEA``; writes
    ``OUTPUT_HI_THIS_STEP+result``.

    Phase 7.E.3: gate uses :func:`dim_ref` for the
    ``(opcode_flag, LEA)`` semantic pair; the ``CARRY+0`` carry-in read
    resolves through ``dim_ref("carry", "alu", 0)`` (byte 0 of the
    inter-byte ALU carry cascade). Structural reads
    (``ALU_HI+a`` / ``FETCH_HI+b``) and the result-nibble write stay as
    ``+N`` (value-bus one-hot lookups).

    DERIVED (2026-07): the 10-way amplified AND (MARK_AX(+20) + the seven
    non-AX marker blockers(-1000) + ALU_HI nibble(+1) + FETCH_HI immediate
    nibble(+20) + CARRY+0 carry-in discrimination(+/-8), thresholds 40.5 /
    48.5, writing ``(a+b+carry_in)%16``) is THE shared ``reg + live-operand``
    adder ADJ/ENT also use. The per-value loop is routed through
    ``wide_alu_dsl.amplified_nibble_adder_rules``; ADJ's SP-add is derived
    from this same generator via the ``SpDelta`` (``runtime_add``)
    ``RegisterDeltaSpec`` kind (``isa_semantics_dsl.control_op``). Byte-identity
    gated by the whole-model golden hash.
    """
    return amplified_nibble_adder_rules(
        op="add",
        operand_a_band="ALU_HI", operand_b_band="FETCH_HI",
        marker_gate="MARK_AX", blocker_dims=_NON_AX_BLOCKERS,
        blocker_weight=1000.0,
        gate=dim_ref("opcode_flag", "LEA"),
        carry_in_dim=dim_ref("carry", "alu", 0), carry_in_weight=8.0,
        threshold_no_carry=40.5, threshold_with_carry=48.5,
        result_band="OUTPUT_HI_THIS_STEP", write_scale=2.0 / S,
        name_fn=lambda c, a, b: f"lea_hi_c{c}_a{a}_b{b}",
    )


def _adj_control_op(S: float):
    """The ADJ ``control_op`` frame descriptor — ADJ = ``SP += imm``.

    ``docs/semantic_spec_CONTROL.md`` §2b: ``ADJ = ControlOp(pc_next=SEQUENTIAL,
    frame_delta=[SP_DELTA(+, imm)])``. The single ``SP_DELTA(+, imm)`` is the
    ``SpDelta`` runtime adder — ``SP += imm`` where ``imm`` is a LIVE operand
    (the instruction immediate's high nibble in ``FETCH_HI``), NOT a compile-time
    constant. It DELEGATES to the shared L9 amplified nibble adder (the same one
    LEA/ENT use). The whole 512-unit ADJ hi-nibble band is now this ONE
    ``RuntimeAddDelta`` DATA row.

    Byte-identical to the prior hand-authored ``_adj_hi_nibble_rules`` (proven
    by the whole-model golden hash == ``91f55411``): the derived rules reproduce
    the amplified MARK_SE_ONLY AND (``+20`` marker, the seven non-AX blockers at
    ``-1000``, ``ALU_HI[a]`` ``+1``, ``FETCH_HI[b]`` ``+20``, ``CARRY+0``
    carry-in ``+/-8``, thresholds 42.0 / 50.0) writing ``(a+b+carry_in)%16`` to
    ``OUTPUT_HI_THIS_STEP`` — cell-for-cell.

    Cluster 6 (2026-06-10) — the marker is ``MARK_SE_ONLY`` (not ``MARK_AX``):
    the Wave A ``step_end_operand_relay`` head broadcasts ALU_HI/FETCH_HI/CARRY/
    OP_ADJ into the STEP_END row so the amplified AND fires there over identical
    operand state, keeping the write off the post-ADJ AX byte
    (``IMM 42; PSH; ADJ 8; EXIT`` -> 42, not 10). See
    ``docs/SMOKE_TRIAGE_2026_06_10.md`` Cluster 6.
    """
    gate_adj = dim_ref("opcode_flag", "ADJ")
    sp_delta = RegisterDeltaSpec(
        name="adj_sp_delta",
        kind="runtime_add",
        write_scale=2.0 / S,
        conditions=((gate_adj, 1.0),),
        runtime_add=RuntimeAddDelta(
            gate=gate_adj,
            op="add",
            operand_a_band="ALU_HI",
            operand_b_band="FETCH_HI",
            result_band="OUTPUT_HI_THIS_STEP",
            marker_gate="MARK_SE_ONLY",
            blocker_dims=tuple(_NON_AX_BLOCKERS),
            carry_in_dim=dim_ref("carry", "alu", 0),
            threshold_no_carry=42.0,
            threshold_with_carry=50.0,
            write_scale=2.0 / S,
            name_fn=lambda c, a, b: f"adj_hi_c{c}_a{a}_b{b}_step_end",
        ),
    )
    return control_op("ADJ", [sp_delta], instr_width=8, pc_offset=2)


def _adj_hi_nibble_rules(S: float) -> tuple[FFNRule, ...]:
    """ADJ hi-nibble band (512 units) — DERIVED from the ADJ ``control_op``.

    ADJ = ``SP += imm`` is a single ``SP_DELTA(+, imm)`` (§2b). That delta is
    the ``SpDelta`` runtime adder (:class:`isa_semantics_dsl.RuntimeAddDelta`,
    ``kind="runtime_add"``), whose amount comes from the LIVE ``FETCH_HI``
    operand band and delegates to the shared L9 amplified nibble adder. The
    previously hand-authored 512-unit ``for carry_in: for a: for b:``
    cross-product is DELETED — this is now :func:`_adj_control_op`'s one derived
    band. Byte-identical to the prior loop (golden hash == ``91f55411``).
    """
    return _adj_control_op(S).rules_builder()


def _sub_hi_nibble_rules(S: float) -> tuple[FFNRule, ...]:
    """SUB hi-nibble cross-product (512 units).

    Mirrors the SUB loop in :func:`vm_step._set_layer9_alu` -- 4-way
    AND at the AX marker over ``ALU_HI[a]`` (stack-top high nibble) and
    ``AX_CARRY_HI[b]`` (AX high nibble) with borrow-in discrimination on
    ``CARRY[0]``. SUB computes ``stack_top - AX = a - b``, so the
    written nibble is ``(a - b - borrow_in) % 16``. Gated by ``OP_SUB``.

    Phase 7.E.3: gate uses :func:`dim_ref` for the
    ``(opcode_flag, SUB)`` semantic pair.

    DERIVED (2026-07): the hi-nibble SUB lookup with borrow-IN from
    CARRY+0 (`result=(a-b-bin)%16`) routed through
    ``wide_alu_dsl.nibble_alu_lane_rules`` (mirror of add_hi, op="sub");
    byte-identity gated by ``tools/verify_l8l9_addsub_derived.py``.
    """
    return nibble_alu_lane_rules(
        op="sub", emit="result",
        operand_a_band="ALU_HI", operand_b_band="AX_CARRY_HI",
        marker_gate="MARK_AX", marker_weight=1.0, mark_pc_weight=-2.0,
        gate=dim_ref("opcode_flag", "SUB"),
        threshold_no_carry=2.5, threshold_with_carry=4.5,
        carry_in_dim=dim_ref("carry", "alu", 0), carry_in_weight=2.0,
        result_band="OUTPUT_HI_THIS_STEP", write_scale=2.0 / S,
        name_fn=lambda c, a, b: f"sub_hi_b{c}_a{a}_b{b}",
    )


def _layer9_ent_hi_nibble_rules(S: float) -> tuple[FFNRule, ...]:
    """ENT hi-nibble cross-product (512 units).

    Same structural shape as the LEA/ADJ amplified loops (AX-marker
    amplification with non-AX markers blocked at ``-S*1000``, ``ALU_HI``
    + ``FETCH_HI`` cross-product, ``CARRY[0]`` borrow discrimination)
    but gated on ``OP_ENT``. ENT computes ``SP - imm`` with borrow
    propagation; ``a`` is SP's high nibble (``ALU_HI``) and ``b`` is the
    immediate's high nibble (``FETCH_HI``). Writes
    ``OUTPUT_HI_THIS_STEP+((sp_hi - imm_hi - borrow_in) % 16)``.

    Phase 7.E.3: gate uses :func:`dim_ref` for the
    ``(opcode_flag, ENT)`` semantic pair.
    """

    gate_ent = dim_ref("opcode_flag", "ENT")
    carry_byte0 = dim_ref("carry", "alu", 0)
    rules: list[FFNRule] = []
    # Same 10-way amplified AND shape as _lea_hi_nibble_rules /
    # _adj_hi_nibble_rules but gated on OP_ENT, computing SP - imm
    # (a = SP's hi nibble via ALU_HI, b = imm's hi nibble via FETCH_HI).
    # Thresholds 42.0 / 50.0 match the ADJ tuning. Writes
    # (sp_hi - imm_hi - borrow_in) % 16 to OUTPUT_HI_THIS_STEP.
    #
    # BUG FIX 2026-06-11 (step-1 ENT-AX 0xf0 leak, genesis): this band
    # materializes the ENT SP hi nibble (SP - (8+imm)) into OUTPUT_HI at
    # the AX *register-marker* row, where it doubles as the step-0 BP
    # cascade source (BP = old_SP - 8 reads the AX-row frame value). On the
    # **first** ENT step (HAS_SE = 0) that contribution is load-bearing —
    # ``test_lea_basic``'s LEA result depends on it. But on a **subsequent**
    # ENT step (HAS_SE = 1, e.g. the JSR→ENT prologue: var/func/loop/rec),
    # AX byte0 is already correct (OUTPUT_HI[0]) entering this layer, and
    # the SP value comes from the dedicated AX_CARRY→SP writeback — NOT
    # from this OUTPUT_HI. There the OP_ENT in-step broadcast (~11.4 at the
    # AX marker) over-amplifies the band so the (sp_hi-imm_hi-borrow)=15
    # unit writes ~+177 on OUTPUT_HI[15], flipping the AX byte-0 marker-row
    # argmax 0→15 and emitting AX byte0 = 0xf0. ``layer14_ent_ax_bytes_zero``
    # only suppresses AX bytes 1-3 (IS_BYTE-gated), so byte 0 leaked.
    #
    # Fix (same class as the L16 JSR→BP prologue fix — neutralize the
    # genesis writer rather than fight its broadcast-amplified magnitude
    # downstream): add a hard ``HAS_SE`` NOT-blocker so the band fires
    # ONLY on the first ENT step. Subsequent-step ENT then keeps the clean
    # OUTPUT_HI[0] it already carries; step-0 ENT (test_lea_basic frame
    # setup) is byte-identical.
    for borrow_in in (0, 1):
        for sp_hi in range(16):
            for imm_hi in range(16):
                result = (sp_hi - imm_hi - borrow_in) % 16
                conditions: list[tuple[str, float]] = [("MARK_AX", 20.0)]
                conditions.extend(
                    (dim, -1000.0) for dim in _NON_AX_BLOCKERS
                )
                # Subsequent-step ENT (HAS_SE) must NOT re-materialize the
                # SP hi nibble onto the AX row; hard-block so the OP_ENT
                # broadcast cannot reopen it. First-step ENT (HAS_SE=0)
                # keeps the BP-cascade contribution.
                conditions.append(("HAS_SE", -1000.0))
                conditions.append((f"ALU_HI+{sp_hi}", 1.0))
                conditions.append((f"FETCH_HI+{imm_hi}", 20.0))
                if borrow_in == 0:
                    conditions.append((carry_byte0, -8.0))
                    threshold = 42.0
                else:
                    conditions.append((carry_byte0, 8.0))
                    threshold = 50.0
                rules.append(multi_way_and_rule(
                    name=f"ent_hi_b{borrow_in}_sp{sp_hi}_imm{imm_hi}",
                    conditions=tuple(conditions),
                    threshold=threshold,
                    gate=gate_ent,
                    writes=((f"OUTPUT_HI_THIS_STEP+{result}", 2.0 / S),),
                ))
    return tuple(rules)


def _layer9_cmp_rules(S: float) -> tuple[FFNRule, ...]:
    """Comparison-flag cross-products (272 units total).

    Mirrors the four CMP loops in :func:`vm_step._set_layer9_alu`:

      * ``hi_eq`` -- 16 units writing ``CMP+1`` (high-nibble equality)
      * ``lo_eq`` -- 16 units writing ``CMP+2`` (low-nibble equality)
      * ``hi_lt`` -- 120 units writing ``CMP+0`` (high-nibble less-than)
      * ``lo_lt`` -- 120 units writing ``CMP+3`` (low-nibble less-than)

    Every CMP unit is a 3-way AND at the AX marker (``MARK_AX`` and
    operand a/b dims, with ``MARK_PC`` repelled at ``-S * 2``) gated on
    ``CMP_GROUP`` (the L8 comparison-opcode-group flag). Threshold 2.5
    for the 3-way AND; writes ``2.0 / S`` to the matching CMP output
    nibble. Equality uses ``a == k`` for both operand dims; less-than
    uses ``a < b`` over the (a, b) pairs.

    Phase 7.E.3: gate uses :func:`dim_ref` for the
    ``(cmp_flag, group)`` pair; CMP-cascade byte writes use
    ``dim_ref("cmp_flag", "cascade", byte_index)`` -- the ``+0..+3``
    offsets are role-meaningful (hi_lt/hi_eq/lo_eq/lo_lt cascade byte
    positions). Operand ``ALU_*+k`` / ``AX_CARRY_*+k`` reads remain
    structural (per-nibble one-hot lookups).
    """

    # Wave A v2 (2026-06-10): SE-tagged CMP_GROUP gate. The rule now
    # fires at MARK_SE_ONLY where the raw CMP_GROUP dim is zero (it was
    # written at MARK_AX by L8). ``layer9_step_end_operand_relay``
    # mirrors CMP_GROUP into ``SE_CMP_GROUP`` at the SE row, so the
    # gate stays semantically equivalent. The cmp cascade outputs
    # remain raw ``CMP+k`` (downstream BZ/BNZ predicate gate reads the
    # raw cascade at MARK_AX cross-step via prev-step OUTPUT relays).
    gate_cmp_group = "SE_CMP_GROUP+0"
    cmp_byte0 = dim_ref("cmp_flag", "cascade", 0)
    cmp_byte1 = dim_ref("cmp_flag", "cascade", 1)
    cmp_byte2 = dim_ref("cmp_flag", "cascade", 2)
    cmp_byte3 = dim_ref("cmp_flag", "cascade", 3)
    # Each CMP rule is a 4-way AND at the AX marker: MARK_AX(+1) +
    # MARK_PC(-2) blocker + two operand-nibble one-hots(+1 each), gated
    # on CMP_GROUP. The MARK_PC negative weight prevents the rule firing
    # at PC marker positions where MARK_AX might leak in. The explicit
    # threshold=2.5 is passed (the default derivation requires all
    # positive weights and would not pass the negative MARK_PC blocker).

    # Wave B Cluster 3 rows 1-4 + Wave A v2 (2026-06-10): the CMP
    # factory's 4-way AND now fires at MARK_SE_ONLY (STEP_END) instead
    # of MARK_AX. The Wave A v2 ``layer9_step_end_operand_relay``
    # attention head mirrors ALU_LO/HI / AX_CARRY_LO/HI / CMP /
    # OP_<cmp> from the same step's MARK_AX row into the SE_-tagged
    # SE_ALU_LO/HI / SE_AX_CARRY_LO/HI / SE_CMP / SE_OP_<cmp> slots
    # at MARK_SE_ONLY. The CMP rules read the SE_-tagged mirrors so
    # they fire at the SE row without colliding with the raw band
    # readers downstream (the previous L11 relay 10ca51a7 wrote raw
    # ALU_LO/HI at MARK_SE and regressed 9 tests by polluting the
    # cross-step / downstream-layer consumers of those bands; the
    # SE_-prefixed mirrors are scoped ``mark == SE_ONLY`` so non-CMP
    # readers never see them). The MARK_PC negative blocker is
    # retained: at MARK_SE_ONLY the dim is zero so the term is
    # structurally inert but preserves the AND threshold arithmetic.
    # See ``step_end_migration.py`` for the helper and
    # ``WAVE_B_CLUSTER_3_PLAN_2026_06_10.md`` for the migration recipe.
    #
    # DERIVED (2026-07): the four CMP lanes are NIBBLE COMPARATORS —
    # ``a == b`` equality (hi_eq / lo_eq) and ``a < b`` less-than (hi_lt /
    # lo_lt) over the SE_-tagged operand mirrors. The per-value loops are
    # deleted and routed through
    # ``wide_alu_dsl.nibble_compare_lane_rules`` (the comparator sibling of
    # ``nibble_alu_lane_rules``). Byte-identity gated by
    # ``tools/verify_l9_cmp_derived.py`` (and the golden hash). This does NOT
    # hit the MARK_AX-vs-MARK_SE_ONLY wall from
    # ``project_wave_b_cmp_needs_l9_internal_relay.md``: that wall was already
    # closed by the ``layer9_step_end_operand_relay`` head, and the generator
    # simply consumes the SE_-tagged mirror bands it produces.
    rules: list[FFNRule] = []

    # hi_eq: 16 units -> CMP+1 (a == b for hi nibble)
    rules.extend(nibble_compare_lane_rules(
        mode="equal",
        operand_a_band="SE_ALU_HI", operand_b_band="SE_AX_CARRY_HI",
        marker_gate="MARK_SE_ONLY", marker_weight=1.0, mark_pc_weight=-2.0,
        gate=gate_cmp_group, threshold=2.5,
        flag_dim=cmp_byte1, flag_scale=2.0 / S,
        name_fn=lambda a, b: f"l9_cmp_hi_eq_{a}_step_end",
    ))

    # lo_eq: 16 units -> CMP+2 (a == b for lo nibble)
    rules.extend(nibble_compare_lane_rules(
        mode="equal",
        operand_a_band="SE_ALU_LO", operand_b_band="SE_AX_CARRY_LO",
        marker_gate="MARK_SE_ONLY", marker_weight=1.0, mark_pc_weight=-2.0,
        gate=gate_cmp_group, threshold=2.5,
        flag_dim=cmp_byte2, flag_scale=8.0 / S,
        name_fn=lambda a, b: f"l9_cmp_lo_eq_{a}_step_end",
    ))

    # hi_lt: 120 units -> CMP+0 (a < b for hi nibble)
    rules.extend(nibble_compare_lane_rules(
        mode="less_than",
        operand_a_band="SE_ALU_HI", operand_b_band="SE_AX_CARRY_HI",
        marker_gate="MARK_SE_ONLY", marker_weight=1.0, mark_pc_weight=-2.0,
        gate=gate_cmp_group, threshold=2.5,
        flag_dim=cmp_byte0, flag_scale=2.0 / S,
        name_fn=lambda a, b: f"l9_cmp_hi_lt_a{a}_b{b}_step_end",
    ))

    # lo_lt: 120 units -> CMP+3 (a < b for lo nibble)
    rules.extend(nibble_compare_lane_rules(
        mode="less_than",
        operand_a_band="SE_ALU_LO", operand_b_band="SE_AX_CARRY_LO",
        marker_gate="MARK_SE_ONLY", marker_weight=1.0, mark_pc_weight=-2.0,
        gate=gate_cmp_group, threshold=2.5,
        flag_dim=cmp_byte3, flag_scale=2.0 / S,
        name_fn=lambda a, b: f"l9_cmp_lo_lt_a{a}_b{b}_step_end",
    ))

    # ``_CMP_RELAYED`` documents the Wave A dims this factory depends
    # on; the constant is read by audit tooling / future relay
    # validators.
    _ = _CMP_RELAYED

    return tuple(rules)


def _add_carry_out_rules(S: float) -> tuple[FFNRule, ...]:
    """ADD hi-nibble carry-out -> CARRY+1 (256 units).

    For every (a, b, carry_in) where ``a + b + carry_in >= 16`` the unit
    fires at the AX marker and writes ``CARRY+1`` (byte-level carry for
    inter-byte propagation). Same 3-way operand AND used by the ADD
    hi-nibble sub-stage, but the ``CARRY[0]`` discrimination uses the
    weaker ``+/- 0.01`` weights documented in the legacy helper -- the
    raw stack-top carry signal is at value ~1.0 here, not amplified, so
    the threshold itself does most of the discrimination work. Gated by
    ``OP_ADD``.

    Phase 7.E.3: gate + carry-output refs use :func:`dim_ref` for the
    ``(opcode_flag, ADD)`` and ``(carry, alu, byte_index=1)`` pairs.
    The carry-output offset 1 is role-meaningful (byte 1 of the
    inter-byte ALU carry cascade); operand reads stay structural.

    DERIVED (2026-07): the byte-LEVEL ADD carry-out lookup
    (`a+b+cin>=16` -> CARRY+1) routed through
    ``wide_alu_dsl.nibble_alu_lane_rules`` (emit="carry_flag"). Weak
    +/-0.01/S CARRY+0 discrimination + relaxed thresholds (2.5/2.9);
    byte-identity gated by ``tools/verify_l8l9_addsub_derived.py``.
    """
    return nibble_alu_lane_rules(
        op="add", emit="carry_flag",
        operand_a_band="ALU_HI", operand_b_band="AX_CARRY_HI",
        marker_gate="MARK_AX", marker_weight=1.0, mark_pc_weight=-2.0,
        gate=dim_ref("opcode_flag", "ADD"),
        threshold_no_carry=2.5, threshold_with_carry=2.9,
        carry_in_dim=dim_ref("carry", "alu", 0), carry_in_weight=0.01 / S,
        carry_flag_dim=dim_ref("carry", "alu", 1), carry_flag_scale=2.0 / S,
        name_fn=lambda c, a, b: f"add_carry_out_c{c}_a{a}_b{b}",
    )


def _sub_borrow_out_rules(S: float) -> tuple[FFNRule, ...]:
    """SUB hi-nibble borrow-out -> CARRY+2 (256 units).

    Mirrors the SUB borrow-out loop in :func:`vm_step._set_layer9_alu`.
    The pair-filter selects only the operand combinations that produce a
    borrow: ``a < b`` when ``borrow_in == 0``, or ``a <= b`` when
    ``borrow_in == 1``. Same weak ``±0.01`` carry discrimination /
    relaxed thresholds as the ADD carry-out band. Gated by ``OP_SUB``.

    Phase 7.E.3: gate + carry-output refs use :func:`dim_ref` for the
    ``(opcode_flag, SUB)`` and ``(carry, alu, byte_index=2)`` pairs.
    The ``+2`` offset is the byte-2 position of the inter-byte ALU
    carry cascade.

    DERIVED (2026-07): the byte-LEVEL SUB borrow-out lookup routed
    through ``wide_alu_dsl.nibble_alu_lane_rules`` (emit="carry_flag",
    op="sub"). The pair-filter `a<b (bin=0) / a<=b (bin=1)` == the
    op="sub" overflow `(a-b-bin)<0`. Writes CARRY+2 (byte-2 of the
    inter-byte cascade, the downstream CarryPropagation SUB input); weak
    +/-0.01/S CARRY+0 discrimination + relaxed thresholds (2.5/2.9);
    byte-identity gated by ``tools/verify_l8l9_addsub_derived.py``.
    """
    return nibble_alu_lane_rules(
        op="sub", emit="carry_flag",
        operand_a_band="ALU_HI", operand_b_band="AX_CARRY_HI",
        marker_gate="MARK_AX", marker_weight=1.0, mark_pc_weight=-2.0,
        gate=dim_ref("opcode_flag", "SUB"),
        threshold_no_carry=2.5, threshold_with_carry=2.9,
        carry_in_dim=dim_ref("carry", "alu", 0), carry_in_weight=0.01 / S,
        carry_flag_dim=dim_ref("carry", "alu", 2), carry_flag_scale=2.0 / S,
        name_fn=lambda c, a, b: f"sub_borrow_out_b{c}_a{a}_b{b}",
    )


# Opcodes that don't need ALU operand gather; the ALU clear units fire
# at MARK_AX positions when ANY of these opcodes is active so residual
# ALU_LO/HI values do not contaminate Layer 10 bitwise / MUL units.
# Mirrors the inline ``non_alu_opcodes`` list inside
# :func:`vm_step._set_layer9_alu`.
_NON_ALU_OPCODES: tuple[str, ...] = (
    "OP_IMM",
    "OP_NOP",
    "OP_JMP",
    "OP_JSR",
    "OP_EXIT",
    "OP_BZ",
    "OP_BNZ",
    "OP_ENT",
    "OP_ADJ",
    "OP_LEV",
    "OP_PSH",
    "OP_LI",
    "OP_LC",
    "OP_SI",
    "OP_SC",
)


def _alu_clear_rules(S: float) -> tuple[FFNRule, ...]:
    """ALU LO/HI clear at non-ALU opcodes (32 units).

    Pair of identical 16-unit bands writing ``-10.0 / S`` to
    ``ALU_LO+k`` and ``ALU_HI+k`` respectively. Both bands fire when
    ``MARK_AX`` and any non-ALU opcode is active (threshold 1.5,
    constant gate at ``gate_bias=1.0``). The residual ALU_* values from
    earlier layers would otherwise leak into Layer 10's bitwise / MUL
    units at non-ALU positions; the clear band cancels them.
    """

    # Conditions = MARK_AX + OR over the 15 non-ALU opcodes (all unit weights).
    # Threshold 1.5 implements "MARK_AX AND any one non-ALU opcode" semantics:
    # MARK_AX(=1) + exactly one OP_*(=1) sums to 2.0 > 1.5 while OP_* alone
    # only sums to 1.0 (blocked). multi_way_and_rule with explicit
    # threshold=1.5 captures the same structural N-way conditional write
    # (default threshold derivation would over-shoot to 15.5 = "all on").
    common_conditions = [("MARK_AX", 1.0)]
    common_conditions.extend((dim, 1.0) for dim in _NON_ALU_OPCODES)
    # Opcode-broadcast hardening (2026-06-11, batch fix): _NON_ALU_OPCODES
    # includes the broadcasting frame/control opcodes OP_JSR (~17.2),
    # OP_ENT (~11.4), OP_ADJ / OP_LEV (each carried as ``(dim, 1.0)``). Those
    # flags are broadcast IN-STEP to EVERY register/byte/marker row (NOT
    # one-hot at the opcode's own marker), so on a JSR/ENT step the opcode sum
    # ALONE (e.g. OP_JSR*1 = 17.2) clears thr 1.5 even where MARK_AX = 0, so the
    # clear could fire at a wrong NON-AX register MARKER row.
    #
    # IMPORTANT (legit-row coupling — verified spec_k=0 at the L9-input residual
    # for the SI/LI program, tools probe): this clear LEGITIMATELY fires at the
    # AX *byte* rows (IS_BYTE=1, OP_LI/SI ~5) as well as the AX marker — it is a
    # broad ALU residue scrubber across the whole AX register during non-ALU
    # ops, not a single-marker write. So an IS_BYTE NOT-blocker would veto the
    # legit byte-row clear and corrupt the loaded value (regresses
    # test_si_li_16bit_value). We therefore harden ONLY the NON-AX register
    # MARK_* dims, which ARE 0 at every legit firing row (AX marker + AX byte
    # rows): these -1e6 NOT-blockers are SUBTRACTIVE (byte-identical at the
    # legit rows) and veto the broadcast at every other-register marker row.
    # MARK_AX stays a positive condition; IS_BYTE is deliberately NOT blocked.
    common_conditions = tuple(common_conditions) + (
        ("MARK_PC", -1e6),
        ("MARK_SP", -1e6),
        ("MARK_BP", -1e6),
        ("MARK_STACK0", -1e6),
        ("MARK_MEM", -1e6),
    )

    # ALU_LO clear (16 units) then ALU_HI clear (16 units).
    #
    # Reduction map ⑥: this per-cell BYTE-CLEAR band is the shared
    # cross-layer primitive (mirrors the l14 OUTPUT-clear bands); delegate
    # to ``byte_clear_rules``. Byte-identical: band-major then cell-major
    # rule order, ``-10.0 / S`` per cell, legacy names preserved via
    # ``name_by_band``.
    return byte_clear_rules(
        bands=("ALU_LO", "ALU_HI"),
        conditions=common_conditions,
        threshold=1.5,
        write_value=-10.0,
        S=S,
        name_by_band={"ALU_LO": "alu_lo_clear", "ALU_HI": "alu_hi_clear"},
    )


def _layer9_bp_plus8_shift_rules(S: float) -> tuple[FFNRule, ...]:
    """``ADDR_B0_LO`` BP+8 shift at PC marker for LEV return (16 units).

    Mirrors the 16-unit shift block in :func:`vm_step._set_layer9_alu`:
    at the LEV PC marker the previous BP byte 0 needs ``+8`` so the
    return-address gather lands on ``[BP+8]``. Each unit reads
    ``ADDR_B0_LO[k]`` as its gate (with ``b_gate=-2.5`` to require the
    legitimate ~3.0 value from L9 head 0 and reject the ~2.0 opcode
    fetch contamination from L5), cancels the original ``+k`` slot, and
    re-energises the rotated ``(k + 8) % 16`` slot. PC-marker activation
    is gated by ``MARK_PC + OP_LEV/5`` with strong negative blockers for
    MARK_BP / MARK_SP to keep the unit from firing at those rows during
    LEV.
    """

    # Wave B Cluster 3 row 5: this rule originally gated on MARK_PC
    # (not MARK_AX); after Wave A relays OP_LEV / BP_FRAME_* into
    # STEP_END the same shift fires one row later at MARK_SE_ONLY
    # byte-identically. The MARK_BP / MARK_SP blockers are retained
    # for structural symmetry (the dims are zero at STEP_END so they
    # contribute nothing to the AND). See
    # ``WAVE_B_CLUSTER_3_PLAN_2026_06_10.md`` for the recipe.
    rules: list[FFNRule] = []
    for k in range(16):
        new_k = (k + 8) % 16
        rules.append(multi_way_and_rule(
            name=f"l9_bp_plus8_shift_{k}_step_end",
            conditions=(
                ("MARK_SE_ONLY", 1.0),
                ("OP_LEV", 1.0 / 5.0),
                ("MARK_BP", -10.0),
                ("MARK_SP", -10.0),
                # MARK_MEM hard blocker — the gate dim ADDR_B0_LO+k
                # aliases OPCODE_BYTE_LO at byte rows. MARK_SE_ONLY in
                # conditions restricts to STEP_END rows at runtime; the
                # blocker tightens the dim_alias_verifier's effective
                # predicate so MEM rows are explicitly excluded.
                ("MARK_MEM", -1e6),
            ),
            threshold=1.5,
            gate=f"ADDR_B0_LO+{k}",
            gate_weight=1.0,
            gate_bias=-2.5,
            writes=(
                (f"ADDR_B0_LO+{k}", -0.67 / S),
                (f"ADDR_B0_LO+{new_k}", 0.67 / S),
            ),
        ))
    # ``_BP_PLUS8_RELAYED`` documents the Wave A dims this factory
    # depends on; read by audit tooling / future relay validators.
    _ = _BP_PLUS8_RELAYED
    return tuple(rules)


def _addr_b1_set_and_cascade_rules(S: float) -> tuple[FFNRule, ...]:
    """``ADDR_B1`` 0xff set + BP=0xfff8 cascade fixup (6 units).

    Six contiguous units that finish setting up the LEV return-address
    bytes at the PC marker:

      * Unit 0 (``addr_b1_lo_set``) -- unconditional ``ADDR_B1_LO[15] = 1``
        (byte 1 = 0xff for stack-address pages). Constant gate on
        ``CONST`` (``W_gate[CONST]=1``) with ``b_gate=0``.
      * Unit 1 (``addr_b1_hi_set``) -- mirror unit writing
        ``ADDR_B1_HI[15] = 1``.
      * Units 2-5 (``cascade_b0_hi``, ``cascade_b1_lo``, ``cascade_b1_hi``,
        ``cascade_b2_lo``) -- fire only when ``ADDR_B0_LO[0]`` AND
        ``ADDR_B0_HI[15]`` are both high (the post-shift fingerprint of
        BP=0xf8). They cancel the just-set 0xff bytes and stamp the
        carried 0x10000 address (b0_hi[0], b1_lo[0], b1_hi[0],
        b2_lo[1]). The 2-way gate is implemented with
        ``gate="ADDR_B0_LO+0"`` plus a ``gate_terms`` entry for
        ``ADDR_B0_HI+15`` and ``gate_bias=-15.0`` so the AND threshold is
        only crossed in the cascade case.
    """

    common_conditions = (
        ("MARK_PC", 1.0),
        ("OP_LEV", 1.0 / 5.0),
        ("MARK_BP", -10.0),
        ("MARK_SP", -10.0),
        # MARK_MEM hard blocker — the cascade rules' gate
        # `ADDR_B0_LO+0` aliases the MEM-address slot at MEM rows.
        # MARK_PC in conditions already restricts firing to PC rows at
        # runtime; the blocker tightens the dim_alias_verifier's effective
        # predicate so MEM rows are explicitly excluded.
        ("MARK_MEM", -1e6),
    )
    threshold = 1.5

    rules: list[FFNRule] = []

    # Unit 0: ADDR_B1_LO[15] = 1 (byte 1 = 0xff lo nibble).
    rules.append(multi_way_and_rule(
        name="addr_b1_lo_set",
        conditions=common_conditions,
        threshold=threshold,
        gate="CONST",
        gate_weight=1.0,
        gate_bias=0.0,
        writes=(("ADDR_B1_LO+15", 0.22 / S),),
    ))

    # Unit 1: ADDR_B1_HI[15] = 1 (byte 1 = 0xff hi nibble).
    rules.append(multi_way_and_rule(
        name="addr_b1_hi_set",
        conditions=common_conditions,
        threshold=threshold,
        gate="CONST",
        gate_weight=1.0,
        gate_bias=0.0,
        writes=(("ADDR_B1_HI+15", 0.22 / S),),
    ))

    # Units 2-5: BP=0xf8 cascade. Gate is a 2-way AND over
    # ADDR_B0_LO[0] (post-shift high signal) and ADDR_B0_HI[15] (BP top
    # nibble) with b_gate=-15.0 so legitimate BP=0xf8 fires (LO[0]~15,
    # HI[15]~3 -> 18 - 15 = 3 > 0) while BP=0xf0 does not (LO[0]~0).
    cascade_writes: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = (
        # (rule_suffix, writes)
        ("cascade_b0_hi", (
            ("ADDR_B0_HI+15", -0.67 / S),
            ("ADDR_B0_HI+0",   0.67 / S),
        )),
        ("cascade_b1_lo", (
            ("ADDR_B1_LO+15", -0.5 / S),
            ("ADDR_B1_LO+0",  0.5 / S),
        )),
        ("cascade_b1_hi", (
            ("ADDR_B1_HI+15", -0.5 / S),
            ("ADDR_B1_HI+0",  0.5 / S),
        )),
        ("cascade_b2_lo", (
            ("ADDR_B2_LO+1", 0.67 / S),
        )),
    )
    for suffix, writes in cascade_writes:
        rules.append(multi_way_and_rule(
            name=f"addr_cascade_{suffix}",
            conditions=common_conditions,
            threshold=threshold,
            gate="ADDR_B0_LO+0",
            gate_weight=1.0,
            gate_terms=(("ADDR_B0_HI+15", 1.0),),
            gate_bias=-15.0,
            writes=writes,
        ))

    return tuple(rules)


def _marker_suppress_rules(S: float) -> tuple[FFNRule, ...]:
    """``NEXT_*`` marker-suppression band (7 units).

    Mirrors :func:`vm_step._set_layer9_marker_suppress`. One unit per
    NEXT_* dim (PC/AX/SP/BP/STACK0/MEM/SE) suppresses the natural
    OUTPUT_LO[0..15] and OUTPUT_HI[0..15] values from the multibyte
    routing whenever the corresponding NEXT_* flag is high (~1.37 from
    L8). The unit's up branch is sharply tuned with raw
    ``W_up[NEXT]=100`` / ``b_up=-80`` (i.e. NOT S-scaled), so the rule
    declares ``conditions=((NEXT_*, 100/S),)`` and ``threshold=80/S`` to
    reproduce the same un-scaled W_up cells after the ``S`` rescaling in
    the lowerer. The gate path uses the same NEXT_* dim as gate with
    raw ``W_gate=5.0`` / ``b_gate=-3.0`` (no scaling). The 32 write
    cells (16 OUTPUT_LO + 16 OUTPUT_HI_THIS_STEP) all carry raw weight
    ``-1.0``.
    """

    next_dims = (
        "NEXT_PC",
        "NEXT_AX",
        "NEXT_SP",
        "NEXT_BP",
        "NEXT_STACK0",
        "NEXT_MEM",
        "NEXT_SE",
    )
    rules: list[FFNRule] = []
    for next_dim in next_dims:
        writes: list[tuple[str, float]] = []
        for k in range(16):
            writes.append((f"OUTPUT_LO+{k}", -1.0))
            writes.append((f"OUTPUT_HI_THIS_STEP+{k}", -1.0))
        rules.append(multi_way_and_rule(
            name=f"marker_suppress_{next_dim.lower()}",
            conditions=((next_dim, 100.0 / S),),
            threshold=80.0 / S,
            gate=next_dim,
            gate_weight=5.0,
            gate_bias=-3.0,
            writes=tuple(writes),
        ))
    return tuple(rules)


def _alu_rules(S: float) -> tuple[FFNRule, ...]:
    """Full ordered ``FFNRule`` sequence for ``layer9_alu`` (3405 units).

    Concatenates every L9 ALU sub-stage in the order declared by
    :data:`_L9_ALU_UNIT_LAYOUT` so a single ``start_unit=0`` lowering
    via :func:`Primitives.lower_ffn_rules` reproduces the legacy
    ``_set_layer9_alu`` + ``_set_layer9_marker_suppress`` byte-for-byte.

    The bundled grouping (5 hi-nibble ALU bands, the CMP family, the
    carry/borrow propagators, the ALU-clear pair, the BP+8 / ADDR_B1 /
    cascade address fixups, and the NEXT_* marker-suppression band)
    matches the legacy helper's monotonic ``unit`` walk so the pinned
    unit slots in :data:`_L9_ALU_UNIT_LAYOUT` stay in lock-step with the
    rule order.
    """

    return (
        _add_hi_nibble_rules(S)
        + _lea_hi_nibble_rules(S)
        + _adj_hi_nibble_rules(S)
        + _sub_hi_nibble_rules(S)
        + _layer9_ent_hi_nibble_rules(S)
        + _layer9_cmp_rules(S)
        + _add_carry_out_rules(S)
        + _sub_borrow_out_rules(S)
        + _alu_clear_rules(S)
        + _layer9_bp_plus8_shift_rules(S)
        + _addr_b1_set_and_cascade_rules(S)
        + _marker_suppress_rules(S)
    )


def _alu_ir(S: float = 100.0) -> CompilerIR:
    """Build the declarative ``CompilerIR`` exposed by ``layer9_alu``."""
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_alu_rules(S))
    return ir


def make_layer9_alu_op(alu_mode: str = "lookup") -> Operation:
    """L9 FFN: ADD/SUB hi nibble + bitwise ops byte 0, plus marker suppression.

    Originally two inline calls inside ``set_vm_weights`` (in the
    ``alu_mode == 'lookup'`` branch):
        ``n9 = _set_layer9_alu(ffn9, S, BD)``
        ``_set_layer9_marker_suppress(ffn9, S, BD, n9)``

    Phase 6 Wave 4C migration: the entire 3405-unit weight surface is now
    declared via :func:`_alu_rules` -- a concatenation of 12
    sub-stage rule helpers (ADD/LEA/ADJ/SUB/ENT hi nibble, CMP family,
    ADD carry-out, SUB borrow-out, ALU LO/HI clear, BP+8 shift, ADDR_B1
    set + cascade, NEXT_* marker suppress). The bake_fn lowers them
    in one shot through :func:`Primitives.lower_ffn_rules` against the
    compiler-allocated dim positions; the legacy
    :func:`vm_step._set_layer9_alu` /
    :func:`vm_step._set_layer9_marker_suppress` helpers are no longer
    referenced from this op.

    Migrated as ``kind="block"`` pinned to ``layer_idx=9`` with
    ``migrated=True``; the inline call pair has been removed from
    ``set_vm_weights`` to avoid double-bake. Phase stays at 9. Fires in
    both lookup and efficient ALU modes -- the lookup-branch nesting was
    incidental and the rules are alu_mode-agnostic. In the efficient
    path the imperative :func:`_suppress_legacy_addsub_writes` post-pass
    still zeroes the ADD/SUB legacy output units and legacy carry rows;
    we keep it imperative because it operates on the already-lowered
    ``W_down`` cells as a downstream mutation rather than as a rule fan-out.
    """
    def bake(block, dim_positions, S):
        proxy = _as_setdim_proxy(dim_positions)

        # Per-bake FFN-unit allocator. Each L9 ALU sub-stage is pinned to
        # its existing offset so the rule lowering below -- which appends
        # units monotonically starting at ``start_unit=0`` -- lands
        # byte-identically with the legacy ``_set_layer9_alu`` cursor walk.
        allocator = _allocate_alu_units()
        marker_range = next(
            r for r in allocator.ranges()
            if r.op_name == "layer9_alu.marker_suppress"
        )
        marker_end = marker_range.start + marker_range.n_units
        # Make the allocator available for inspection / extension by
        # downstream tools (e.g. a future L9 op family claiming a free
        # gap). The block-level attribute mirrors the ``_l14_unit_counter``
        # convention used by sibling layers, but carries the allocator
        # object so the layout is structured, not just a monotonic int.
        block.ffn._l9_unit_allocator = allocator

        rules = _alu_rules(S)
        rule_dim_positions = Primitives.dim_positions_from_bd(
            proxy,
            Primitives.ffn_rule_dim_names(rules),
        )
        end_unit = Primitives.lower_ffn_rules(
            block.ffn,
            rules,
            rule_dim_positions,
            start_unit=0,
            S=S,
        )
        # Byte-identity guard: the rule lowering MUST land exactly where
        # the allocator's marker_suppress range ends. If the table drifts
        # from the rule order, this assertion fires before any downstream
        # consumer reads the L9 weights.
        assert end_unit == marker_end, (
            f"L9 ALU rule lowering ended at {end_unit}, allocator "
            f"expected {marker_end}"
        )
        if alu_mode == "efficient":
            _suppress_legacy_addsub_writes(block.ffn, proxy)

    # Dim-ownership claims (W_down output cells). Mirrors ``_set_layer9_alu``
    # in ``vm_step.py`` (3398 units) followed by ``_set_layer9_marker_suppress``
    # (7 units). The bake assigns hidden units in a fixed order from 0..3404
    # so the unit indices are deterministic per layer slot. We declare the
    # W_down output cells (partial-claims convention -- the per-output
    # ownership identifies which residual dims the op writes; input-side
    # W_up/W_gate selectors are left unclaimed to mirror the L14/L6 style).
    _claims: set = set()
    unit = 0
    # ADD hi nibble (carry_in in [0, 1]): 512 units -> W_down[OUTPUT_HI+result]
    for carry_in in (0, 1):
        for a in range(16):
            for b in range(16):
                result = (a + b + carry_in) % 16
                _claims.add((9, "ffn_W_down", str(unit), f"OUTPUT_HI+{result}"))
                unit += 1
    # LEA hi nibble: 512 units -> W_down[OUTPUT_HI+result]
    for carry_in in (0, 1):
        for a in range(16):
            for b in range(16):
                result = (a + b + carry_in) % 16
                _claims.add((9, "ffn_W_down", str(unit), f"OUTPUT_HI+{result}"))
                unit += 1
    # ADJ hi nibble: 512 units -> W_down[OUTPUT_HI+result]
    for carry_in in (0, 1):
        for a in range(16):
            for b in range(16):
                result = (a + b + carry_in) % 16
                _claims.add((9, "ffn_W_down", str(unit), f"OUTPUT_HI+{result}"))
                unit += 1
    # SUB hi nibble: 512 units -> W_down[OUTPUT_HI+result]
    for borrow_in in (0, 1):
        for a in range(16):
            for b in range(16):
                result = (a - b - borrow_in) % 16
                _claims.add((9, "ffn_W_down", str(unit), f"OUTPUT_HI+{result}"))
                unit += 1
    # ENT hi nibble: 512 units -> W_down[OUTPUT_HI+result]
    for borrow_in in (0, 1):
        for sp_hi in range(16):
            for imm_hi in range(16):
                result = (sp_hi - imm_hi - borrow_in) % 16
                _claims.add((9, "ffn_W_down", str(unit), f"OUTPUT_HI+{result}"))
                unit += 1
    # hi_eq: 16 units -> W_down[CMP+1]
    for _k in range(16):
        _claims.add((9, "ffn_W_down", str(unit), "CMP+1"))
        unit += 1
    # lo_eq: 16 units -> W_down[CMP+2]
    for _k in range(16):
        _claims.add((9, "ffn_W_down", str(unit), "CMP+2"))
        unit += 1
    # hi_lt: 120 units -> W_down[CMP+0]
    for a in range(16):
        for b in range(a + 1, 16):
            _claims.add((9, "ffn_W_down", str(unit), "CMP+0"))
            unit += 1
    # lo_lt: 120 units -> W_down[CMP+3]
    for a in range(16):
        for b in range(a + 1, 16):
            _claims.add((9, "ffn_W_down", str(unit), "CMP+3"))
            unit += 1
    # ADD hi-nibble carry-out: skips units where a+b+carry_in < 16.
    # 120 (carry_in=0) + 136 (carry_in=1) = 256 units -> W_down[CARRY+1]
    for carry_in in (0, 1):
        for a in range(16):
            for b in range(16):
                if a + b + carry_in < 16:
                    continue
                _claims.add((9, "ffn_W_down", str(unit), "CARRY+1"))
                unit += 1
    # SUB hi-nibble borrow-out: skips units with no borrow-out.
    # 120 (borrow_in=0: a<b) + 136 (borrow_in=1: a<=b) = 256 units -> W_down[CARRY+2]
    for borrow_in in (0, 1):
        for a in range(16):
            for b in range(16):
                if borrow_in == 0:
                    if a >= b:
                        continue
                else:
                    if a > b:
                        continue
                _claims.add((9, "ffn_W_down", str(unit), "CARRY+2"))
                unit += 1
    # ALU clearing LO: 16 units -> W_down[ALU_LO+k]
    for k in range(16):
        _claims.add((9, "ffn_W_down", str(unit), f"ALU_LO+{k}"))
        unit += 1
    # ALU clearing HI: 16 units -> W_down[ALU_HI+k]
    for k in range(16):
        _claims.add((9, "ffn_W_down", str(unit), f"ALU_HI+{k}"))
        unit += 1
    # BP+8 shift to ADDR_B0_LO: 16 units; each writes ADDR_B0_LO[k] (cancel)
    # and ADDR_B0_LO[(k+8)%16] (set).
    for k in range(16):
        new_k = (k + 8) % 16
        _claims.add((9, "ffn_W_down", str(unit), f"ADDR_B0_LO+{k}"))
        _claims.add((9, "ffn_W_down", str(unit), f"ADDR_B0_LO+{new_k}"))
        unit += 1
    # ADDR_B1=0xff at PC marker for LEV (2 units): ADDR_B1_LO[15], ADDR_B1_HI[15]
    _claims.add((9, "ffn_W_down", str(unit), "ADDR_B1_LO+15"))
    unit += 1
    _claims.add((9, "ffn_W_down", str(unit), "ADDR_B1_HI+15"))
    unit += 1
    # Cascade carry units for BP=0xfff8 + 8 = 0x10000:
    # Unit +0: clear ADDR_B0_HI[15], set ADDR_B0_HI[0]
    _claims.add((9, "ffn_W_down", str(unit), "ADDR_B0_HI+15"))
    _claims.add((9, "ffn_W_down", str(unit), "ADDR_B0_HI+0"))
    unit += 1
    # Unit +1: cancel ADDR_B1_LO[15], set ADDR_B1_LO[0]
    _claims.add((9, "ffn_W_down", str(unit), "ADDR_B1_LO+15"))
    _claims.add((9, "ffn_W_down", str(unit), "ADDR_B1_LO+0"))
    unit += 1
    # Unit +2: cancel ADDR_B1_HI[15], set ADDR_B1_HI[0]
    _claims.add((9, "ffn_W_down", str(unit), "ADDR_B1_HI+15"))
    _claims.add((9, "ffn_W_down", str(unit), "ADDR_B1_HI+0"))
    unit += 1
    # Unit +3: set ADDR_B2_LO[1]
    _claims.add((9, "ffn_W_down", str(unit), "ADDR_B2_LO+1"))
    unit += 1
    # _set_layer9_marker_suppress: 7 units (one per NEXT_* dim), each writes
    # W_down[OUTPUT_LO+k] and W_down[OUTPUT_HI+k] for k in 0..15.
    for _next_dim in (
        "NEXT_PC", "NEXT_AX", "NEXT_SP", "NEXT_BP",
        "NEXT_STACK0", "NEXT_MEM", "NEXT_SE",
    ):
        for k in range(16):
            _claims.add((9, "ffn_W_down", str(unit), f"OUTPUT_LO+{k}"))
            _claims.add((9, "ffn_W_down", str(unit), f"OUTPUT_HI+{k}"))
        unit += 1
    # Total expected unit index after bake: 3405 (matches ffn_units_used).
    _claims = frozenset(_claims)

    # Declarative IR exposed to symbolic execution / verifier tooling.
    # The ``efficient`` alu_mode skips the IR-only dispatch path because
    # the post-bake :func:`_suppress_legacy_addsub_writes` mutation
    # zeroes a subset of the lowered cells -- a step the IR does not
    # model. The bake_fn handles both modes correctly; only the
    # declarations-only path (compiler_ir lowering without bake_fn) needs
    # to be gated.
    compiler_ir = _alu_ir() if alu_mode == "lookup" else None

    return Operation(
        name="layer9_alu",
        # Phase 8.A: CARRY_PREV_STEP marks the CARRY read as cross-step
        # relative to the L10 CARRY writers (layer10_carry_relay,
        # layer10_carry_relay_bake, l10_post_ops_combined) that fire AFTER
        # L9 in the same step. The L9 ALU consumes the byte-level carry-in
        # bit (``CARRY[0]``) from the previous step's L10 carry relay write
        # to discriminate ADD/SUB hi-nibble outputs (carry_in=0 vs 1) at
        # subsequent AX byte positions. Same numeric position as CARRY so
        # baked weight cells are byte-identical. Breaks the 3 L10 -> L9
        # back-edges (CARRY rank 9 in the latest scheduler SCC audit).
        # Phase 9.B (ALU_HI SCC rename): ALU_HI -> ALU_HI.*.-1 marks the
        # read as SSA cross-step. L10 stack0_byte_relay{,_bake} stage
        # ALU_HI for the NEXT step's L9 consumption (PSH/SI/SC stack-byte
        # staging); same-step fresh ALU_HI from L7 operand_gather is
        # still observed via consumes_fresh (ALU_HI@AX_byte0) declared
        # below. Same numeric slot via SSA alias; byte-identical bake.
        # Breaks 2 L10 -> L9 back-edges.
        reads={"MARK_AX", "MARK_PC", "ALU_HI.*.-1", "AX_CARRY_HI", "FETCH_HI",
               "CARRY.*.-1",
               # HAS_SE: the ENT hi-nibble band (2026-06-11 step-1 ENT-AX
               # 0xf0 leak fix) blocks on HAS_SE so it fires only on the
               # first ENT step. Declared so the compact per-op audit layout
               # allocates HAS_SE inside d_model.
               "HAS_SE",
               "OP_ADD", "OP_SUB", "OP_OR", "OP_XOR", "OP_AND",
               "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
               # Phase 9.B (ALU_LO SCC rename): ALU_LO -> ALU_LO.*.-1 marks
               # the read as SSA cross-step. L10/L16 ALU_LO writers stage
               # NEXT-step residuals; same-step fresh ALU_LO at AX_byte0
               # is observed via the existing consumes_fresh-style invariants
               # (the bake reads ALU_LO directly through the dim slot;
               # the SSA alias shares the numeric column).
               "ALU_LO.*.-1", "AX_CARRY_LO",
               # V2/G7 LEV detector: in-step topology edge from
               # lev_detector_head (phase=8.06) replaces the cross-step
               # requires["after"]=layer16_lev_routing below.
               "PC_VIA_LEV_DETECTOR_LO"},
        writes={"OUTPUT_HI", "CMP", "OUTPUT_LO", "CARRY"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir=compiler_ir,
        declarative_authority="spec_generated",
        # Phase 8.A.4 retry: layer_idx=9 literal dropped. ``target_op_name``
        # binds this block op to the layer of ``layer9_marker_suppress``
        # (kind="ffn", L9 anchor pinned via ``requires["after"]: layer8_alu``).
        target_op_name="layer9_marker_suppress",
        migrated=True,
        claims=_claims,
        # Staleness invariants: the L9 ALU consumes ALU_HI as operand A hi
        # nibble at the AX marker. Produced by ``layer7_operand_gather`` (L7
        # head 0 + head 1, phase=7) at AX byte 0.
        # Phase 9.D: ALU_LO cycle-graph constraint satisfied by the
        # PC_VIA_LEV_DETECTOR_LO read above (lev_detector_head phase=8.06
        # is in-step producer). Previous: requires={"after":
        # "layer16_lev_routing"}. See CONTROL_FLOW_DETECTOR_HEADS.md §2.4.
        # ``_set_layer9_alu`` writes the ADD/LEA/SUB/AND/OR/XOR/CMP/etc.
        # cross-product cluster (~3398 units), and the bake chains into
        # ``_set_layer9_marker_suppress`` for 7 more NEXT_* suppression
        # units. Cumulative L9 FFN max: 3405. No other op writes to L9
        # FFN so this op holds the per-layer width annotation.
        ffn_units_used=3405,
        smoke_tests={
            "TestSmoke32Bit::test_add_16bit",
            "TestSmoke32Bit::test_and_16bit",
            "TestSmoke32Bit::test_or_16bit",
            "TestSmoke32Bit::test_sub_16bit",
            "TestSmoke32Bit::test_xor_16bit",
            "TestSmokeBasic::test_add_basic",
            "TestSmokeBasic::test_sub_basic",
            "TestSmokeBitwise::test_and_basic",
            "TestSmokeBitwise::test_or_basic",
            "TestSmokeBitwise::test_xor_basic",
            "TestSmokeComparison::test_eq_false",
            "TestSmokeComparison::test_eq_true",
            "TestSmokeComparison::test_ge_true",
            "TestSmokeComparison::test_gt_true",
            "TestSmokeComparison::test_le_true",
            "TestSmokeComparison::test_lt_true",
            "TestSmokeComparison::test_ne_true",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
    )


def _suppress_legacy_addsub_writes(ffn, BD) -> None:
    """Let the efficient add/sub block own byte carry/borrow state.

    L8/L9 legacy nibble logic leaves ``CARRY[0]`` as an intra-byte
    carry/borrow signal. The efficient AddSub5StageBlock that now runs before
    L9 computes the full byte result and the byte-level ``CARRY[1]/CARRY[2]``
    needed by the later carry-propagation post-ops. L9's legacy ADD/SUB
    hi-nibble units see amplified raw one-hot operands at the AX marker in the
    efficient path and can swamp the already-correct marker result. Zero just
    those ADD/SUB legacy output units and the legacy carry rows; comparisons,
    LEA/ADJ/ENT, and bitwise units remain intact.
    """

    # _set_layer9_alu unit layout:
    #   0..511: ADD hi nibble
    #   512..1023: LEA hi nibble
    #   1024..1535: ADJ hi nibble
    #   1536..2047: SUB hi nibble
    add_units = slice(0, 512)
    sub_units = slice(1536, 2048)
    ffn.W_down.data[BD.OUTPUT_HI:BD.OUTPUT_HI + 16, add_units] = 0.0
    ffn.W_down.data[BD.OUTPUT_HI:BD.OUTPUT_HI + 16, sub_units] = 0.0
    ffn.W_down.data[BD.CARRY + 1, :] = 0.0
    ffn.W_down.data[BD.CARRY + 2, :] = 0.0


def make_lev_addr_relay_op() -> Operation:
    """L9 attention head 0: BP byte 0 → ADDR_B0 at SP marker for LEV.

    Originally an inline call inside ``set_vm_weights`` (in the
    ``alu_mode == 'lookup'`` branch):
        ``_set_layer9_lev_addr_relay(attn9, S, BD, HD)``

    Migrated as ``kind="block"`` pinned to ``layer_idx=9`` with
    ``migrated=True``: the inline call has been removed to avoid
    double-bake. Phase=9.0 to preserve ordering with sibling
    ``layer9_lev_bp_to_pc_relay`` (phase=9.1). Fires in both lookup
    and efficient ALU modes — the helper performs identical setup
    regardless of alu_mode, and the model is built once.

    Also sets ``alibi_slopes[0] = 0.2`` (shallow slope for d=29 relay
    from SP marker back to previous BP byte 0); previously set inline
    alongside the legacy bake call.
    """
    def bake(block, dim_positions, S):
        attn = block.attn
        # Per-bake attention-head allocator with the full L9 head layout
        # pinned. Looking up the relay head by name keeps its index
        # identical to the legacy ``head_idx=0`` literal without baking
        # the integer into the call site.
        allocator = _allocate_alu_attention_heads()
        attn._l9_head_allocator = allocator
        relay_head_idx = next(
            h.head_idx for h in allocator.heads()
            if h.op_name == "layer9_lev_addr_relay"
        )
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            # head 0: shallow slope for d=29 relay. Index from the
            # allocator so the slot stays in sync with the spec below.
            attn.alibi_slopes[relay_head_idx] = 0.2
        HD = attn.W_q.shape[0] // attn.num_heads
        Primitives.generate_attention_head(
            attn,
            _lev_addr_relay_head_spec(_as_setdim_proxy(dim_positions)),
            HD,
        )

    # Dim-ownership claims: L9 attn head 0 LEV addr relay.
    #   W_v[0*HD + 1 + k, CLEAN_EMBED_LO + k]    for k=0..15
    #   W_v[0*HD + 17 + k, CLEAN_EMBED_HI + k]   for k=0..15
    #   W_o[ADDR_B0_LO + k, 0*HD + 1 + k]         for k=0..15
    #   W_o[ADDR_B0_HI + k, 0*HD + 17 + k]        for k=0..15
    _claims = set()
    for k in range(16):
        _claims.add((9, "attn_W_v", f"0_{1 + k}", f"CLEAN_EMBED_LO+{k}"))
        _claims.add((9, "attn_W_v", f"0_{17 + k}", f"CLEAN_EMBED_HI+{k}"))

    return Operation(
        name="layer9_lev_addr_relay",
        # Declaration audit (2026-06-05): added CONST -- the GATE Q slot
        # _lev_addr_relay_head_spec uses on Q[33, BD.CONST] /
        # K[33, BD.CONST] / Q[0, BD.CONST] gating writes.
        reads={"MARK_SP", "OP_LEV", "L1H1", "BYTE_INDEX_0", "CONST",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"ADDR_B0_LO", "ADDR_B0_HI"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir_factory=_lev_addr_relay_ir,
        # Phase 8.A.4 retry: layer_idx=9 literal dropped. ``target_op_name``
        # binds this block op to the layer of ``layer9_marker_suppress``
        # (kind="ffn", L9 anchor pinned via ``requires["after"]: layer8_alu``).
        target_op_name="layer9_marker_suppress",
        migrated=True,
        declarative_authority="spec_generated",
        claims=_claims,
        smoke_tests={"TestSmokeFunctionCall::test_simple_function"},
        spec_section="BLOG_SPEC.md#function-calls",
    )


def make_lev_bp_to_pc_relay_op() -> Operation:
    """L9 attention head 1: BP byte 0 → ADDR_B0 at PC marker for LEV return.

    Originally an inline call inside ``set_vm_weights`` (in the
    ``alu_mode == 'lookup'`` branch):
        ``_set_layer9_lev_bp_to_pc_relay(attn9, S, BD, HD)``

    Migrated as ``kind="block"`` pinned to ``layer_idx=9`` with
    ``migrated=True``: the inline call has been removed to avoid
    double-bake. Phase=9.1 so this op runs AFTER
    ``layer9_lev_addr_relay`` (phase=9.0), matching the legacy
    in-set_vm_weights ordering. Fires in both lookup and efficient
    ALU modes — the helper performs identical setup regardless of
    alu_mode.

    Also sets ``alibi_slopes[1] = 0.5`` (BP→PC relay slope for d=15
    tokens); previously set inline alongside the legacy bake call.
    """
    def bake(block, dim_positions, S):
        attn = block.attn
        # Per-bake attention-head allocator with the full L9 head layout
        # pinned. Resolving the BP→PC relay head by name reproduces the
        # legacy ``head_idx=1`` literal without depending on the
        # addr-relay op having already populated ``attn._l9_head_allocator``.
        allocator = _allocate_alu_attention_heads()
        attn._l9_head_allocator = allocator
        relay_head_idx = next(
            h.head_idx for h in allocator.heads()
            if h.op_name == "layer9_lev_bp_to_pc_relay"
        )
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            # head 1: BP→PC relay for LEV (d=15 tokens). Index from the
            # allocator so the slot stays in sync with the spec below.
            attn.alibi_slopes[relay_head_idx] = 0.5
        HD = attn.W_q.shape[0] // attn.num_heads
        Primitives.generate_attention_head(
            attn,
            _lev_bp_to_pc_relay_head_spec(_as_setdim_proxy(dim_positions)),
            HD,
        )

    # Dim-ownership claims: L9 attn head 1 LEV BP→PC relay.
    #   W_v[1*HD + 1 + k, CLEAN_EMBED_LO + k]    for k=0..15
    #   W_v[1*HD + 17 + k, CLEAN_EMBED_HI + k]   for k=0..15
    #   W_o[ADDR_B0_LO + k, 1*HD + 1 + k]         for k=0..15
    #   W_o[ADDR_B0_HI + k, 1*HD + 17 + k]        for k=0..15
    _claims = set()
    for k in range(16):
        _claims.add((9, "attn_W_v", f"1_{1 + k}", f"CLEAN_EMBED_LO+{k}"))
        _claims.add((9, "attn_W_v", f"1_{17 + k}", f"CLEAN_EMBED_HI+{k}"))

    return Operation(
        name="layer9_lev_bp_to_pc_relay",
        # Declaration audit (2026-06-05): added CONST -- the GATE Q slot
        # _lev_bp_to_pc_relay_head_spec uses on Q[33, BD.CONST] /
        # K[33, BD.CONST] / Q[0, BD.CONST] gating writes.
        reads={"MARK_PC", "OP_LEV", "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
               "L1H1", "BYTE_INDEX_0", "CONST"},
        writes={"ADDR_B0_LO", "ADDR_B0_HI"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir_factory=_lev_bp_to_pc_relay_ir,
        # Phase 8.A.4 retry: layer_idx=9 literal dropped. ``target_op_name``
        # binds this block op to the layer of ``layer9_marker_suppress``
        # (kind="ffn", L9 anchor pinned via ``requires["after"]: layer8_alu``).
        target_op_name="layer9_marker_suppress",
        migrated=True,
        declarative_authority="spec_generated",
        claims=_claims,
        smoke_tests={"TestSmokeFunctionCall::test_simple_function"},
        spec_section="BLOG_SPEC.md#function-calls",
    )


def _lev_addr_relay_ir(dim_positions, HD) -> CompilerIR:
    BD = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.append(_lev_addr_relay_head_spec(BD))
    return ir


def _lev_bp_to_pc_relay_ir(dim_positions, HD) -> CompilerIR:
    BD = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.append(_lev_bp_to_pc_relay_head_spec(BD))
    return ir


def _lev_addr_relay_head_spec(BD) -> DeclarativeAttentionHeadSpec:
    """Declarative L9 head 0: previous BP byte0 -> ADDR_B0 at SP marker."""

    L = 50.0
    BP_I = 3
    GATE = 33
    v = []
    o = []
    for k in range(16):
        v.append(AP(1 + k, BD.CLEAN_EMBED_LO + k, 3.0))
        v.append(AP(17 + k, BD.CLEAN_EMBED_HI + k, 3.0))
        o.append(AO(BD.ADDR_B0_LO + k, 1 + k, 1.0))
        o.append(AO(BD.ADDR_B0_HI + k, 17 + k, 1.0))
    return DeclarativeAttentionHeadSpec(
        # Pull the head index from the shared L9 layout rather than
        # baking in a ``head_idx=0`` literal here. Both the bake and IR
        # paths consult the same source of truth, so renumbering the
        # layout in one place stays consistent across every consumer.
        head_idx=_alu_head_idx("layer9_lev_addr_relay"),
        q=(
            AP(0, BD.MARK_SP, L),
            AP(0, BD.OP_LEV, L / 5),
            AP(0, BD.CONST, -2 * L),
            AP(GATE, BD.MARK_SP, L),
            AP(GATE, BD.CONST, -L / 2),
        ),
        k=(
            AP(0, BD.L1H1 + BP_I, L),
            AP(0, BD.BYTE_INDEX_0, L),
            AP(GATE, BD.CONST, L),
        ),
        v=tuple(v),
        o=tuple(o),
    )


def _lev_bp_to_pc_relay_head_spec(BD) -> DeclarativeAttentionHeadSpec:
    """Declarative L9 head 1: previous BP byte0 -> ADDR_B0 at PC marker."""

    L = 50.0
    BP_I = 3
    GATE = 33
    v = []
    o = []
    for k in range(16):
        v.append(AP(1 + k, BD.CLEAN_EMBED_LO + k, 3.0))
        v.append(AP(17 + k, BD.CLEAN_EMBED_HI + k, 3.0))
        o.append(AO(BD.ADDR_B0_LO + k, 1 + k, 1.0))
        o.append(AO(BD.ADDR_B0_HI + k, 17 + k, 1.0))
    return DeclarativeAttentionHeadSpec(
        # Pull the head index from the shared L9 layout rather than
        # baking in a ``head_idx=1`` literal here. Both the bake and IR
        # paths consult the same source of truth, so renumbering the
        # layout in one place stays consistent across every consumer.
        head_idx=_alu_head_idx("layer9_lev_bp_to_pc_relay"),
        q=(
            AP(0, BD.MARK_PC, L),
            AP(0, BD.OP_LEV, L / 5),
            AP(0, BD.CONST, -2 * L),
            AP(GATE, BD.MARK_PC, L),
            AP(GATE, BD.CONST, -L / 2),
        ),
        k=(
            AP(0, BD.L1H1 + BP_I, L),
            AP(0, BD.BYTE_INDEX_0, L),
            AP(GATE, BD.CONST, L),
        ),
        v=tuple(v),
        o=tuple(o),
    )


def make_format_string_fetch_head_op(enable_conversational_io: bool = False) -> Operation:
    """L9 attention head 0: fetch byte from format string at FORMAT_PTR+POS.

    Originally an inline call in ``set_vm_weights`` (nested under
    ``alu_mode == 'lookup'`` + ``enable_conversational_io``):
        ``_set_format_string_fetch_head(attn9, S, BD, HD)``
        plus ``attn9.alibi_slopes.fill_(0.5)``.

    Migrated as ``kind="block"`` pinned to ``layer_idx=9`` with
    ``migrated=True``. Registered unconditionally; the bake is a no-op
    when ``enable_conversational_io`` is False. The original lookup-mode
    nesting was incidental — the helper writes only attn head 0 weights
    and has no dependency on lookup-mode-specific weights, so this op
    fires regardless of alu_mode whenever the flag is set. Phase=9.5 so
    this runs AFTER ``layer9_lev_addr_relay`` (phase=9.0) and
    ``layer9_lev_bp_to_pc_relay`` (phase=9.1); the ``fill_(0.5)`` call
    intentionally clobbers slopes[0] and [1] that those ops set, matching
    legacy ordering (the legacy convo-io block ran ``fill_(0.5)`` AFTER
    the L9 LEV setup as well).
    """
    def bake(block, dim_positions, S):
        del S
        if not enable_conversational_io:
            return
        attn = block.attn
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes.fill_(0.5)
        HD = attn.W_q.shape[0] // attn.num_heads
        Primitives.generate_attention_head(
            attn,
            _format_string_fetch_head_spec(_as_setdim_proxy(dim_positions)),
            HD,
        )

    return Operation(
        name="format_string_fetch_head",
        # Phase 9.B (IO_IN_OUTPUT_MODE SCC rename): IO_IN_OUTPUT_MODE ->
        # IO_IN_OUTPUT_MODE.*.-1 marks the read as SSA cross-step.
        # null_terminator_detection (phase 10.6) stages the value for the
        # NEXT step's L9 format-fetch gating. Same numeric slot via SSA
        # alias; byte-identical bake. Breaks the null_terminator_detection
        # -> format_string_fetch_head IO_IN_OUTPUT_MODE back-edge (SCC #3).
        reads={"IO_IN_OUTPUT_MODE.*.-1", "FORMAT_PTR_LO", "FORMAT_PTR_HI",
               "ADDR_KEY", "EMBED_LO", "EMBED_HI"},
        writes={"OUTPUT_BYTE_LO", "OUTPUT_BYTE_HI"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir_factory=(
            _format_string_fetch_head_ir
            if enable_conversational_io
            else _empty_compiler_ir_factory
        ),
        declarative_authority="spec_generated",
        # Phase 8.A.4 retry: layer_idx=9 literal dropped. ``target_op_name``
        # binds this block op to the layer of ``layer9_marker_suppress``
        # (kind="ffn", L9 anchor pinned via ``requires["after"]: layer8_alu``).
        target_op_name="layer9_marker_suppress",
        migrated=True,
        # Phase 8.A targeted (SCC audit step 7): pin the ADDR_KEY K-side
        # reader to ``layer4_pc_relay`` so the dim-flow analyser's R-OH-2
        # rule suppresses the spurious L7/L14 ADDR_KEY back-edges into
        # this op. The K-side matches format-string code byte ADDR_KEYs
        # (stable from embedding); no same-step dep on the L14 writers.
        requires={"after": "layer4_pc_relay"},
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#how-bytecode-is-passed-to-the-network",
    )


def _format_string_fetch_head_ir(dim_positions, HD) -> CompilerIR:
    del HD
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.append(_format_string_fetch_head_spec(proxy))
    return ir


def _format_string_fetch_head_spec(BD) -> DeclarativeAttentionHeadSpec:
    L = 15.0
    q = [AP(0, BD.IO_IN_OUTPUT_MODE, L)]
    k = []
    v = []
    o = []
    for idx in range(16):
        q.append(AP(1 + idx, BD.FORMAT_PTR_LO + idx, 1.0))
        q.append(AP(17 + idx, BD.FORMAT_PTR_HI + idx, 1.0))
        k.append(AP(1 + idx, BD.ADDR_KEY + idx, L))
        k.append(AP(17 + idx, BD.ADDR_KEY + 16 + idx, L))
        v.append(AP(1 + idx, BD.EMBED_LO + idx, 1.0))
        v.append(AP(17 + idx, BD.EMBED_HI + idx, 1.0))
        o.append(AO(BD.OUTPUT_BYTE_LO + idx, 1 + idx, 1.0))
        o.append(AO(BD.OUTPUT_BYTE_HI + idx, 17 + idx, 1.0))
    return DeclarativeAttentionHeadSpec(
        head_idx=0,
        q=tuple(q),
        k=tuple(k),
        v=tuple(v),
        o=tuple(o),
    )


def make_alibi_mem_attn_op(enable: bool = False) -> Operation:
    """L9 attention head 2: ALiBi-based memory-propagation attention.

    PROOF-OF-CONCEPT for the directive "all of the memory conversions should
    be alibi attention stuff" (2026-05-11). Replaces (eventually) the
    runner-side ``_inject_mem_store`` / ``_mem_history`` shadow-memory
    pipeline with attention that gathers PSH'd values from prior step
    OUTPUT positions via ALiBi recency bias.

    Design
    ------
    Q (at MEM val byte 0 positions during LI/LC/POP with ADDR_KEY = SP-1):
      - W_q[head, BD.OP_LI_RELAY] = L
      - W_q[head, BD.OP_LC_RELAY] = L
      - W_q[head, BD.CMP+3] = L              (POP group flag at STACK0)
      - W_q[head, BD.MEM_VAL_B0] = L         (gate to val byte 0)
      - W_q[head, BD.ADDR_KEY + k] = scale * (bit_val)   (address bits)
      - W_q[head, BD.CONST] = -threshold     (suppress non-fire positions)

    K (at PSH/SI/SC OUTPUT positions — STACK0 value byte 0):
      - W_k[head, BD.PSH_AT_SP] = L          (only match PSH output positions)
      - W_k[head, BD.ADDR_KEY + k] = scale * (bit_val)
      - W_k[head, BD.MEM_STORE] = L          (only match store entries)
      - W_k[head, BD.CONST] = -threshold

    V (PSH-output STACK0's CLEAN_EMBED value, copied to OUTPUT at the
      current load position):
      - W_v[head*HD + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0  for k in 0..15
      - W_v[head*HD + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
      - W_o[BD.OUTPUT_LO + k, head*HD + 1 + k] = 1.0
      - W_o[BD.OUTPUT_HI + k, head*HD + 17 + k] = 1.0

    ALiBi slope tuning
    ------------------
    Score budget at target Q (load) attending to PSH outputs:
      - Address match (24-bit, scale=10): +300 at exact match, -300 random
      - PSH gate match: +L^2/HD (target+PSH=+L*L/HD, target+non-PSH=-)
      - ALiBi penalty: -slope * |i - j|  (one VM step = 35 tokens)

    With slope=0.5, going back 1 step costs -0.5*35 = -17.5; with two
    PSHes at the same SP, the more recent one wins by +17.5 score (>>
    softmax noise threshold). Going back 10 steps costs -175 which is
    below the +300 address-match contribution, so legitimate matches still
    fire across the typical KV-cache window. The slope should be tuned
    higher if cross-PSH leak from old values becomes a problem; lower if
    long-range matches fail.

    Status
    ------
    ``enable=False`` by default: the op IS registered (so the dep graph and
    layer_idx gates see it), but the bake is a no-op. This keeps existing
    tests byte-identical. Set ``enable=True`` to activate the head and
    flip ``alibi_slopes[2]`` from 0 to the tuned value.

    Concrete next steps for full Phase 2 PSH/POP support
    -----------------------------------------------------
    1. Bake ADDR_KEY at PSH/SI/SC OUTPUT positions (STACK0 value byte 0
       carries SP-derived address) — new FFN at L8 or earlier, ~50 LoC.
    2. Verify K-side ADDR_KEY at PSH output matches what the load-side Q
       expects. Today ADDR_KEY only lives at code byte positions
       (``_add_code_addr_keys``) and at MEM section val-byte positions
       (``_inject_mem_metadata``); we need it at PSH-step STACK0 too.
    3. Set ``enable=True`` here and ``alibi_slopes[2] = 0.5``.
    4. Drop the runner-side ``_inject_mem_section`` / ``_track_mem_access``
       calls once attention-only mode is stable.

    Time budget proof-of-concept: registers the op (passes layer_idx gate)
    and demonstrates the design pattern without disturbing existing tests.
    """
    def bake(block, dim_positions, S):
        # When disabled: op is a no-op. The head's alibi_slopes[2] stays at
        # its module-init default (a small power-of-2 value from
        # AutoregressiveAttention.__init__), but the head's W_q/W_k/W_v/W_o
        # weights are all zero — so attention output for head 2 is 0 (V is 0)
        # and W_o for head 2 dims is 0 → no contribution to the residual.
        if not enable:
            return

        from ...vm_step import _SetDim as BD_DEFAULT
        attn = block.attn
        HD = attn.W_q.shape[0] // attn.num_heads
        # Per-bake attention-head allocator with the full L9 head layout
        # pinned. Looking up the ALiBi mem-attn head by name keeps its
        # index identical to the legacy ``head = 2`` literal without
        # baking the integer into the call site. Stash on the attn
        # block so downstream tooling can inspect the layout.
        allocator = _allocate_alu_attention_heads()
        attn._l9_head_allocator = allocator
        head = next(
            h.head_idx for h in allocator.heads()
            if h.op_name == "layer9_alibi_mem_attn"
        )
        base = head * HD

        # Slope tuned to favor most-recent matching PSH within a typical
        # 4096-token (~117-step) KV-cache window. See docstring for analysis.
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes[head] = 0.5

        BD = _as_setdim_proxy(dim_positions)
        L = 50.0
        scale = 10.0

        # === Q side: fire at MEM val byte 0 position during LI/LC/POP ===
        # MEM_VAL_B0 is set at val-byte-0 positions in every MEM section.
        # During load ops, that position is the natural "I am about to read
        # a memory value" anchor.
        attn.W_q[base, BD.MEM_VAL_B0] = L
        attn.W_q[base, BD.OP_LI_RELAY] = L / 5  # OP relay gates the load
        attn.W_q[base, BD.OP_LC_RELAY] = L / 5
        attn.W_q[base, BD.CMP + 3] = L / 5      # POP group
        attn.W_q[base, BD.CONST] = -L * 1.5    # threshold

        # === K side: match PSH-output STACK0 positions ===
        # PSH_AT_SP is set at SP/STACK0 positions during PSH steps.
        # MEM_STORE is set at MEM val-bytes (and at PSH-step's STACK0
        # output, via L6 head 6 / L7 head 7 broadcast in step W).
        attn.W_k[base, BD.PSH_AT_SP] = L
        attn.W_k[base, BD.MEM_STORE] = L / 2
        attn.W_k[base, BD.CONST] = -L * 0.5

        # === Address matching: 24 binary bits across 3 address bytes ===
        # Both sides use the same ADDR_KEY encoding. Address match gives
        # +300 to score (per-dim contribution after /sqrt(HD)). Mismatch
        # gives ~0 (random) to -300 (anti-match).
        addr_dim = 4
        addr_bases = [
            (BD.ADDR_B0_LO, BD.ADDR_B0_HI),
            (BD.ADDR_B1_LO, BD.ADDR_B1_HI),
            (BD.ADDR_B2_LO, BD.ADDR_B2_HI),
        ]
        for ab_lo, ab_hi in addr_bases:
            for nibble_base in [ab_lo, ab_hi]:
                for bit in range(4):
                    for k in range(16):
                        bit_val = 2 * ((k >> bit) & 1) - 1
                        attn.W_q[base + addr_dim, nibble_base + k] = scale * bit_val
                        attn.W_k[base + addr_dim, nibble_base + k] = scale * bit_val
                    addr_dim += 1

        # === V/O: copy CLEAN_EMBED bytes → OUTPUT at load position ===
        # The PSH step's STACK0 has CLEAN_EMBED_LO/HI = AX value at time of
        # PSH. Carrying it to OUTPUT at the load position writes the value
        # into the load result.
        scale_v = 1.0
        for k in range(16):
            attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = scale_v
            attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = scale_v
        for k in range(16):
            attn.W_o[BD.OUTPUT_LO + k, base + 1 + k] = 1.0
            attn.W_o[BD.OUTPUT_HI + k, base + 17 + k] = 1.0

    # Dim-ownership claims: L9 attn head 2 ALiBi mem attn.
    # When enable=True, V slots 1..32 carry CLEAN_EMBED to OUTPUT.
    _claims = set()
    if enable:
        for k in range(16):
            _claims.add((9, "attn_W_v", f"2_{1 + k}", f"CLEAN_EMBED_LO+{k}"))
            _claims.add((9, "attn_W_v", f"2_{17 + k}", f"CLEAN_EMBED_HI+{k}"))

    return Operation(
        name="layer9_alibi_mem_attn",
        # Phase 8.G.6: phase=9.2 dropped (had trailing comment); the
        # ``target_op_name`` below binds the layer and
        # ``requires["after"]: layer4_pc_relay`` supplies the strict-mode
        # dep edge.
        reads={"MEM_VAL_B0", "OP_LI_RELAY", "OP_LC_RELAY", "CMP", "CONST",
               "PSH_AT_SP", "MEM_STORE", "ADDR_KEY",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake if not enable else None,
        # Phase 8.A.4 retry: layer_idx=9 literal dropped. ``target_op_name``
        # binds this block op to the layer of ``layer9_marker_suppress``
        # (kind="ffn", L9 anchor pinned via ``requires["after"]: layer8_alu``).
        target_op_name="layer9_marker_suppress",
        migrated=True,
        claims=_claims,
        # Phase 8.A targeted (SCC audit step 7): pin the ADDR_KEY reader to
        # ``layer4_pc_relay`` so R-OH-2 in the dim-flow analyser
        # suppresses spurious back-edges from later ADDR_KEY writers
        # (``layer7_memory_heads`` / L14). When ``enable=False`` (the
        # default) the bake is a no-op, but the dep declaration still
        # contributes to the scheduler's cycle decomposition.
        requires={"after": "layer4_pc_relay"},
        # Phase 11.A IR exposure: at default ``enable=False`` the bake body
        # is ``if not enable: return``, so empty CompilerIR is byte-identical.
        # When ``enable=True`` the head spec mixes a declarative attention
        # spec with alibi_slopes[head]=0.5 mutation -- the slope side needs
        # an :class:`AttentionOp` fragment (DSL Wave W7: a bake-fn carried
        # on the IR's ``fragments`` list, selected at IR-build time rather
        # than via a runtime predicate).
        compiler_ir=CompilerIR(),
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#the-attention-layer",
    )


# =====================================================================
# Wave A v2 (2026-06-10): register-tagged STEP_END operand relay.
# =====================================================================
#
# Two declarative attention heads in L9 that mirror the same-step
# operand state from MARK_AX into the SE_-tagged dims at MARK_SE_ONLY.
# Sits in L9 attention so it fires BEFORE L9 FFN's CMP rules
# (``_layer9_cmp_rules``, migrated to MARK_SE_ONLY by commit 62b64449)
# and provides the operand bands the CMP rules need at the SE row.
#
# Head A (mirrors compute flags + ALU values):
#   * SE_OP_EQ / NE / LT / GT / LE / GE (6 slots)
#   * SE_CMP_GROUP (1 slot)
#   * SE_CMP+0..3 (4 slots)
#   * SE_ALU_LO+0..15 (16 slots)
#   * SE_ALU_HI+0..15 (16 slots)
#   = 43 slots total (fits HD=64)
#
# Head B (mirrors AX carry staging):
#   * SE_AX_CARRY_LO+0..15 (16 slots)
#   * SE_AX_CARRY_HI+0..15 (16 slots)
#   = 32 slots total (fits HD=64)
#
# Why two heads (not one)? Total slot count (75) exceeds HD=64 for a
# single head. Splitting along the ALU / CARRY boundary keeps each
# head's V/O writes within budget without forcing a wider model.
#
# Why TAGGED dims (SE_*) instead of raw (ALU_LO etc.)? The L11 relay
# (10ca51a7, enable=False) broadcasts RAW ALU_LO/HI at MARK_SE_ONLY
# and regresses 9 smoke tests because downstream readers (BZ/BNZ gate,
# LI/LC, memory ops) consume raw ALU_LO/HI at MARK_AX cross-step via
# OUTPUT_LO_PREV_STEP-style relays; writing them again at MARK_SE_ONLY
# pollutes that bus. The SE_ prefix keeps the operand band scoped to
# ``mark == SE_ONLY`` (semantics declared in dim_registry.py) so the
# raw band readers cannot see them. See memory note
# ``project_wave_b_cmp_needs_l9_internal_relay.md``.
#
# ALiBi slope 0.2 keeps the relay step-local: within-step MARK_AX is
# 29 rows back from MARK_SE_ONLY, prior-step MARK_AX is ~64 back. With
# slope 0.2 the within-step score is L^2/sqrt(HD) - 0.2*29 = 12.5-5.8
# = 6.7, prior-step 12.5-12.8 = -0.3, so softmax favours the
# within-step AX by ~exp(7) ≈ 1100. Same convention as L1 head 6
# (``_step_end_reg_present_head_spec``) and the L11 step_end_relay.


def _step_end_operand_relay_head_specs(
    BD,
) -> tuple[DeclarativeAttentionHeadSpec, DeclarativeAttentionHeadSpec]:
    """Build the two Wave A v2 relay head specs.

    Q at MARK_SE_ONLY (the STEP_END row). K at MARK_AX (the in-step
    operand-state row). V copies each named source dim with weight 1.0;
    O writes the same value into the SE_-tagged sister dim at the Q row
    with weight 1.0.

    A positive ALiBi slope (0.2) plus L=10 Q/K weights keeps the relay
    step-local: at distance 29 (within-step MARK_AX -> MARK_SE_ONLY)
    the score is ``L^2/sqrt(HD) - 0.2*29 ≈ 6.7``; the prior step's
    MARK_AX sits at distance ~64 (loses by ~exp(7) ≈ 1100 in softmax).
    Mirrors the L1 head 6 / L11 step_end_relay shape.

    DERIVED via the generic :func:`marker_broadcast` BARE-MARKER mode
    (``is_byte_dim=None``): a plain Q@MARK_SE_ONLY / K@MARK_AX
    marker-row -> marker-row relay with NO IS_BYTE fire-site and NO
    CONST confirm slot — identical structure to the L11 step_end relay,
    the ONLY differences being the ``SE_``-tagged ``target_band`` (each
    band's O writes the register-tagged sister dim, not the source dim)
    and the reserved slot 0 (``v_slot_base`` starts at 1). Byte-identical
    to the hand-built heads (proof: ``tools/_isa_golden_hash.py``
    unchanged).
    """
    L = 10.0
    _CMP_OPS = ("OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE")

    # Resolve dim names -> positions through the same ``BD`` proxy the
    # hand-built head used (identical ``_SetDim`` fallback for undeclared
    # names).
    class _DimView:
        def __getitem__(self, name):
            return getattr(BD, name)
    dim_view = _DimView()

    def _relay_spec(name, bands, head_idx):
        return marker_broadcast(MarkerBroadcastSpec(
            name=name,
            fire_slot_dim="MARK_SE_ONLY",      # Q fires at the STEP_END marker
            source_marker="MARK_AX",           # K selects the in-step AX marker
            broadcast_bands=tuple(bands),
            weight=L,
            is_byte_dim=None, const_dim=None, gate_slot=None,  # BARE marker mode
            alibi_slope=0.2,
        )).head_spec_builder(dim_view, head_idx)

    # --- Head A: SE_OP_<cmp> + SE_CMP_GROUP + SE_CMP + SE_ALU_LO/HI ---
    # Slot 0 RESERVED for the Q-K score; broadcast bands start at slot 1.
    bands_a: list = []
    slot = 1
    for op_name in _CMP_OPS:
        bands_a.append(MarkerBroadcastBand(op_name, f"SE_{op_name}", 1, slot, 1.0))
        slot += 1
    bands_a.append(MarkerBroadcastBand("CMP_GROUP", "SE_CMP_GROUP", 1, slot, 1.0))
    slot += 1
    bands_a.append(MarkerBroadcastBand("CMP", "SE_CMP", 4, slot, 1.0))
    slot += 4
    bands_a.append(MarkerBroadcastBand("ALU_LO", "SE_ALU_LO", 16, slot, 1.0))
    slot += 16
    bands_a.append(MarkerBroadcastBand("ALU_HI", "SE_ALU_HI", 16, slot, 1.0))
    slot += 16
    head_a_idx = _alu_head_idx("layer9_step_end_operand_relay")
    spec_a = _relay_spec(
        "layer9_step_end_operand_relay.head_a", bands_a, head_a_idx,
    )

    # --- Head B: SE_AX_CARRY_LO + SE_AX_CARRY_HI ---
    # Slot 0 RESERVED for the Q-K score; broadcast bands start at slot 1.
    bands_b: list = []
    slot = 1
    bands_b.append(MarkerBroadcastBand("AX_CARRY_LO", "SE_AX_CARRY_LO", 16, slot, 1.0))
    slot += 16
    bands_b.append(MarkerBroadcastBand("AX_CARRY_HI", "SE_AX_CARRY_HI", 16, slot, 1.0))
    slot += 16
    spec_b = _relay_spec(
        "layer9_step_end_operand_relay.head_b", bands_b, head_a_idx + 1,
    )

    return spec_a, spec_b


def _step_end_operand_relay_ir(dim_positions, HD) -> CompilerIR:
    """``compiler_ir_factory`` for the L9 step_end_operand_relay heads."""
    del HD  # head_dim is layer-default
    BD = _as_setdim_proxy(dim_positions)
    spec_a, spec_b = _step_end_operand_relay_head_specs(BD)
    ir = CompilerIR()
    ir.layer(0).attention.append(
        spec_a, name="layer9_step_end_operand_relay.head_a",
    )
    ir.layer(0).attention.append(
        spec_b, name="layer9_step_end_operand_relay.head_b",
    )
    return ir


def make_layer9_step_end_operand_relay_op() -> Operation:
    """L9 Wave A v2: register-tagged STEP_END operand relay (2 heads).

    Two declarative attention heads inside L9 attn that mirror the
    same-step ALU/CARRY/CMP/OP operand bands at MARK_AX into the
    register-tagged ``SE_*`` dims at MARK_SE_ONLY. Runs as the L9
    attn-side complement to the migrated L9 CMP rules (commit
    62b64449) which gate on MARK_SE_ONLY and consume the SE_-tagged
    bands. See the long header comment above the spec helper and
    ``docs/STEP_END_COMPUTE_ARCHITECTURE_2026_06_10.md`` for the design
    rationale.

    Why not the L11 ``step_end_operand_relay`` (10ca51a7, enable=False)?
    That earlier relay (i) writes raw ALU_LO/HI at MARK_SE_ONLY,
    polluting cross-step / downstream readers (9 measured smoke
    regressions), AND (ii) fires AT L11 -- AFTER L9 FFN, too late for
    the L9 CMP rules. This L9 relay fixes BOTH: per-register tagging
    via the SE_-prefixed slots AND placement upstream of L9 FFN.
    """
    def bake(block, dim_positions, S):
        del S  # ALiBi slope is the only scale factor and is set below
        attn = block.attn
        # Per-bake L9 head allocator with the full layout pinned. Look
        # up our relay head A by name; head B claims the next slot.
        allocator = _allocate_alu_attention_heads()
        attn._l9_head_allocator = allocator
        head_a_idx = next(
            h.head_idx for h in allocator.heads()
            if h.op_name == "layer9_step_end_operand_relay"
        )
        head_b_idx = head_a_idx + 1
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes[head_a_idx] = 0.2
            attn.alibi_slopes[head_b_idx] = 0.2
        HD = attn.W_q.shape[0] // attn.num_heads
        spec_a, spec_b = _step_end_operand_relay_head_specs(
            _as_setdim_proxy(dim_positions),
        )
        Primitives.generate_attention_head(attn, spec_a, HD)
        Primitives.generate_attention_head(attn, spec_b, HD)

    # Dim-ownership claims: 2 heads on L9 attn. Resolved head indices
    # come from ``_alu_head_idx`` (head A) and ``head_a + 1`` (head B).
    # We pin them as ``A_<slot>`` / ``B_<slot>`` strings; the verifier
    # accepts the symbolic head names as long as they're unique per op.
    _CMP_OPS = ("OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE")
    _claims: set = set()
    head_a_idx = _alu_head_idx("layer9_step_end_operand_relay")
    head_b_idx = head_a_idx + 1
    # Slot 0 reserved for Q-K score; V/O start at slot 1.
    slot = 1
    for op_name in _CMP_OPS:
        _claims.add((9, "attn_W_v", f"{head_a_idx}_{slot}", f"{op_name}+0"))
        _claims.add((9, "attn_W_o", f"{head_a_idx}_{slot}", f"SE_{op_name}+0"))
        slot += 1
    _claims.add((9, "attn_W_v", f"{head_a_idx}_{slot}", "CMP_GROUP+0"))
    _claims.add((9, "attn_W_o", f"{head_a_idx}_{slot}", "SE_CMP_GROUP+0"))
    slot += 1
    for k_idx in range(4):
        _claims.add((9, "attn_W_v", f"{head_a_idx}_{slot}", f"CMP+{k_idx}"))
        _claims.add((9, "attn_W_o", f"{head_a_idx}_{slot}", f"SE_CMP+{k_idx}"))
        slot += 1
    for k_idx in range(16):
        _claims.add((9, "attn_W_v", f"{head_a_idx}_{slot}", f"ALU_LO+{k_idx}"))
        _claims.add((9, "attn_W_o", f"{head_a_idx}_{slot}", f"SE_ALU_LO+{k_idx}"))
        slot += 1
    for k_idx in range(16):
        _claims.add((9, "attn_W_v", f"{head_a_idx}_{slot}", f"ALU_HI+{k_idx}"))
        _claims.add((9, "attn_W_o", f"{head_a_idx}_{slot}", f"SE_ALU_HI+{k_idx}"))
        slot += 1
    slot = 1
    for k_idx in range(16):
        _claims.add(
            (9, "attn_W_v", f"{head_b_idx}_{slot}", f"AX_CARRY_LO+{k_idx}"),
        )
        _claims.add(
            (9, "attn_W_o", f"{head_b_idx}_{slot}", f"SE_AX_CARRY_LO+{k_idx}"),
        )
        slot += 1
    for k_idx in range(16):
        _claims.add(
            (9, "attn_W_v", f"{head_b_idx}_{slot}", f"AX_CARRY_HI+{k_idx}"),
        )
        _claims.add(
            (9, "attn_W_o", f"{head_b_idx}_{slot}", f"SE_AX_CARRY_HI+{k_idx}"),
        )
        slot += 1

    return Operation(
        name="layer9_step_end_operand_relay",
        reads={"MARK_SE_ONLY", "MARK_AX",
               "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
               "CMP_GROUP", "CMP",
               "ALU_LO", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI"},
        writes={"SE_OP_EQ", "SE_OP_NE", "SE_OP_LT", "SE_OP_GT",
                "SE_OP_LE", "SE_OP_GE",
                "SE_CMP_GROUP", "SE_CMP",
                "SE_ALU_LO", "SE_ALU_HI",
                "SE_AX_CARRY_LO", "SE_AX_CARRY_HI"},
        kind="block",
        # Phase 9.3 places this AFTER the L9 LEV relays (phase 9.0,
        # 9.1) and the disabled mem_attn (phase 9.2) so our
        # alibi_slopes overrides on heads 3/4 don't get clobbered by
        # a sibling op that fills the whole slope vector.
        phase=9.3,
        declarative_bake_fn=bake,
        compiler_ir_factory=_step_end_operand_relay_ir,
        # Bind to ``layer9_marker_suppress`` (the L9 FFN topology
        # anchor) so the block op resolves to the same physical layer
        # as the other L9 attention ops (which today resolves to L11
        # due to the 23-logical / 36-physical layer expansion -- the
        # L11 attn fires BEFORE the L11 FFN that hosts the migrated
        # L9 CMP rules, so the relay output is visible to the rules).
        target_op_name="layer9_marker_suppress",
        migrated=True,
        declarative_authority="spec_generated",
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def make_layer9_se_relay_slope_op() -> Operation:
    """Wave B Phase 2.1: re-assert the SE operand relay's ALiBi slopes.

    The L9 ``step_end_operand_relay`` heads (A/B) land physically on the
    relay block (``model.blocks[10]`` pre-expansion -> physical block 11
    = logical L10 after ``expand_wrapper_blocks``). Their intended slope
    is 0.2 (set in ``make_layer9_step_end_operand_relay_op``'s bake at
    phase 9.3), which keeps the relay step-local: over the d=29
    MARK_AX->MARK_SE gap a 0.2 slope is only a -5.8 penalty (within-step
    AX wins by ~exp(7) over the prior step's AX at d~64).

    BUT ``make_layer10_residual_alibi_slopes_op`` runs LATER (phase
    999.1) and unconditionally overwrites ``blocks[10].alibi_slopes[3] =
    0.5`` and ``[4] = 1.0`` -- the legacy "PSH STACK0 passthrough" /
    "STACK0 byte relay" slopes for heads that the relay has since
    physically displaced (verified: blocks[11] heads 3/4 carry ONLY the
    relay's Q@MARK_SE / K@MARK_AX / V(ALU,CARRY,CMP,OP) / O(SE_*) -- no
    surviving L10 passthrough V/O). A 0.5 slope over the d=29 gap is a
    -14.5 penalty that swamps the L=10 QK match, so the relay attends to
    nothing and ``SE_*`` collapses to its bias floor (~-1.4) -- the
    documented Wall-2 transmission failure
    (``project_attention_dsl_alibi_slope_gap``).

    Fix per the slope-ownership rule (set the slope LAST): re-assert the
    relay heads' slope to 0.2 AFTER ``layer10_residual_alibi_slopes``.
    This op owns NO weights -- only the two slope cells the relay needs.
    The L10 passthrough functions that previously read heads 3/4 are
    already gone from this block in the baseline (the MARK_AX ordering
    engine drives lt/le today), so re-sloping is additive, not a
    passthrough regression.
    """
    def _bake(model, dim_positions, S):
        del dim_positions, S
        # Same pre-expansion index residual_alibi writes (blocks[10] ->
        # physical block 11 = logical L10 post-expansion). The relay's
        # head A/B indices come from the L9 allocator layout.
        if len(model.blocks) <= 10:
            return
        attn = model.blocks[10].attn
        if not (hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None):
            return
        head_a = _alu_head_idx("layer9_step_end_operand_relay")
        head_b = head_a + 1
        if head_b < attn.alibi_slopes.shape[0]:
            attn.alibi_slopes[head_a] = 0.2  # SE relay head A (ALU/CMP/OP)
            attn.alibi_slopes[head_b] = 0.2  # SE relay head B (AX_CARRY)

    return Operation(
        name="layer9_se_relay_slope",
        # Run AFTER layer10_residual_alibi_slopes (phase 999.1) so the
        # relay's 0.2 slope wins the last write, and before
        # expand_wrapper_blocks (phase 1300) -- the same pre-expansion
        # window residual_alibi occupies. Model-ops are applied in phase
        # order (layer_compiler sorts model_ops by ``o.phase``), so the
        # explicit phase 999.2 -- not just ``requires["after"]`` -- is
        # what guarantees we run last among the slope writers.
        requires={"after": ("layer10_residual_alibi_slopes",)},
        phase=999.2,
        reads=set(),
        writes=set(),
        kind="model",
        declarative_bake_fn=_bake,
        migrated=True,
        declarative_authority="structural_model",
        # Slope-only re-assertion; no per-cell weight claims.
        claims=set(),
        smoke_tests=set(),
        spec_section="BLOG_SPEC.md#the-attention-layer",
    )


def make_marker_suppress_op() -> Operation:
    """No-op dep anchor for marker suppression owned by ``layer9_alu``.

    ``make_layer9_alu_op`` owns the true migrated bake: it calls
    ``_set_layer9_alu`` and then threads the returned unit cursor into
    ``_set_layer9_marker_suppress``. This standalone op remains only to
    preserve dependency topology for downstream audits; baking the helper here
    would either overlap L9 ALU units or double-write marker-suppression
    weights.
    """
    def bake(ffn, dim_positions, S):
        # No-op: actual bake is chained from `layer9_alu`.
        return

    return Operation(
        name="layer9_marker_suppress",
        reads={"MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_STACK0",
               "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
               "OP_OR", "OP_XOR", "OP_AND"},
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="ffn",
        migrated=True,
        declarative_authority="topology_anchor",
        # Phase 8.A.4 retry: this op is the L9 layer anchor. The scheduler
        # (commit 78ec127c) honors block-op refs in ``requires["after"]`` --
        # pointing at ``layer8_alu`` (kind="block", layer_idx=8) forces the
        # anchor to ``earliest = 9``. L9 block ops (layer9_alu and friends)
        # then bind to this anchor's resolved layer via ``target_op_name``.
        requires={"after": "layer8_alu"},
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
        # Phase 11.A IR exposure: empty IR exposes the topology-anchor's
        # noop weight semantics to the dim-multiplexer (Phase 10.E/F).
        compiler_ir=CompilerIR(),
    )
