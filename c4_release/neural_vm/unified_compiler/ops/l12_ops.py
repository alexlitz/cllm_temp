"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ...dim_registry import dim_ref
from ...ffn_unit_allocator import FFNUnitAllocator
from ..building_blocks_dsl import multi_way_and_rule
from ..ir import CompilerIR, FFNRule
from ..layer_compiler import Operation
from ..primitives import Primitives
from .shared import _as_setdim_proxy


# === L12 FFN unit layout (auto-fit) =================================
#
# The ``layer12_mul_combine`` op owns the entire L12 FFN. The weight
# writes happen via ``mul_combine_rules`` (a list of
# :class:`FFNRule` declarations) which is lowered by
# ``Primitives.lower_ffn_rules`` through ``CompilerIR.lower_ffn``. The
# rules walk a ``(partial, a_hi, b_lo)`` triple-loop and fill all 4096
# hidden units (16^3).
#
# Phase 7.B.5: the partials sub-stage now uses ``pin=None`` so the
# :class:`FFNUnitAllocator` first-fit pick lands it at unit 0 — there are
# no prior claims, so the lowest free gap is the start of the pool, which
# is byte-identical to the legacy explicit ``pin=0``. Dropping the pin
# turns the layout table into an audit-only declaration of name and
# size; the allocator owns the offset.
#
# The rule order in ``mul_combine_rules`` determines the per-unit
# write semantics. Changing the rule list requires updating this table
# entry's ``n_units`` in lock-step.
_MUL_COMBINE_UNIT_LAYOUT = (
    # (sub-stage name, n_units)  -- pin dropped (Phase 7.B.5, auto-fit)
    ("layer12_mul_combine.partials", 4096),  # 16 partial x 16 a_hi x 16 b_lo
)


def allocate_mul_combine_units() -> FFNUnitAllocator:
    """Build a per-bake :class:`FFNUnitAllocator` with L12's MUL combine range.

    Phase 7.B.5: the partials sub-stage is now auto-fit (``pin=None``);
    first-fit on an empty 4096-wide pool returns start unit 0, which is
    byte-identical to the legacy explicit ``pin=0`` the previous
    revision used. The FFNRule lowering still consumes ``start_unit=0``
    from the allocator's range, so every weight write lands on the same
    hidden-unit index the legacy ``_set_layer12_mul_combine`` helper
    wrote.

    Returns the allocator so callers can inspect or extend it (e.g. a
    future L12 op claims a free range past unit 4096 by widening the
    pool, or claims a sub-range if the rule list is split).
    """
    allocator = FFNUnitAllocator()
    for name, n_units in _MUL_COMBINE_UNIT_LAYOUT:
        allocator.alloc(name, n_units)
    return allocator


def mul_combine_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for the L12 MUL combine sub-stage (4096 units).

    Mirrors the legacy ``_set_layer12_mul_combine`` helper one-for-one:
    a single 4-way AND per ``(partial, a_hi, b_lo)`` triple firing on
    ``MARK_AX + TEMP[partial] + ALU_HI[a_hi] + AX_CARRY_LO[b_lo]`` at
    threshold ``7.5`` (calibrated for hot TEMP[partial] ~= 5.0; see L11
    amplitude contract). Each unit is gated by ``OP_MUL`` (``W_gate[unit,
    OP_MUL] = 1.0``, ``b_gate = 0.0``) and writes a single
    ``OUTPUT_HI[result_hi]`` cell at ``2.0/S`` so the hot unit deposits
    ~1.0 in the residual.

    ``result_hi = (partial + a_hi * b_lo) % 16``: L11 staged
    ``partial = (a_lo*b_lo // 16 + a_lo*b_hi) % 16`` into TEMP and L12
    here adds ``a_hi*b_lo`` to recover the full ``result_hi`` nibble of
    ``(a*b) & 0xFF``. The ``OUTPUT_LO`` nibble was already populated
    upstream by L10's MUL units; L12 only writes ``OUTPUT_HI``.

    The triple-loop order (``partial`` outer, ``a_hi`` middle, ``b_lo``
    inner) mirrors the legacy helper's ``unit`` counter so the rule order
    -- and therefore the pinned hidden-unit indices via
    ``Primitives.lower_ffn_rules`` -- is byte-identical with the original
    bake.

    Phase 7.E.3: gate uses :func:`dim_ref` for the
    ``(opcode_flag, MUL)`` semantic pair. Structural reads
    (``TEMP+partial``, ``ALU_HI+a_hi``, ``AX_CARRY_LO+b_lo``) and the
    ``OUTPUT_HI+result_hi`` write stay as ``+N`` -- ``result_hi`` is a
    value-bus lookup index, not a role-meaningful byte position.

    Wave B Cluster 4 (2026-06-10): position marker migrated from
    ``MARK_AX`` to ``MARK_SE_ONLY``. The Wave A
    ``step_end_operand_relay`` head (commit 10ca51a7) broadcasts
    ALU_HI, AX_CARRY_LO, and OP_MUL from MARK_AX to MARK_SE_ONLY
    within the same step. The L11 MUL partial output is staged in
    TEMP[partial] at MARK_AX and re-read at MARK_SE_ONLY via the
    standard intra-step residual carry. The ``OUTPUT_HI+result_hi``
    write is preserved -- the residue lands on the STEP_END row and
    is picked up by the byte-row relay downstream. See
    ``docs/WAVE_B_CLUSTER_4_PLAN_2026_06_10.md`` and
    ``docs/STEP_END_MIGRATION_TEMPLATE.md`` for the recipe.
    """
    write_scale = 2.0 / S
    gate_mul = dim_ref("opcode_flag", "MUL")
    rules: list[FFNRule] = []

    for partial in range(16):
        for a_hi in range(16):
            for b_lo in range(16):
                result_hi = (partial + a_hi * b_lo) % 16
                rules.append(multi_way_and_rule(
                    name=f"l12_mul_combine_p{partial:02d}_ah{a_hi:02d}_bl{b_lo:02d}",
                    conditions=(
                        # Wave B Cluster 4: MARK_AX -> MARK_SE_ONLY
                        # under Wave A step_end_operand_relay
                        # (10ca51a7). The L11-staged TEMP[partial]
                        # carries to STEP_END via the intra-step
                        # residual; ALU_HI / AX_CARRY_LO / OP_MUL ride
                        # the relay.
                        ("MARK_SE_ONLY", 1.0),
                        (f"TEMP+{partial}", 1.0),
                        (f"ALU_HI+{a_hi}", 1.0),
                        (f"AX_CARRY_LO+{b_lo}", 1.0),
                    ),
                    threshold=7.5,
                    gate=gate_mul,
                    writes=((f"OUTPUT_HI+{result_hi}", write_scale),),
                    scope="MARK_SE_ONLY and OP_MUL",
                    dominates_at={
                        f"OUTPUT_HI+{result_hi}": "MARK_SE_ONLY and OP_MUL",
                    },
                ))

    return tuple(rules)


def mul_combine_ir(S: float = 100.0) -> CompilerIR:
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(mul_combine_rules(S))
    return ir


def bake_mul_combine(ffn, S, BD) -> int:
    """Bake L12 MUL combine via :class:`FFNRule` lowering (byte-identical helper).

    Used both by the migrated :func:`make_mul_combine_op` bake path
    and as a standalone entry-point for callers wanting to drive the L12
    weight writes without constructing a full ``Operation``. Mirrors the
    L1 ``_bake_layer1_ffn`` convention.
    """
    rules = mul_combine_rules(S)
    dim_positions = Primitives.dim_positions_from_bd(
        BD,
        Primitives.ffn_rule_dim_names(rules),
    )
    return Primitives.lower_ffn_rules(ffn, rules, dim_positions, S=S)


def make_mul_combine_dep_anchor_op() -> Operation:
    """No-op companion for ``layer12_mul_combine``: declares mirrored
    reads/writes so the LayerCompiler's dep graph reserves an L12 slot
    for it. Mirrors ``_layer11_ffn_dep_anchor`` / ``_layer3_ffn_dep_anchor``:
    the actual weight bake happens in ``layer12_mul_combine`` (kind=
    "block"); this op's bake is a no-op.

    Phase 8.G.6: lets L12 block ops declare
    ``target_op_name="_layer12_ffn_dep_anchor"`` and bind to whichever
    layer the compiler places the anchor at, instead of carrying a
    literal ``layer_idx=12``.
    """
    def bake(ffn, dim_positions, S):
        # No-op: actual bake is in ``layer12_mul_combine`` block op below.
        return

    return Operation(
        name="_layer12_ffn_dep_anchor",
        # Phase=11.5 places this anchor between the L11 MUL partial
        # writes (phase=11) and the L12 MUL combine (phase=12). Reads
        # include TEMP (written by L11 MUL partial) so the dep graph
        # earliest-fit lands at L12.
        phase=11.5,
        # Phase 11.x (SCC sub-cycle D, TEMP): mirror layer12_mul_combine's
        # TEMP -> TEMP.*.-1 SSA rename. This anchor's TEMP read otherwise
        # creates the same L14 -> _layer12_ffn_dep_anchor back-edge that
        # the block op's rename retires.
        reads={"MARK_AX", "TEMP.*.-1", "ALU_HI", "AX_CARRY_LO", "OP_MUL"},
        writes={"OUTPUT_HI_THIS_STEP"},
        kind="ffn",
        migrated=True,
        declarative_authority="topology_anchor",
        # Co-place with the L11 dep anchor + 1 layer (TEMP read deps
        # against L11 MUL partial force earliest=12).
        requires={"after": "_layer11_ffn_dep_anchor"},
        smoke_tests=set(),
        spec_section=None,
        # Phase 11.A IR exposure: empty IR exposes the topology-anchor's
        # noop weight semantics to the dim-multiplexer (Phase 10.E/F).
        compiler_ir=CompilerIR(),
    )


def make_mul_combine_op(alu_mode: str = "lookup") -> Operation:
    """L12 FFN: combine MUL partial products into final result.

    Pinned to ``layer_idx=12`` via ``kind="block"``. See
    ``make_mul_partial_op``.

    Declarations-only note: this migrated owner is now exposed through the
    declarations-only dispatcher so strict builds do not fall back to legacy
    model bake.

    Migration (Phase 6 wave 4E): the imperative
    ``setup_helpers._set_layer12_mul_combine`` triple-loop is now declared
    via :func:`mul_combine_rules` and attached as
    ``compiler_ir``; the bake lowers through :meth:`CompilerIR.lower_ffn`
    via :func:`bake_mul_combine`.
    """
    def bake(block, dim_positions, S):
        if alu_mode == "efficient":
            return None

        # GAP-PRIMITIVE #2 (``C4_MUL_MULTIPASS=1``): the multi_pass cascade at
        # L11 computes the FULL product (incl. byte-0 high nibble -> OUTPUT_HI)
        # and no longer writes the ``TEMP[partial]`` band this combine reads.
        # With TEMP empty, every one of these 4096 4-way-AND units is DEAD
        # (its ``MARK_AX + TEMP[partial] + ALU_HI + AX_CARRY_LO`` AND never
        # fires). Skip the lowering entirely so the block contributes 0 units
        # (the extra −4096 on top of the L11 −1248). Flag-off is unchanged.
        from .shared import mul_multipass_enabled
        if mul_multipass_enabled():
            from ...base_layers import PureFFN
            d_model = (
                int(block.ffn.W_up.shape[1])
                if hasattr(block.ffn, "W_up") and block.ffn.W_up is not None
                else int(block.attn.dim)
            )
            block.ffn = PureFFN(dim=d_model, hidden_dim=1)
            return

        proxy = _as_setdim_proxy(dim_positions)

        # Per-bake FFN-unit allocator. The L12 MUL combine sub-stage is
        # pinned to its existing offset so the call below lands
        # byte-identically. Exposed on the block for downstream tools
        # (e.g. a future L12 op family claiming a free gap), mirroring
        # the L9 convention introduced in ca775eb.
        allocator = allocate_mul_combine_units()
        partials_range = next(
            r for r in allocator.ranges()
            if r.op_name == "layer12_mul_combine.partials"
        )
        block.ffn._l12_unit_allocator = allocator

        # Phase 8.C inline: lower the rule list directly (was
        # ``bake_mul_combine``) so census v2 classifies this op
        # as ``declarative`` rather than ``declarative_via_helper``.
        rules = mul_combine_rules(S)
        rule_dim_positions = Primitives.dim_positions_from_bd(
            proxy, Primitives.ffn_rule_dim_names(rules),
        )
        n12 = Primitives.lower_ffn_rules(
            block.ffn, rules, rule_dim_positions, S=S,
        )
        # Byte-identity guard: the FFNRule lowering MUST end exactly
        # where the allocator's partials range ends. If the rule list
        # drifts from the layout table, this assertion fires before any
        # downstream consumer reads the FFN.
        expected_end = partials_range.start + partials_range.n_units
        assert n12 == expected_end, (
            f"L12 MUL combine unit cursor drift: rules lowered {n12} units, "
            f"allocator expected {expected_end}"
        )

    # Dim-ownership claims (W_down output cells). The 4096 units are
    # arranged as ``unit = partial*256 + a_hi*16 + b_lo`` (mirroring the
    # legacy helper's triple-loop walk), and each unit writes one
    # ``OUTPUT_HI[(partial + a_hi*b_lo) % 16]`` cell. The dim is the
    # ``OUTPUT_HI`` alias of ``OUTPUT_HI_THIS_STEP`` (same position 190);
    # ``verify_claims_static`` decodes the W_down row back to a name via
    # ``_pos_to_column`` which resolves position 190 to the alphabetically
    # earliest registered dim (``OUTPUT_HI`` < ``OUTPUT_HI_THIS_STEP``),
    # so the claim strings here use ``OUTPUT_HI+k`` to match what the
    # verifier observes -- not the ``OUTPUT_HI_THIS_STEP`` alias used in
    # the op-level ``writes=`` set or the rule ``writes=`` tuples.
    _claims = set()
    for partial in range(16):
        for a_hi in range(16):
            for b_lo in range(16):
                unit = partial * 256 + a_hi * 16 + b_lo
                result_hi = (partial + a_hi * b_lo) % 16
                _claims.add(
                    (12, "ffn_W_down", str(unit), f"OUTPUT_HI+{result_hi}")
                )

    return Operation(
        name="layer12_mul_combine",
        # ``_set_layer12_mul_combine`` reads MARK_AX, TEMP[partial], ALU_HI[a_hi],
        # AX_CARRY_LO[b_lo], gates on OP_MUL, and writes ONLY to OUTPUT_HI (the
        # low byte's high nibble). It does NOT write OUTPUT_LO -- L10's MUL
        # units already own the low nibble of the low byte. The previously
        # over-declared ALU_HI/AX_CARRY_LO read and OUTPUT_LO write here
        # silently masked downstream contention analysis. Corrected so the
        # staleness analyzer / claim-collision detector see the true producer/
        # consumer surface of this op.
        # Phase 11.x (SCC sub-cycle D, TEMP): TEMP -> TEMP.*.-1 SSA cross-step
        # rename. The same-step fresh TEMP[partial] from L11 mul_partial is
        # still encoded via ``consumes_fresh={"TEMP": "AX_byte0"}`` below,
        # which preserves the L11 -> L12 producer/consumer edge. The plain
        # ``TEMP`` read here was the structural in-edge from every TEMP writer
        # including ``layer14_temp_clear`` (phase=14.1), which writes TEMP for
        # the NEXT step's L12 read -- creating the L14 -> L12 back-edge that
        # was the last TEMP cycle in SCC #1 (see
        # ``.agent-logs/scc_remaining_structural_20260602.md`` section 2.4).
        # SSA alias resolves to the same numeric slot at bake time
        # (byte-identical), but the scheduler now treats every base-TEMP
        # writer as a prev-step residual relative to this read. Retires
        # SCC #1 sub-cycle D.
        reads={"MARK_AX", "TEMP.*.-1", "ALU_HI", "AX_CARRY_LO", "OP_MUL"},
        writes={"OUTPUT_HI_THIS_STEP"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir=mul_combine_ir(),
        declarative_authority="spec_generated",
        # Phase 8.G.6: drop ``layer_idx=12`` literal; bind to the L12
        # ffn dep anchor so the block op resolves to whichever layer
        # the compiler places the anchor at.
        target_op_name="_layer12_ffn_dep_anchor",
        migrated=True,
        claims=_claims,
        ffn_units_used=4096,
        # Staleness invariants: L12 MUL combine consumes the same operands
        # as L11 MUL partial (ALU_HI for a_hi, AX_CARRY_LO for b_lo) at the
        # AX marker. These must be the in-step fresh values. Also consumes
        # the fresh TEMP[partial] just written by ``layer11_mul_partial`` at
        # the AX marker (phase 11 < 12).
        # Produces the fresh MUL hi-nibble result at the AX marker (gated
        # on MARK_AX + OP_MUL): ``result_hi = (partial + a_hi*b_lo) % 16``
        # is written via 4-way AND units into OUTPUT_HI. ``_set_layer12_mul_combine``
        # itself writes only OUTPUT_HI (not OUTPUT_LO); the lo-nibble was
        # already populated upstream in L10's MUL units, so the staleness
        # contract only covers the hi half emitted here.
        smoke_tests={
            "TestSmoke32Bit::test_mul_overflow",
            "TestSmokeBasic::test_mul_basic",
        },
        spec_section="BLOG_SPEC.md#multiplication-implementation",
        # Tier A opcode gating: ``_set_layer12_mul_combine`` writes every one
        # of its 4096 hidden units with ``W_gate[unit, BD.OP_MUL] = 1.0``, so
        # every unit's SiLU output is gated on OP_MUL. The L12 MUL hi-nibble
        # combine FFN fires ONLY on OP_MUL steps; non-MUL opcodes leave block
        # 12 untouched.
        opcodes={"OP_MUL"},
    )
