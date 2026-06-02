"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ...dim_registry import dim_ref
from ...ffn_unit_allocator import FFNUnitAllocator
from ..ir import CompilerIR, FFNRule
from ..layer_compiler import Operation
from ..primitives import Primitives
from .shared import _as_setdim_proxy


# === L12 FFN unit layout (auto-fit) =================================
#
# The ``layer12_mul_combine`` op owns the entire L12 FFN. The weight
# writes happen via ``_layer12_mul_combine_rules`` (a list of
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
# The rule order in ``_layer12_mul_combine_rules`` determines the per-unit
# write semantics. Changing the rule list requires updating this table
# entry's ``n_units`` in lock-step.
_L12_MUL_COMBINE_UNIT_LAYOUT = (
    # (sub-stage name, n_units)  -- pin dropped (Phase 7.B.5, auto-fit)
    ("layer12_mul_combine.partials", 4096),  # 16 partial x 16 a_hi x 16 b_lo
)


def _allocate_layer12_mul_combine_units() -> FFNUnitAllocator:
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
    for name, n_units in _L12_MUL_COMBINE_UNIT_LAYOUT:
        allocator.alloc(name, n_units)
    return allocator


def _layer12_mul_combine_rules(S: float) -> tuple[FFNRule, ...]:
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
    """
    write_scale = 2.0 / S
    gate_mul = dim_ref("opcode_flag", "MUL")
    rules: list[FFNRule] = []

    for partial in range(16):
        for a_hi in range(16):
            for b_lo in range(16):
                result_hi = (partial + a_hi * b_lo) % 16
                rules.append(FFNRule.gated_write(
                    name=f"l12_mul_combine_p{partial:02d}_ah{a_hi:02d}_bl{b_lo:02d}",
                    conditions=(
                        ("MARK_AX", 1.0),
                        # structural offset: partial/a_hi/b_lo are
                        # nibble-value one-hot lookup indices into the
                        # TEMP scratch and operand bands.
                        (f"TEMP+{partial}", 1.0),
                        (f"ALU_HI+{a_hi}", 1.0),
                        (f"AX_CARRY_LO+{b_lo}", 1.0),
                    ),
                    threshold=7.5,
                    gate=gate_mul,
                    gate_weight=1.0,
                    gate_bias=0.0,
                    # structural offset: result_hi is the computed
                    # high-byte nibble of (a*b) & 0xFF (value-bus
                    # lookup), not a role-meaningful byte position.
                    writes=((f"OUTPUT_HI+{result_hi}", write_scale),),
                    scope="MARK_AX and OP_MUL",
                    dominates_at={
                        f"OUTPUT_HI+{result_hi}": "MARK_AX and OP_MUL",
                    },
                ))

    return tuple(rules)


def _layer12_mul_combine_ir(S: float = 100.0) -> CompilerIR:
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_layer12_mul_combine_rules(S))
    return ir


def _bake_layer12_mul_combine(ffn, S, BD) -> int:
    """Bake L12 MUL combine via :class:`FFNRule` lowering (byte-identical helper).

    Used both by the migrated :func:`make_layer12_mul_combine_op` bake path
    and as a standalone entry-point for callers wanting to drive the L12
    weight writes without constructing a full ``Operation``. Mirrors the
    L1 ``_bake_layer1_ffn`` convention.
    """
    rules = _layer12_mul_combine_rules(S)
    dim_positions = Primitives.dim_positions_from_bd(
        BD,
        Primitives.ffn_rule_dim_names(rules),
    )
    return Primitives.lower_ffn_rules(ffn, rules, dim_positions, S=S)


def make_layer12_ffn_dep_anchor_op() -> Operation:
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
        reads={"MARK_AX", "TEMP", "ALU_HI", "AX_CARRY_LO", "OP_MUL"},
        writes={"OUTPUT_HI_THIS_STEP"},
        kind="ffn",
        migrated=True,
        declarative_authority="topology_anchor",
        # Co-place with the L11 dep anchor + 1 layer (TEMP read deps
        # against L11 MUL partial force earliest=12).
        requires={"after": "_layer11_ffn_dep_anchor"},
        smoke_tests=set(),
        spec_section=None,
    )


def make_layer12_mul_combine_op(alu_mode: str = "lookup") -> Operation:
    """L12 FFN: combine MUL partial products into final result.

    Pinned to ``layer_idx=12`` via ``kind="block"``. See
    ``make_layer11_mul_partial_op``.

    Declarations-only note: this migrated owner is now exposed through the
    declarations-only dispatcher so strict builds do not fall back to legacy
    model bake.

    Migration (Phase 6 wave 4E): the imperative
    ``setup_helpers._set_layer12_mul_combine`` triple-loop is now declared
    via :func:`_layer12_mul_combine_rules` and attached as
    ``compiler_ir``; the bake lowers through :meth:`CompilerIR.lower_ffn`
    via :func:`_bake_layer12_mul_combine`.
    """
    def bake(block, dim_positions, S):
        if alu_mode == "efficient":
            return None
        proxy = _as_setdim_proxy(dim_positions)

        # Per-bake FFN-unit allocator. The L12 MUL combine sub-stage is
        # pinned to its existing offset so the call below lands
        # byte-identically. Exposed on the block for downstream tools
        # (e.g. a future L12 op family claiming a free gap), mirroring
        # the L9 convention introduced in ca775eb.
        allocator = _allocate_layer12_mul_combine_units()
        partials_range = next(
            r for r in allocator.ranges()
            if r.op_name == "layer12_mul_combine.partials"
        )
        block.ffn._l12_unit_allocator = allocator

        n12 = _bake_layer12_mul_combine(block.ffn, S, proxy)
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
        phase=12,
        # ``_set_layer12_mul_combine`` reads MARK_AX, TEMP[partial], ALU_HI[a_hi],
        # AX_CARRY_LO[b_lo], gates on OP_MUL, and writes ONLY to OUTPUT_HI (the
        # low byte's high nibble). It does NOT write OUTPUT_LO -- L10's MUL
        # units already own the low nibble of the low byte. The previously
        # over-declared ALU_HI/AX_CARRY_LO read and OUTPUT_LO write here
        # silently masked downstream contention analysis. Corrected so the
        # staleness analyzer / claim-collision detector see the true producer/
        # consumer surface of this op.
        reads={"MARK_AX", "TEMP", "ALU_HI", "AX_CARRY_LO", "OP_MUL"},
        writes={"OUTPUT_HI_THIS_STEP"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir=_layer12_mul_combine_ir(),
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
        consumes_fresh={
            "ALU_HI": "AX_byte0",
            "AX_CARRY_LO": "AX_byte0",
            "TEMP": "AX_byte0",
        } if alu_mode == "lookup" else {},
        # Produces the fresh MUL hi-nibble result at the AX marker (gated
        # on MARK_AX + OP_MUL): ``result_hi = (partial + a_hi*b_lo) % 16``
        # is written via 4-way AND units into OUTPUT_HI. ``_set_layer12_mul_combine``
        # itself writes only OUTPUT_HI (not OUTPUT_LO); the lo-nibble was
        # already populated upstream in L10's MUL units, so the staleness
        # contract only covers the hi half emitted here.
        produces={
            "OUTPUT_HI_THIS_STEP": "AX_byte0",
        },
        smoke_tests={
            "TestSmoke32Bit::test_mul_overflow",
            "TestSmokeBasic::test_mul_basic",
        },
        spec_section="BLOG_SPEC.md#multiplication-implementation",
    )
