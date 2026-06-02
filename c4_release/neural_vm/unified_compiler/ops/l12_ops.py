"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ...ffn_unit_allocator import FFNUnitAllocator
from ..ir import CompilerIR, FFNRule
from ..layer_compiler import Operation
from ..primitives import Primitives
from .shared import _as_setdim_proxy


# === L12 FFN unit layout (pinned offsets) ===========================
#
# The ``layer12_mul_combine`` op owns the entire L12 FFN. The weight
# writes happen via ``_layer12_mul_combine_rules`` (a list of
# :class:`FFNRule` declarations) which is lowered by
# ``Primitives.lower_ffn_rules`` through ``CompilerIR.lower_ffn``. The
# rules walk a ``(partial, a_hi, b_lo)`` triple-loop and fill all 4096
# hidden units (16^3). Migration to :class:`FFNUnitAllocator` keeps that
# helper byte-identical -- we declare the sub-stage's range at its
# existing pinned offset so the layout is auditable rather than implicit.
# Adding a new L12 op family later will go through
# ``allocator.alloc(name, n)`` without a pin, and the allocator will pick
# the first free gap above 4096 (or any earlier gap, though none exist in
# the current bake).
#
# The offsets below mirror the rule order in
# ``_layer12_mul_combine_rules``. Changing the rule list requires updating
# this table in lock-step.
_L12_MUL_COMBINE_UNIT_LAYOUT = (
    # (sub-stage name, pinned start, n_units)
    ("layer12_mul_combine.partials", 0, 4096),  # 16 partial x 16 a_hi x 16 b_lo
)


def _allocate_layer12_mul_combine_units() -> FFNUnitAllocator:
    """Build a per-bake :class:`FFNUnitAllocator` with L12's MUL combine range.

    The sub-stage is pinned at its existing offset so the FFNRule lowering
    -- which appends one hidden unit per rule starting at ``start_unit=0``
    -- lands on exactly the same hidden-unit indices the legacy
    ``_set_layer12_mul_combine`` helper always wrote. This call is
    byte-identical bookkeeping: the allocator declares the range by name,
    the rule lowering writes the weights. A future refactor can split the
    monolithic rule list into per-range bake functions that consume
    ``allocator.alloc(...)`` directly.

    Returns the allocator so callers can inspect or extend it (e.g. a
    future L12 op claims a free range past unit 4096 by widening the
    pool, or claims a sub-range if the rule list is split).
    """
    allocator = FFNUnitAllocator()
    for name, start, n_units in _L12_MUL_COMBINE_UNIT_LAYOUT:
        allocator.alloc(name, n_units, pin=start)
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
    """
    write_scale = 2.0 / S
    rules: list[FFNRule] = []

    for partial in range(16):
        for a_hi in range(16):
            for b_lo in range(16):
                result_hi = (partial + a_hi * b_lo) % 16
                rules.append(FFNRule.gated_write(
                    name=f"l12_mul_combine_p{partial:02d}_ah{a_hi:02d}_bl{b_lo:02d}",
                    conditions=(
                        ("MARK_AX", 1.0),
                        (f"TEMP+{partial}", 1.0),
                        (f"ALU_HI+{a_hi}", 1.0),
                        (f"AX_CARRY_LO+{b_lo}", 1.0),
                    ),
                    threshold=7.5,
                    gate="OP_MUL",
                    gate_weight=1.0,
                    gate_bias=0.0,
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
        bake_fn=bake,
        declarative_bake_fn=bake,
        compiler_ir=_layer12_mul_combine_ir(),
        declarative_authority="spec_generated",
        layer_idx=12,
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
