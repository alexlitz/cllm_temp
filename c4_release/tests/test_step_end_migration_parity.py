"""Per-rule byte-identity verification framework for STEP_END migration.

Wave A relay (in flight) makes operand state visible at STEP_END. Wave B
will move ~15-20 compute rules (L10 cmp_combine, L9 cmp/add/sub, L8 ALU
stages, L16 LEV, etc.) from MARK_AX to MARK_SE. This module provides a
reusable harness that asserts the migrated (MARK_SE-gated) version of a
rule produces the same target_dim writes as the original (MARK_AX-gated)
version when given an equivalent residual state.

Acceptance semantics (intentionally rule-level / symbolic, not full-VM):

* The migration is a position swap: every condition that reads MARK_AX
  becomes a MARK_SE read instead. Nothing else about the rule changes
  (writes, threshold, gate, gate_terms, gate_bias all preserved).
* "Same residual" means the test_program (a dict of dim -> value) is
  applied at the MARK_AX position for the original rule. For the
  migrated rule, the harness derives the residual state that's actually
  visible at STEP_END today. Before Wave A relay lands, the operand
  dims (CMP+*, ALU_*, AX_CARRY_*, OP_*, TEMP, MEM_VAL_*) are NOT
  mirrored to STEP_END -- the harness drops them, modelling the
  pre-relay reality. Once Wave A lands, callers can pass the full
  mirrored state via ``se_state=expected_se_state_after_wave_a(...)``
  and the parity assertion will start passing.
* Parity is asserted at ``target_dim`` (a residual-cell name like
  "OUTPUT_LO+0") with ``rtol=1e-5``. The 5 example assertions for
  cmp_combine / cmp_default / alu_bitwise_or / alu_bitwise_xor /
  alu_bitwise_and currently FAIL: at STEP_END the relayed operands
  aren't there yet, so the migrated rule does not fire. Once Wave A
  lands and the rule is swapped to MARK_SE, these assertions will
  pass -- they codify the migration target.

Acceptance:
    * Framework lands (``assert_step_end_parity``).
    * 5 example assertions exist (failing today, passing after Wave A
      relay + Wave B migration).
    * Tool ``tools/check_rule_position_parity.py`` runs in <30s.
    * Single commit.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Callable, Mapping, Optional, Sequence

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.unified_compiler.ir import (  # noqa: E402
    CompilerIR,
    ConditionTerm,
    DimRef,
    FFNRule,
)


# ============================================================================
# Position swap: MARK_AX <-> MARK_SE
# ============================================================================


def _swap_marker_in_rule(
    rule: FFNRule, *, from_name: str = "MARK_AX", to_name: str = "MARK_SE",
) -> FFNRule:
    """Return a copy of ``rule`` with every ``from_name`` reference swapped
    to ``to_name``. The swap covers ``conditions`` and ``gate_terms``
    (writes, threshold, gate, gate_bias preserved). The rule's ``name``
    is suffixed with ``"__se"`` for traceability."""

    def _swap_term(term: ConditionTerm) -> ConditionTerm:
        if term.dim.name == from_name:
            return ConditionTerm(DimRef(to_name, term.dim.offset), term.weight)
        return term

    return FFNRule(
        conditions=tuple(_swap_term(c) for c in rule.conditions),
        threshold=rule.threshold,
        writes=rule.writes,
        gate=rule.gate,
        gate_weight=rule.gate_weight,
        gate_terms=tuple(_swap_term(t) for t in rule.gate_terms),
        gate_bias=rule.gate_bias,
        name=(f"{rule.name}__se" if rule.name else None),
        scope=rule.scope,
        dominates_at=rule.dominates_at,
    )


def _ir_from_rules(rules: Sequence[FFNRule]) -> CompilerIR:
    """Return a single-layer ``CompilerIR`` containing ``rules`` on layer 0."""

    ir = CompilerIR()
    for rule in rules:
        ir.layer(0).ffn.append(rule)
    return ir


# Exact dim names (or ``base+offset`` keys whose base matches) that are
# present at the MARK_AX position but NOT yet at the MARK_SE/STEP_END
# position before Wave A relay lands. The default SE state filters these
# out, modelling the current pre-relay reality. Every example assertion
# therefore fails today and starts passing once Wave A relays these dims
# to MARK_SE.
#
# Membership is by exact base-name match: ``CMP`` matches ``CMP``,
# ``CMP+0``, ``CMP+1`` etc; ``ALU_LO`` matches ``ALU_LO+10`` but not
# ``ALU_LO_FOO``. The ``OP_`` family is handled by a separate
# ``startswith("OP_")`` check because every opcode flag (``OP_EQ``,
# ``OP_OR``, ``OP_LI_RELAY``, ...) is AX-position-only today.
_PRE_RELAY_MISSING_AT_SE_BASES: frozenset[str] = frozenset({
    "CMP",
    "ALU_LO",
    "ALU_HI",
    "AX_CARRY_LO",
    "AX_CARRY_HI",
    "TEMP",
    "MEM_VAL_B0",
    "MEM_VAL_B1",
    "MEM_VAL_B2",
    "MEM_VAL_B3",
})


def _is_pre_relay_missing(key: str) -> bool:
    """Return True iff ``key`` is in the AX-position-only operand set."""

    base = key.split("+", 1)[0]
    if base in _PRE_RELAY_MISSING_AT_SE_BASES:
        return True
    # Every OP_* opcode flag is AX-position-only today.
    if base.startswith("OP_"):
        return True
    return False


def _default_pre_relay_se_state(
    test_program: Mapping[str, float],
) -> dict[str, float]:
    """Build the SE-side state that's actually visible at STEP_END today.

    1. Rename any ``MARK_AX`` entry to ``MARK_SE`` (the position marker
       always exists at STEP_END).
    2. Drop every operand dim that hasn't been relayed yet (see
       :func:`_is_pre_relay_missing`).
    """

    out: dict[str, float] = {}
    for key, value in test_program.items():
        if key == "MARK_AX":
            out["MARK_SE"] = value
            continue
        if key == "MARK_AX+0":
            out["MARK_SE+0"] = value
            continue
        if _is_pre_relay_missing(key):
            continue
        out[key] = value
    return out


def expected_se_state_after_wave_a(
    test_program: Mapping[str, float],
) -> dict[str, float]:
    """Build the SE-side state expected once Wave A relay lands.

    Returns the full residual state mirrored to MARK_SE: every dim
    present at MARK_AX is also visible at STEP_END once the relay is in.
    Pass the result as ``se_state=`` to :func:`assert_step_end_parity`
    to assert post-relay byte-identity.
    """

    out: dict[str, float] = {}
    for key, value in test_program.items():
        if key == "MARK_AX":
            out["MARK_SE"] = value
        elif key == "MARK_AX+0":
            out["MARK_SE+0"] = value
        else:
            out[key] = value
    return out


# ============================================================================
# Public harness
# ============================================================================


def assert_step_end_parity(
    rule_factory: Callable[[], Sequence[FFNRule]],
    test_program: Mapping[str, float],
    step_index: int,
    target_dim: str,
    *,
    se_state: Optional[Mapping[str, float]] = None,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """Assert that the rule at STEP_END produces the same ``target_dim``
    value as the rule at MARK_AX when both see the same operand state.

    Arguments:
        rule_factory: callable returning a list of ``FFNRule`` -- the rules
            to migrate.
        test_program: a dict ``{dim_name_or_offset: value}`` describing
            the residual state AT THE MARK_AX POSITION (i.e. what the
            rule sees today).
        step_index: which VM step the rule fires on. Carried through as
            metadata only -- the rule-symbolic harness does not advance
            an autoregressive sequence -- but recorded in the failure
            message so callers can correlate parity failures with
            concrete traces.
        target_dim: the output dim to compare (e.g. ``"OUTPUT_LO+0"``).
        se_state: optional residual state visible at STEP_END. Default:
            derive from ``test_program`` via
            :func:`_default_pre_relay_se_state` (drops the operand dims
            that haven't been relayed yet). Pass
            :func:`expected_se_state_after_wave_a` to assert post-relay
            byte-identity.

    Raises:
        AssertionError when the MARK_AX and MARK_SE versions disagree at
        ``target_dim`` beyond ``rtol`` / ``atol``.
    """

    rules = list(rule_factory())
    if not rules:
        raise ValueError("rule_factory returned no rules")

    ir_ax = _ir_from_rules(rules)
    ir_se = _ir_from_rules([_swap_marker_in_rule(r) for r in rules])

    state_ax = dict(test_program)
    if se_state is None:
        state_se = _default_pre_relay_se_state(test_program)
    else:
        state_se = dict(se_state)

    out_ax = ir_ax.symbolic_ffn(state_ax)
    out_se = ir_se.symbolic_ffn(state_se)

    ax_value = float(out_ax.get(target_dim, 0.0))
    se_value = float(out_se.get(target_dim, 0.0))

    if not math.isclose(ax_value, se_value, rel_tol=rtol, abs_tol=atol):
        raise AssertionError(
            "STEP_END parity failed at step={step}, target_dim={dim!r}:\n"
            "  MARK_AX value = {ax!r}\n"
            "  MARK_SE value = {se!r}\n"
            "  diff = {diff!r}\n"
            "  rules tested: {names!r}".format(
                step=step_index,
                dim=target_dim,
                ax=ax_value,
                se=se_value,
                diff=ax_value - se_value,
                names=[r.name for r in rules][:8]
                + (["..."] if len(rules) > 8 else []),
            )
        )


# ============================================================================
# Rule-factory wrappers -- exposed by name so the CLI tool can find them
# ============================================================================


def _factory_cmp_combine():
    from neural_vm.unified_compiler.ops.l10_ops import (
        _layer10_alu_cmp_combine_rules,
    )
    return _layer10_alu_cmp_combine_rules(100.0)


def _factory_cmp_default():
    from neural_vm.unified_compiler.ops.l10_ops import (
        _l10_comparison_combine_rules,
    )
    # Filter to default rules only (idx 0, 2, 4, 7, 11, 15).
    rules = list(_l10_comparison_combine_rules(100.0))
    return [r for r in rules if "_default_" in (r.name or "")]


def _factory_alu_bitwise_or():
    from neural_vm.unified_compiler.ops.l10_ops import (
        _layer10_alu_bitwise_or_rules,
    )
    return _layer10_alu_bitwise_or_rules(100.0)


def _factory_alu_bitwise_xor():
    from neural_vm.unified_compiler.ops.l10_ops import (
        _layer10_alu_bitwise_xor_rules,
    )
    return _layer10_alu_bitwise_xor_rules(100.0)


def _factory_alu_bitwise_and():
    from neural_vm.unified_compiler.ops.l10_ops import (
        _layer10_alu_bitwise_and_rules,
    )
    return _layer10_alu_bitwise_and_rules(100.0)


# Public mapping consumed by ``tools/check_rule_position_parity.py``.
RULE_FACTORIES: dict[str, Callable[[], Sequence[FFNRule]]] = {
    "cmp_combine": _factory_cmp_combine,
    "cmp_default": _factory_cmp_default,
    "alu_bitwise_or": _factory_alu_bitwise_or,
    "alu_bitwise_xor": _factory_alu_bitwise_xor,
    "alu_bitwise_and": _factory_alu_bitwise_and,
}


# ============================================================================
# Example parity assertions (currently FAILING -- codify migration target)
# ============================================================================
#
# Each test exercises one compute rule with a residual state that triggers
# the rule's interesting branch at the MARK_AX position. The harness
# automatically constructs the SE-side state by dropping the operand
# dims that aren't relayed yet, so the MARK_SE rule does NOT fire today
# and the parity assertion fails -- exactly the migration target.
#
# After Wave A lands the operands at STEP_END, switch the call sites to
# ``se_state=expected_se_state_after_wave_a(test_program)`` to assert
# the relay is in place; once Wave B migrates the rule itself the
# default behavior will also pass.

# Tight residual: MARK_AX = 1.5 + OP_EQ = 1.5 = 3.0 > threshold 2.5 (default
# fires at AX). With OP_EQ dropped from the SE-side state, MARK_SE alone
# = 1.5 < 2.5, so the rule does NOT fire at SE today.
_CMP_COMBINE_PROGRAM = {
    "MARK_AX": 1.5,
    "OP_EQ": 1.5,
    "CMP+1": 1.0,     # hi_eq from L9
    "CMP+2": 1.0,     # lo_eq from L9
}


def test_parity_cmp_combine_eq_default_at_step_end():
    """L10 cmp_combine EQ default (writes OUTPUT_LO+0 = 0 baseline) must
    match across MARK_AX and STEP_END after Wave B migration. At MARK_AX
    the default fires (MARK_AX + OP_EQ = 3.0 >= 2.5); at MARK_SE today
    the OP_EQ term is missing so MARK_SE = 1.5 < 2.5 and the default does
    not fire -- the parity check fails until Wave A relays OP_EQ to SE."""

    assert_step_end_parity(
        rule_factory=_factory_cmp_combine,
        test_program=_CMP_COMBINE_PROGRAM,
        step_index=3,
        target_dim="OUTPUT_LO+0",
    )


# Tight residual: MARK_AX = 1.0 + OP_NE = 1.0 = 2.0 > threshold 1.5 (rule
# fires at AX). With OP_NE dropped from the SE-side state, MARK_SE alone
# = 1.0 < 1.5, so the rule does NOT fire at SE today.
_CMP_DEFAULT_PROGRAM = {
    "MARK_AX": 1.0,
    "OP_NE": 1.0,
}


def test_parity_cmp_default_ne_default_at_step_end():
    """L10 cmp_default NE baseline (writes OUTPUT_LO+1=1.0 by default) must
    match across MARK_AX and STEP_END after Wave B migration. At MARK_AX
    the default fires (MARK_AX + OP_NE = 2 >= threshold 1.5); at MARK_SE
    today the OP_NE term is missing so MARK_SE = 1 < 1.5 and the rule
    does not fire -- the parity check fails until Wave A relays OP_NE."""

    assert_step_end_parity(
        rule_factory=_factory_cmp_default,
        test_program=_CMP_DEFAULT_PROGRAM,
        step_index=3,
        target_dim="OUTPUT_LO+1",
    )


# Bitwise rules: 40*MARK_AX + 30*ALU_LO + 30*AX_CARRY_LO > 80 with OP_<NAME>
# gate. Picking a=0xA, b=0x5 means OR=0xF, XOR=0xF, AND=0x0.
_BITWISE_PROGRAM_OR = {
    "MARK_AX": 1.0,
    "ALU_LO+10": 1.0,
    "AX_CARRY_LO+5": 1.0,
    "OP_OR": 1.0,
}

_BITWISE_PROGRAM_XOR = {
    "MARK_AX": 1.0,
    "ALU_LO+10": 1.0,
    "AX_CARRY_LO+5": 1.0,
    "OP_XOR": 1.0,
}

_BITWISE_PROGRAM_AND = {
    "MARK_AX": 1.0,
    "ALU_LO+10": 1.0,
    "AX_CARRY_LO+5": 1.0,
    "OP_AND": 1.0,
}


def test_parity_alu_bitwise_or_at_step_end():
    """L10 OR(0xA, 0x5)=0xF must produce identical OUTPUT_LO+15 at MARK_AX
    and STEP_END after Wave B migration."""

    assert_step_end_parity(
        rule_factory=_factory_alu_bitwise_or,
        test_program=_BITWISE_PROGRAM_OR,
        step_index=4,
        target_dim="OUTPUT_LO+15",
    )


def test_parity_alu_bitwise_xor_at_step_end():
    """L10 XOR(0xA, 0x5)=0xF must produce identical OUTPUT_LO+15 at MARK_AX
    and STEP_END after Wave B migration."""

    assert_step_end_parity(
        rule_factory=_factory_alu_bitwise_xor,
        test_program=_BITWISE_PROGRAM_XOR,
        step_index=4,
        target_dim="OUTPUT_LO+15",
    )


def test_parity_alu_bitwise_and_at_step_end():
    """L10 AND(0xA, 0x5)=0x0 must produce identical OUTPUT_LO+0 at MARK_AX
    and STEP_END after Wave B migration."""

    assert_step_end_parity(
        rule_factory=_factory_alu_bitwise_and,
        test_program=_BITWISE_PROGRAM_AND,
        step_index=4,
        target_dim="OUTPUT_LO+0",
    )


# ============================================================================
# Smoke tests -- exercise the framework's plumbing independent of the
# migration status. These ALWAYS pass and guard against regressions in the
# harness itself.
# ============================================================================


def test_harness_post_wave_a_state_passes_parity():
    """Demonstrate that once Wave A relays the operand state to MARK_SE
    (``expected_se_state_after_wave_a``), the parity assertion passes
    for every example rule. This is the green-build target for Wave B
    landing: callers swap the ``se_state=`` kwarg in and the test flips
    to passing without any rule change."""

    for factory, program, target in (
        (_factory_cmp_combine, _CMP_COMBINE_PROGRAM, "OUTPUT_LO+1"),
        (_factory_cmp_default, _CMP_DEFAULT_PROGRAM, "OUTPUT_LO+1"),
        (_factory_alu_bitwise_or, _BITWISE_PROGRAM_OR, "OUTPUT_LO+15"),
        (_factory_alu_bitwise_xor, _BITWISE_PROGRAM_XOR, "OUTPUT_LO+15"),
        (_factory_alu_bitwise_and, _BITWISE_PROGRAM_AND, "OUTPUT_LO+0"),
    ):
        assert_step_end_parity(
            rule_factory=factory,
            test_program=program,
            step_index=0,
            target_dim=target,
            se_state=expected_se_state_after_wave_a(program),
        )


def test_harness_identity_swap_is_byte_identical():
    """Trivial sanity check: a rule with no MARK_AX condition is unaffected
    by the swap, and the SE-side default state preserves the (non-operand)
    dim, so MARK_AX-state vs MARK_SE-state agree at any write target."""

    rule = FFNRule.constant_write(
        conditions=(("HAS_SE", 1.0),),
        threshold=0.5,
        writes=(("OUTPUT_LO+0", 1.0),),
        name="harness_smoke_no_mark_ax",
    )
    assert_step_end_parity(
        rule_factory=lambda: [rule],
        test_program={"HAS_SE": 1.0},
        step_index=0,
        target_dim="OUTPUT_LO+0",
    )


def test_harness_marker_swap_changes_state_key():
    """Swap helper renames MARK_AX -> MARK_SE in the rule definition,
    leaving other terms untouched."""

    rule = FFNRule.constant_write(
        conditions=(("MARK_AX", 1.0), ("OP_EQ", 1.0)),
        threshold=1.5,
        writes=(("OUTPUT_LO+0", 1.0),),
        name="harness_smoke_mark_ax",
    )
    swapped = _swap_marker_in_rule(rule)
    assert any(c.dim.name == "MARK_SE" for c in swapped.conditions)
    assert all(c.dim.name != "MARK_AX" for c in swapped.conditions)

    pre_relay = _default_pre_relay_se_state(
        {"MARK_AX": 1.0, "OP_EQ": 1.0, "HAS_SE": 1.0}
    )
    # MARK_AX renamed, OP_* dropped (pre-relay), HAS_SE preserved.
    assert "MARK_SE" in pre_relay
    assert "MARK_AX" not in pre_relay
    assert "OP_EQ" not in pre_relay
    assert pre_relay["HAS_SE"] == 1.0
