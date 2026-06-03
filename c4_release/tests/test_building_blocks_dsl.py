"""Byte-identity tests for ``building_blocks_dsl``.

Validation contract (per the V1 plan):
  * Single-rule primitives (``step_function_rule``, ``multi_way_and_rule``,
    ``cancel_residual_rule``, ``one_hot_indicator_rule``) —
    ``compare_symbolic_to_lowered_ffn`` at S=1.0 with calibrated state.
  * Multi-rule primitives (``band_range_check_rules``,
    ``lookup_table_rules``, ``multi_way_or_rules``) — direct neural
    forward at S=100 with binary one-hot inputs, compared to ground
    truth.

The S=1 + calibrated-state form is used for single-rule patterns where
``_SILU_ONE_INPUT`` makes silu(_calibrated_input_) = 1.0 exactly. The
S=100 + one-hot form is used for multi-rule patterns over discrete
bands where every fired rule sees a clean ``silu(S*0.5) ≈ S*0.5``
output that scales cleanly with ``write_weight = 2/S``.
"""

from __future__ import annotations

import math

import pytest
import torch

from c4_release.neural_vm.base_layers import PureFFN
from c4_release.neural_vm.unified_compiler.building_blocks_dsl import (
    band_range_check_rules,
    cancel_residual_rule,
    lookup_table_rules,
    multi_way_and_rule,
    multi_way_or_rules,
    one_hot_indicator_rule,
    step_function_rule,
)
from c4_release.neural_vm.unified_compiler.ir import (
    compare_symbolic_to_lowered_ffn,
)
from c4_release.neural_vm.unified_compiler.primitives import Primitives


_SILU_ONE_INPUT = 1.278464542761074


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lowered_ffn_at_S(rules, dim_positions, *, S: float = 100.0) -> PureFFN:
    """Build and lower a fresh ``PureFFN`` for a rule list at scale ``S``."""

    n_rules = len(rules)
    d_model = max(dim_positions.values()) + 32
    ffn = PureFFN(dim=d_model, hidden_dim=max(n_rules, 1))
    end = Primitives.lower_ffn_rules(
        ffn, rules, dim_positions, start_unit=0, S=S,
    )
    assert end == n_rules
    return ffn


def _forward_at(
    ffn: PureFFN, dim_positions, state: dict, *, probe_offsets: int = 16
) -> dict:
    """Forward pass returning FFN delta at each ``base+offset``."""

    d_model = ffn.W_up.shape[1]
    x = torch.zeros(1, 1, d_model)
    for key, value in state.items():
        if "+" in key:
            base, off = key.rsplit("+", 1)
            x[0, 0, dim_positions[base] + int(off)] = float(value)
        else:
            x[0, 0, dim_positions[key]] = float(value)
    with torch.no_grad():
        y = ffn(x)
    out: dict = {}
    for name, base_pos in dim_positions.items():
        for off in range(probe_offsets):
            pos = base_pos + off
            if pos >= d_model:
                break
            out[f"{name}+{off}"] = float(y[0, 0, pos].item())
        out[name] = float(y[0, 0, base_pos].item())
    return out


# ===========================================================================
# step_function_rule
# ===========================================================================


def test_step_function_rule_symbolic_matches_lowered_at_S1():
    """Single step at threshold=0.5 reads X, writes OUT+0. Calibrated
    input X = 0.5 + _SILU_ONE_INPUT makes silu = 1.0 exactly at S=1.
    """
    rule = step_function_rule(
        input_dim="X",
        threshold=0.5,
        write_dim="OUT+0",
        write_value=3.0,
        S=1.0,
        name="step_at_half",
    )
    report = compare_symbolic_to_lowered_ffn(
        rule,
        {"X": 0, "OUT": 1},
        {"X": 0.5 + _SILU_ONE_INPUT},
        S=1.0,
        atol=1e-5,
    )
    assert report.ok, report.format()
    assert math.isclose(report.symbolic_state["OUT+0"], 3.0, abs_tol=1e-9)
    assert math.isclose(report.lowered_state["OUT+0"], 3.0, abs_tol=1e-5)


def test_step_function_rule_blocks_below_threshold_neural():
    """At S=100, input=0 must not fire the step at threshold=0.5."""

    rule = step_function_rule(
        input_dim="X", threshold=0.5, write_dim="OUT+0", write_value=2.0,
        S=100.0,
    )
    dim_positions = {"X": 0, "OUT": 1}
    ffn = _lowered_ffn_at_S([rule], dim_positions, S=100.0)
    out = _forward_at(ffn, dim_positions, {"X": 0.0})
    assert abs(out["OUT+0"]) < 1e-4


def test_step_function_rule_with_gate_uses_gated_write():
    rule = step_function_rule(
        input_dim="X", threshold=0.5, write_dim="OUT+0", write_value=2.0,
        gate="G", S=100.0,
    )
    assert rule.gate is not None
    assert rule.gate.name == "G"
    # Default gate_bias for gated_write must be 0.
    assert rule.gate_bias == 0.0


# ===========================================================================
# one_hot_indicator_rule
# ===========================================================================


@pytest.mark.parametrize("active_value", [0, 1, 7, 15])
@pytest.mark.parametrize("target_value", [0, 5, 7, 10])
def test_one_hot_indicator_rule_fires_only_at_target(active_value, target_value):
    """Indicator at ``BAND+target`` fires iff the active cell is the
    target. One-hot input: exactly one cell of BAND is 1.0.
    """
    rule = one_hot_indicator_rule(
        band="BAND",
        value=target_value,
        write_dim="OUT+0",
        write_value=2.0,
        S=100.0,
        name="ind",
    )
    dim_positions = {"BAND": 0, "OUT": 16}
    ffn = _lowered_ffn_at_S([rule], dim_positions, S=100.0)
    state = {f"BAND+{active_value}": 1.0}
    out = _forward_at(ffn, dim_positions, state)
    expected = 1.0 if active_value == target_value else 0.0
    assert abs(out["OUT+0"] - expected) < 5e-4, (
        f"one_hot[v={target_value}] when active={active_value}: "
        f"got {out['OUT+0']!r}, expected {expected!r}"
    )


def test_one_hot_indicator_rule_symbolic_matches_lowered_at_S1():
    """At S=1 with calibrated input, indicator output = write_value."""

    rule = one_hot_indicator_rule(
        band="BAND",
        value=3,
        write_dim="OUT+0",
        write_value=4.0,
        S=1.0,
        name="ind_v3",
    )
    report = compare_symbolic_to_lowered_ffn(
        rule,
        {"BAND": 0, "OUT": 16},
        {"BAND+3": 0.5 + _SILU_ONE_INPUT},
        S=1.0,
        atol=1e-5,
    )
    assert report.ok, report.format()
    assert math.isclose(report.symbolic_state["OUT+0"], 4.0, abs_tol=1e-9)
    assert math.isclose(report.lowered_state["OUT+0"], 4.0, abs_tol=1e-5)


# ===========================================================================
# band_range_check_rules
# ===========================================================================


@pytest.mark.parametrize("active", list(range(16)))
def test_band_range_check_rules_within_band(active):
    """range over [3,8] of a 16-cell band fires iff active cell ∈ [3,8]."""

    rules = band_range_check_rules(
        band="BAND",
        lo=3,
        hi=8,
        write_dim="OUT+0",
        write_value=2.0,
        S=100.0,
        name="rng",
    )
    assert len(rules) == 6  # hi-lo+1 = 6 rules

    dim_positions = {"BAND": 0, "OUT": 16}
    ffn = _lowered_ffn_at_S(rules, dim_positions, S=100.0)
    out = _forward_at(ffn, dim_positions, {f"BAND+{active}": 1.0})
    expected = 1.0 if 3 <= active <= 8 else 0.0
    assert abs(out["OUT+0"] - expected) < 5e-4, (
        f"range[3,8] at BAND+{active}: got {out['OUT+0']!r}, expected "
        f"{expected!r}"
    )


def test_band_range_check_rules_rejects_empty_range():
    with pytest.raises(ValueError, match="empty range"):
        band_range_check_rules(
            band="BAND", lo=8, hi=3, write_dim="OUT+0", S=100.0,
        )


# ===========================================================================
# multi_way_and_rule
# ===========================================================================


def test_multi_way_and_rule_default_threshold_fires_only_when_all_set():
    """Default 3-way AND uniform-weight: fires iff every condition is on."""

    rule = multi_way_and_rule(
        conditions=(("A", 1.0), ("B", 1.0), ("C", 1.0)),
        writes=(("OUT+0", 2.0 / 100.0),),
        name="and3",
    )
    # Default derivation: total=3, max_w=1, threshold=(3+2)/2=2.5
    assert rule.threshold == 2.5

    dim_positions = {"A": 0, "B": 1, "C": 2, "OUT": 3}
    ffn = _lowered_ffn_at_S([rule], dim_positions, S=100.0)
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                out = _forward_at(
                    ffn, dim_positions,
                    {"A": float(a), "B": float(b), "C": float(c)},
                )
                expected = 1.0 if (a and b and c) else 0.0
                assert abs(out["OUT+0"] - expected) < 1e-3, (
                    f"AND({a},{b},{c}): got {out['OUT+0']!r}, expected "
                    f"{expected!r}"
                )


def test_multi_way_and_rule_replicates_bitwise_dsl_balanced_weights():
    """Legacy (40, 30, 30) > 80 invariant survives via explicit args."""

    rule = multi_way_and_rule(
        conditions=(("MARKER", 40.0), ("A", 30.0), ("B", 30.0)),
        threshold=80.0,
        writes=(("OUT+0", 2.0 / 100.0),),
        name="and_bitwise",
    )
    assert rule.threshold == 80.0
    assert 40 + 30 + 30 > 80
    assert max(40 + 30, 40 + 30, 30 + 30) < 80


def test_multi_way_and_rule_rejects_empty_conditions():
    with pytest.raises(ValueError, match="non-empty"):
        multi_way_and_rule(conditions=(), writes=(("OUT", 1.0),))


# ===========================================================================
# multi_way_or_rules
# ===========================================================================


def test_multi_way_or_rules_emits_one_rule_per_condition():
    rules = multi_way_or_rules(
        conditions=("A", "B", "C"),
        writes=(("OUT+0", 2.0 / 100.0),),
        S=100.0,
    )
    assert len(rules) == 3
    for rule, cond in zip(rules, ("A", "B", "C")):
        assert len(rule.conditions) == 1
        assert rule.conditions[0].dim.name == cond


def test_multi_way_or_rules_accumulates_active_conditions():
    """For one-hot the OR fires once, for overlapping it accumulates."""

    rules = multi_way_or_rules(
        conditions=("A", "B", "C"),
        writes=(("OUT+0", 2.0 / 100.0),),
        S=100.0,
    )
    dim_positions = {"A": 0, "B": 1, "C": 2, "OUT": 3}
    ffn = _lowered_ffn_at_S(rules, dim_positions, S=100.0)
    for a, b, c in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (1, 1, 1)]:
        out = _forward_at(
            ffn, dim_positions,
            {"A": float(a), "B": float(b), "C": float(c)},
        )
        expected = float(a + b + c)
        assert abs(out["OUT+0"] - expected) < 5e-4, (
            f"OR({a},{b},{c}): got {out['OUT+0']!r}, expected {expected!r}"
        )


# ===========================================================================
# cancel_residual_rule
# ===========================================================================


def test_cancel_residual_rule_self_cancel_symbolic_at_S1():
    """Self-cancel: when X is set, X += -write_value (cancels at S=1
    calibrated input)."""

    rule = cancel_residual_rule(
        input_dim="X", write_value=3.0, S=1.0, name="cancel",
    )
    report = compare_symbolic_to_lowered_ffn(
        rule,
        {"X": 0},
        {"X": 0.5 + _SILU_ONE_INPUT},
        S=1.0,
        atol=1e-5,
    )
    assert report.ok, report.format()
    # symbolic: out[X+0] = input[X+0] + gate_bias*write = 1.778 + (1 * -3) = -1.222
    assert math.isclose(
        report.symbolic_state["X+0"],
        (0.5 + _SILU_ONE_INPUT) - 3.0,
        abs_tol=1e-5,
    )


def test_cancel_residual_rule_separate_output_dim():
    rule = cancel_residual_rule(
        input_dim="X", output_dim="OUT+0", write_value=2.0, S=100.0,
    )
    assert len(rule.writes) == 1
    w = rule.writes[0]
    assert w.dim.name == "OUT"
    assert w.dim.offset == 0
    assert w.weight == -2.0 / 100.0


# ===========================================================================
# lookup_table_rules
# ===========================================================================


def test_lookup_table_rules_keyed_by_one_hot_key_band():
    """A 4-entry lookup keyed on KEY band; each key writes to a distinct
    output slot.
    """
    key_to_writes = {
        0: [("OUT+0", 1.0)],
        1: [("OUT+1", 1.0)],
        2: [("OUT+2", 1.0)],
        3: [("OUT+3", 1.0)],
    }
    rules = lookup_table_rules(
        key_band="KEY",
        key_to_writes=key_to_writes,
        S=100.0,
        name_prefix="lut",
    )
    assert len(rules) == 4  # one rule per key, not 2

    dim_positions = {"KEY": 0, "OUT": 16}
    ffn = _lowered_ffn_at_S(rules, dim_positions, S=100.0)

    # At S=100 with write weight = 1.0 / S, silu(50) * 1/S ≈ 0.5 per fire.
    # For multiplicative output magnitude, we typically pass weight=2.0
    # and divide by S — verify default scale here gives 0.5 per fire.
    for key in range(4):
        out = _forward_at(ffn, dim_positions, {f"KEY+{key}": 1.0})
        for other in range(4):
            slot = f"OUT+{other}"
            expected = 0.5 if other == key else 0.0
            assert abs(out[slot] - expected) < 5e-4, (
                f"lookup key={key} slot={slot}: got {out[slot]!r}, "
                f"expected {expected!r}"
            )


def test_lookup_table_rules_with_write_value_scale_2():
    """``write_value_scale=2.0`` gives the canonical 2/S amplitude
    (matches ``bitwise_rules`` convention).
    """
    key_to_writes = {
        0: [("OUT+0", 1.0)],
        1: [("OUT+1", 1.0)],
    }
    rules = lookup_table_rules(
        key_band="KEY",
        key_to_writes=key_to_writes,
        write_value_scale=2.0,
        S=100.0,
    )
    dim_positions = {"KEY": 0, "OUT": 16}
    ffn = _lowered_ffn_at_S(rules, dim_positions, S=100.0)
    out = _forward_at(ffn, dim_positions, {"KEY+0": 1.0})
    # Output ≈ 50 * 2.0 / 100 = 1.0
    assert abs(out["OUT+0"] - 1.0) < 5e-4


# ===========================================================================
# Sanity
# ===========================================================================


def test_all_primitives_return_ffn_rule_objects():
    from c4_release.neural_vm.unified_compiler.ir import FFNRule

    assert isinstance(
        step_function_rule(input_dim="X", threshold=0.5, write_dim="OUT+0"),
        FFNRule,
    )
    assert isinstance(
        one_hot_indicator_rule(band="B", value=3, write_dim="OUT+0"),
        FFNRule,
    )
    band_rng = band_range_check_rules(
        band="B", lo=0, hi=3, write_dim="OUT+0",
    )
    assert isinstance(band_rng, tuple) and len(band_rng) == 4
    for r in band_rng:
        assert isinstance(r, FFNRule)

    and_rule = multi_way_and_rule(
        conditions=(("A", 1.0), ("B", 1.0)),
        writes=(("OUT+0", 0.02),),
    )
    assert isinstance(and_rule, FFNRule)

    or_rules = multi_way_or_rules(
        conditions=("A", "B"),
        writes=(("OUT+0", 0.02),),
    )
    assert isinstance(or_rules, tuple) and len(or_rules) == 2

    cancel = cancel_residual_rule(input_dim="X")
    assert isinstance(cancel, FFNRule)

    lut = lookup_table_rules(
        key_band="K", key_to_writes={0: [("OUT+0", 1.0)]},
    )
    assert isinstance(lut, tuple) and len(lut) == 1
