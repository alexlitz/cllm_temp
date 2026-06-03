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

from c4_release.neural_vm.base_layers import PureAttention, PureFFN
from c4_release.neural_vm.kv_cache_eviction import softmax1
from c4_release.neural_vm.unified_compiler.building_blocks_dsl import (
    band_range_check_rules,
    bit_range_extract_rules,
    cancel_residual_rule,
    efficient_exp_attention,
    fetch_byte_attention,
    lookup_table_rules,
    magic_floor_rules,
    memory_load_attention,
    multi_way_and_rule,
    multi_way_or_rules,
    one_hot_indicator_rule,
    opcode_expert_rules,
    step_function_rule,
)
from c4_release.neural_vm.unified_compiler.ir import (
    FFNRule,
    compare_symbolic_to_lowered_ffn,
)
from c4_release.neural_vm.unified_compiler.primitives import (
    DeclarativeAttentionHeadSpec,
    Primitives,
)


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


def test_multi_way_and_rule_passes_non_zero_gate_bias_through():
    """Non-zero ``gate_bias`` reaches the underlying FFNRule.

    Used by L9 patterns like ``_layer9_bp_plus8_shift`` (gate_bias=-2.5)
    and ``_layer9_addr_b1_set_and_cascade`` (gate_bias=-15.0).
    """
    rule = multi_way_and_rule(
        conditions=(("A", 1.0),),
        threshold=0.5,
        writes=(("OUT+0", 0.02),),
        gate="G",
        gate_weight=5.0,
        gate_bias=-3.0,
        name="bias_test",
    )
    assert rule.gate is not None
    assert rule.gate_weight == 5.0
    assert rule.gate_bias == -3.0


def test_multi_way_and_rule_default_gate_bias_is_zero_with_gate():
    rule = multi_way_and_rule(
        conditions=(("A", 1.0),),
        threshold=0.5,
        writes=(("OUT+0", 0.02),),
        gate="G",
    )
    assert rule.gate_bias == 0.0


def test_multi_way_and_rule_default_gate_bias_is_one_without_gate():
    rule = multi_way_and_rule(
        conditions=(("A", 1.0),),
        threshold=0.5,
        writes=(("OUT+0", 0.02),),
    )
    # constant_write path; gate_bias defaults to 1.0
    assert rule.gate_bias == 1.0


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


# ===========================================================================
# V2 helpers
# ===========================================================================


def _floor_ffn_at_S1(rules, dim_positions):
    """Build a fresh PureFFN sized for ``rules`` and lower at S=1 (the
    scale that ``magic_floor_rules`` / ``bit_range_extract_rules``
    require)."""

    n_rules = len(rules)
    d_model = max(dim_positions.values()) + 8
    ffn = PureFFN(dim=d_model, hidden_dim=max(n_rules, 1))
    end = Primitives.lower_ffn_rules(
        ffn, rules, dim_positions, start_unit=0, S=1.0,
    )
    assert end == n_rules
    return ffn


def _ffn_delta(ffn, dim_positions, state):
    """Run one FFN forward pass and return the per-dim delta the layer
    contributed (forward output minus the input residual)."""

    d_model = ffn.W_up.shape[1]
    x = torch.zeros(1, 1, d_model, dtype=torch.float32)
    for key, value in state.items():
        if "+" in key:
            base, off = key.rsplit("+", 1)
            x[0, 0, dim_positions[base] + int(off)] = float(value)
        else:
            x[0, 0, dim_positions[key]] = float(value)
    with torch.no_grad():
        y = ffn(x)
    delta = (y - x)[0, 0]
    out = {}
    for name, base_pos in dim_positions.items():
        out[name] = float(delta[base_pos].item())
    return out


# ===========================================================================
# magic_floor_rules
# ===========================================================================


@pytest.mark.parametrize(
    "value",
    [0.0, 0.4, 1.0, 1.5, 1.7, 2.0, 3.4, 7.9, 100.5, 999.0, 100000.7],
)
def test_magic_floor_rules_matches_python_floor(value):
    rules = magic_floor_rules(
        input_dim="X", output_dim="Y", const_dim="CONST", name="mf",
    )
    assert len(rules) == 2
    dim_positions = {"X": 0, "Y": 1, "CONST": 2}
    ffn = _floor_ffn_at_S1(rules, dim_positions)
    delta = _ffn_delta(ffn, dim_positions, {"X": value, "CONST": 1.0})
    expected = math.floor(value)
    assert abs(delta["Y"] - expected) < 1e-3, (
        f"magic_floor({value}): got {delta['Y']!r}, expected {expected}"
    )


def test_magic_floor_rules_emits_two_rules_of_correct_kinds():
    rules = magic_floor_rules(input_dim="X", output_dim="Y")
    assert len(rules) == 2
    floor_unit, cancel_unit = rules
    # The floor unit reads input + the offset const dim; the cancel unit
    # is pure-constant (no conditions).
    assert len(floor_unit.conditions) == 2
    assert floor_unit.conditions[0].dim.name == "X"
    assert floor_unit.conditions[1].dim.name == "CONST"
    assert len(cancel_unit.conditions) == 0
    # Both writes target the same output dim with opposite signs.
    assert floor_unit.writes[0].dim.name == "Y"
    assert cancel_unit.writes[0].dim.name == "Y"
    assert floor_unit.writes[0].weight == +1.0
    assert cancel_unit.writes[0].weight == -1.0


def test_magic_floor_rules_with_gate_uses_gated_write():
    rules = magic_floor_rules(
        input_dim="X", output_dim="Y", gate="OP_FLOOR",
    )
    for r in rules:
        assert r.gate is not None
        assert r.gate.name == "OP_FLOOR"
        # gated_write sets gate_bias=0 (not 1).
        assert r.gate_bias == 0.0


# ===========================================================================
# bit_range_extract_rules
# ===========================================================================


@pytest.mark.parametrize(
    "value", [0, 0x12, 0x3A, 0x80, 0xFF, 0x100, 0xC, 0x123, 0x1234],
)
@pytest.mark.parametrize(
    "lo_bit, hi_bit", [(0, 4), (4, 8), (0, 8), (8, 16), (4, 12)],
)
def test_bit_range_extract_rules_lowers_correctly(value, lo_bit, hi_bit):
    rules = bit_range_extract_rules(
        input_dim="X",
        lo_shift_dim="LO",
        hi_shift_dim="HI",
        lo_bit=lo_bit,
        hi_bit=hi_bit,
        name="extract",
    )
    assert len(rules) == 4
    dim_positions = {"X": 0, "LO": 1, "HI": 2, "CONST": 3}
    ffn = _floor_ffn_at_S1(rules, dim_positions)
    delta = _ffn_delta(ffn, dim_positions, {"X": float(value), "CONST": 1.0})
    width = hi_bit - lo_bit
    expected_lo = (value >> lo_bit)
    expected_hi = (value >> hi_bit)
    expected_combine = expected_lo - (1 << width) * expected_hi
    expected_bits = (value >> lo_bit) & ((1 << width) - 1)
    assert expected_combine == expected_bits, (
        "test sanity: combine identity should equal masked extract"
    )
    combine = delta["LO"] - float(1 << width) * delta["HI"]
    assert abs(combine - expected_bits) < 1e-3, (
        f"bit_range_extract({value:#x}, lo={lo_bit}, hi={hi_bit}): "
        f"lo={delta['LO']}, hi={delta['HI']}, combine={combine}, "
        f"expected_bits={expected_bits}"
    )


def test_bit_range_extract_rules_rejects_invalid_ranges():
    with pytest.raises(ValueError, match="lo_bit"):
        bit_range_extract_rules(
            input_dim="X", lo_shift_dim="LO", hi_shift_dim="HI",
            lo_bit=-1, hi_bit=4,
        )
    with pytest.raises(ValueError, match="hi_bit"):
        bit_range_extract_rules(
            input_dim="X", lo_shift_dim="LO", hi_shift_dim="HI",
            lo_bit=4, hi_bit=4,
        )
    with pytest.raises(ValueError, match="hi_bit"):
        bit_range_extract_rules(
            input_dim="X", lo_shift_dim="LO", hi_shift_dim="HI",
            lo_bit=4, hi_bit=24,
        )


# ===========================================================================
# efficient_exp_attention
# ===========================================================================


def _softmax1_attn_forward(x, attn_mod):
    """Run a single-head PureAttention forward using softmax1 instead of
    F.softmax. PureAttention defaults to F.softmax; the blog's
    efficient-exp construction depends on softmax1's +1 anchor, so the
    test replays the same matrix products with the correct
    normalization."""

    HD = attn_mod.head_dim
    Q = x @ attn_mod.W_q.T
    K = x @ attn_mod.W_k.T
    V = x @ attn_mod.W_v.T
    scores = Q @ K.transpose(-1, -2) / math.sqrt(HD)
    weights = softmax1(scores, dim=-1, anchor=0.0)
    out = weights @ V
    return x + out @ attn_mod.W_o.T


def test_efficient_exp_attention_spec_structure():
    dim_positions = {"N": 0, "OUT": 1, "BOS_K": 2, "CONST": 3}
    spec = efficient_exp_attention(
        head_idx=0, input_dim="N", output_dim="OUT",
        bos_token_key_dim="BOS_K", bias=2.5, head_dim=8,
        dim_positions=dim_positions,
    )
    assert isinstance(spec, DeclarativeAttentionHeadSpec)
    assert spec.head_idx == 0
    assert spec.alibi_slope == 0.0
    # Q reads input_dim with +1 and bias_dim with -bias.
    assert len(spec.q) == 2
    weights_by_dim = {w.dim: w.weight for w in spec.q}
    assert weights_by_dim[dim_positions["N"]] == 1.0
    assert weights_by_dim[dim_positions["CONST"]] == -2.5
    # K reads bos_token_key_dim with weight sqrt(head_dim).
    assert len(spec.k) == 1
    assert spec.k[0].dim == dim_positions["BOS_K"]
    assert math.isclose(spec.k[0].weight, math.sqrt(8.0))
    # V reads bos_token_key_dim with weight e^B.
    assert len(spec.v) == 1
    assert spec.v[0].dim == dim_positions["BOS_K"]
    assert math.isclose(spec.v[0].weight, math.exp(2.5))
    # O writes the V slot back to OUT.
    assert len(spec.o) == 1
    assert spec.o[0].out_dim == dim_positions["OUT"]


@pytest.mark.parametrize("N_val", [0.0, 0.5, 1.0, 2.0, 2.5])
def test_efficient_exp_attention_matches_blog_softmax1_formula(N_val):
    """In the canonical [BOS, query] two-token setup with softmax1, the
    head's output equals the closed-form formula

        e^B · e^(N-B) / (2 + e^(N-B))

    (T=1 non-BOS token: the query's self-attention contributes one e^0=1
    to the denominator alongside the BOS row's e^(N-B); softmax1's "+1"
    anchor adds the remaining 1). At ``N ≪ B`` this collapses to
    ``e^N / 2`` which is the blog's approximation up to a constant.
    """
    B = 2.5
    dim_positions = {"N": 0, "OUT": 1, "BOS_K": 2, "CONST": 3}
    spec = efficient_exp_attention(
        head_idx=0, input_dim="N", output_dim="OUT",
        bos_token_key_dim="BOS_K", bias=B, head_dim=8,
        dim_positions=dim_positions,
    )
    attn = PureAttention(dim=8, num_heads=1, causal=False)
    Primitives.generate_attention_head(attn, spec, HD=8)
    x = torch.zeros(1, 2, 8, dtype=torch.float32)
    x[0, 0, dim_positions["BOS_K"]] = 1.0
    x[0, 0, dim_positions["CONST"]] = 1.0
    x[0, 1, dim_positions["N"]] = N_val
    x[0, 1, dim_positions["CONST"]] = 1.0
    with torch.no_grad():
        y = _softmax1_attn_forward(x, attn)
    delta = float((y - x)[0, 1, dim_positions["OUT"]].item())
    diff = N_val - B
    expected = math.exp(B) * math.exp(diff) / (2.0 + math.exp(diff))
    assert math.isclose(delta, expected, rel_tol=1e-4, abs_tol=1e-4), (
        f"N={N_val}, B={B}: delta={delta}, expected={expected}"
    )


def test_efficient_exp_attention_rejects_missing_dim():
    with pytest.raises(ValueError, match="missing from dim_positions"):
        efficient_exp_attention(
            head_idx=0, input_dim="MISSING", output_dim="OUT",
            bos_token_key_dim="BOS_K", bias=1.0, head_dim=8,
            dim_positions={"OUT": 0, "BOS_K": 1, "CONST": 2},
        )


def test_efficient_exp_attention_rejects_slot_overflow():
    with pytest.raises(ValueError, match="input_slot"):
        efficient_exp_attention(
            head_idx=0, input_dim="N", output_dim="OUT",
            bos_token_key_dim="BOS_K", bias=1.0, head_dim=4,
            input_slot=4,
            dim_positions={"N": 0, "OUT": 1, "BOS_K": 2, "CONST": 3},
        )


# ===========================================================================
# memory_load_attention
# ===========================================================================


def test_memory_load_attention_fetches_correct_row():
    """Build a KV state with 3 memory rows at distinct one-hot
    addresses and verify the head loads the matching row's value."""

    dim_positions = {
        "Q_A0": 0, "Q_A1": 1, "Q_A2": 2, "Q_A3": 3,
        "K_A0": 4, "K_A1": 5, "K_A2": 6, "K_A3": 7,
        "VAL": 8, "OUT": 9,
    }
    spec = memory_load_attention(
        head_idx=0,
        addr_query_dims=["Q_A0", "Q_A1", "Q_A2", "Q_A3"],
        addr_key_dims=["K_A0", "K_A1", "K_A2", "K_A3"],
        value_dims=["VAL"],
        output_dims=["OUT"],
        head_dim=8, query_weight=15.0, key_weight=15.0,
        dim_positions=dim_positions,
    )
    attn = PureAttention(dim=16, num_heads=1, causal=False)
    Primitives.generate_attention_head(attn, spec, HD=8)
    # Token i: K_A{i}=1, VAL=i*10. Token 3 (query): Q_A{2}=1 → fetch
    # the row whose addr nibble bank-2 is hot (token 2, VAL=20).
    x = torch.zeros(1, 4, 16, dtype=torch.float32)
    x[0, 0, dim_positions["K_A0"]] = 1.0
    x[0, 0, dim_positions["VAL"]] = 10.0
    x[0, 1, dim_positions["K_A1"]] = 1.0
    x[0, 1, dim_positions["VAL"]] = 20.0
    x[0, 2, dim_positions["K_A2"]] = 1.0
    x[0, 2, dim_positions["VAL"]] = 30.0
    x[0, 3, dim_positions["Q_A2"]] = 1.0
    with torch.no_grad():
        y = attn(x)
    delta = float((y - x)[0, 3, dim_positions["OUT"]].item())
    # With L=15.0 and one-hot match, the attention saturates sharply.
    # Score on the matched row is much higher than the non-matched rows
    # (which still attend a little via the query row's self-attention),
    # so the matched row's value (30.0) lands cleanly on OUT.
    assert abs(delta - 30.0) < 1e-2, f"memory_load delta={delta!r}"


def test_memory_load_attention_rejects_width_mismatch():
    dim_positions = {"A": 0, "B": 1, "V": 2, "O": 3}
    with pytest.raises(ValueError, match="length mismatch"):
        memory_load_attention(
            head_idx=0,
            addr_query_dims=["A"],
            addr_key_dims=["A", "B"],
            value_dims=["V"],
            output_dims=["O"],
            dim_positions=dim_positions,
        )
    with pytest.raises(ValueError, match="length mismatch"):
        memory_load_attention(
            head_idx=0,
            addr_query_dims=["A"],
            addr_key_dims=["B"],
            value_dims=["V"],
            output_dims=["O", "O"],
            dim_positions=dim_positions,
        )


def test_memory_load_attention_rejects_slot_overflow():
    dim_positions = {"A": 0, "B": 1, "V": 2, "O": 3}
    with pytest.raises(ValueError, match="head_dim"):
        memory_load_attention(
            head_idx=0,
            addr_query_dims=["A"] * 4,
            addr_key_dims=["B"] * 4,
            value_dims=["V"],
            output_dims=["O"],
            head_dim=4,
            dim_positions=dim_positions,
        )


# ===========================================================================
# fetch_byte_attention
# ===========================================================================


def _build_fetch_byte_kv(*, n_rows, pc_nibbles=2, nibble_bits=4,
                          value_width=2, dim_positions, addr_per_row,
                          val_per_row):
    """Build a synthetic [n_rows] KV state for fetch_byte_attention.

    PC is one-hot encoded across ``pc_nibbles`` banks of
    ``2 ** nibble_bits`` cells. ``addr_per_row[i]`` is the integer
    address stored at row ``i``. ``val_per_row[i]`` is the byte-tuple
    stored at that row.
    """
    cells = 1 << nibble_bits
    d_model = max(dim_positions.values()) + value_width + 8
    x = torch.zeros(1, n_rows, d_model, dtype=torch.float32)
    for i, addr in enumerate(addr_per_row):
        for b in range(pc_nibbles):
            cell = (addr >> (b * nibble_bits)) & (cells - 1)
            x[0, i, dim_positions["ADDR_KEY"] + b * cells + cell] = 1.0
        for j, v in enumerate(val_per_row[i]):
            x[0, i, dim_positions["VAL"] + j] = float(v)
    return x


def test_fetch_byte_attention_zero_offset_loads_pc_row():
    """With ``pc_offset=0`` and one-hot-encoded PC, the query at the
    cell encoding PC=k loads the row whose stored address equals k."""
    pc_nibbles = 2
    nibble_bits = 4
    cells = 1 << nibble_bits
    addr_band_w = pc_nibbles * cells
    value_width = 2
    dim_positions = {
        "PC": 0,
        "ADDR_KEY": addr_band_w,
        "VAL": 2 * addr_band_w,
        "OUT": 2 * addr_band_w + value_width,
    }
    d_model = 2 * addr_band_w + 2 * value_width + 8
    spec = fetch_byte_attention(
        head_idx=0,
        pc_dim_base="PC",
        addr_key_dim_base="ADDR_KEY",
        value_dim_base="VAL",
        output_dim_base="OUT",
        pc_offset=0,
        pc_nibbles=pc_nibbles,
        nibble_bits=nibble_bits,
        value_width=value_width,
        head_dim=64,
        query_weight=15.0,
        key_weight=15.0,
        dim_positions=dim_positions,
    )
    attn = PureAttention(dim=d_model, num_heads=1, causal=False)
    Primitives.generate_attention_head(attn, spec, HD=64)
    # 5 tokens: 4 memory rows storing addresses 0..3 with values
    # (10+i, 100+i), and one query at PC=2 → fetch row 2.
    x = _build_fetch_byte_kv(
        n_rows=5, pc_nibbles=pc_nibbles, nibble_bits=nibble_bits,
        value_width=value_width, dim_positions=dim_positions,
        addr_per_row=[0, 1, 2, 3, 0],
        val_per_row=[(10, 100), (11, 101), (12, 102), (13, 103), (0, 0)],
    )
    # Query at token 4: encode PC=2 as one-hot
    pc_val = 2
    for b in range(pc_nibbles):
        cell = (pc_val >> (b * nibble_bits)) & (cells - 1)
        x[0, 4, dim_positions["PC"] + b * cells + cell] = 1.0
    with torch.no_grad():
        y = attn(x)
    delta_lo = float((y - x)[0, 4, dim_positions["OUT"]].item())
    delta_hi = float((y - x)[0, 4, dim_positions["OUT"] + 1].item())
    assert abs(delta_lo - 12.0) < 5e-2, (
        f"OUT[0] should fetch row 2 VAL[0]=12, got {delta_lo!r}"
    )
    assert abs(delta_hi - 102.0) < 5e-2, (
        f"OUT[1] should fetch row 2 VAL[1]=102, got {delta_hi!r}"
    )


def test_fetch_byte_attention_offset_one_shifts_query_target():
    """With ``pc_offset=1``, query at PC=2 should fetch the row whose
    stored address equals 3 (one ahead)."""
    pc_nibbles = 2
    nibble_bits = 4
    cells = 1 << nibble_bits
    addr_band_w = pc_nibbles * cells
    value_width = 2
    dim_positions = {
        "PC": 0,
        "ADDR_KEY": addr_band_w,
        "VAL": 2 * addr_band_w,
        "OUT": 2 * addr_band_w + value_width,
    }
    d_model = 2 * addr_band_w + 2 * value_width + 8
    spec = fetch_byte_attention(
        head_idx=0,
        pc_dim_base="PC",
        addr_key_dim_base="ADDR_KEY",
        value_dim_base="VAL",
        output_dim_base="OUT",
        pc_offset=1,
        pc_nibbles=pc_nibbles,
        nibble_bits=nibble_bits,
        value_width=value_width,
        head_dim=64,
        query_weight=15.0,
        key_weight=15.0,
        dim_positions=dim_positions,
    )
    attn = PureAttention(dim=d_model, num_heads=1, causal=False)
    Primitives.generate_attention_head(attn, spec, HD=64)
    x = _build_fetch_byte_kv(
        n_rows=5, pc_nibbles=pc_nibbles, nibble_bits=nibble_bits,
        value_width=value_width, dim_positions=dim_positions,
        addr_per_row=[0, 1, 2, 3, 0],
        val_per_row=[(10, 100), (11, 101), (12, 102), (13, 103), (0, 0)],
    )
    pc_val = 2
    for b in range(pc_nibbles):
        cell = (pc_val >> (b * nibble_bits)) & (cells - 1)
        x[0, 4, dim_positions["PC"] + b * cells + cell] = 1.0
    with torch.no_grad():
        y = attn(x)
    delta_lo = float((y - x)[0, 4, dim_positions["OUT"]].item())
    assert abs(delta_lo - 13.0) < 5e-2, (
        f"OUT[0] should fetch row 3 VAL[0]=13 (PC+1 of PC=2), got "
        f"{delta_lo!r}"
    )


def test_fetch_byte_attention_rejects_out_of_range_pc_offset():
    dim_positions = {"PC": 0, "ADDR_KEY": 32, "VAL": 64, "OUT": 72}
    with pytest.raises(ValueError, match="pc_offset"):
        fetch_byte_attention(
            head_idx=0,
            pc_dim_base="PC", addr_key_dim_base="ADDR_KEY",
            value_dim_base="VAL", output_dim_base="OUT",
            pc_offset=16, nibble_bits=4,
            dim_positions=dim_positions,
        )


# ===========================================================================
# opcode_expert_rules
# ===========================================================================


def test_opcode_expert_rules_threads_gate_into_ungated_rule():
    """An ungated rule (constant_write) gets ``gate=OP_X`` and gate_bias
    flips 1→0 to match gated_write convention."""

    rule = one_hot_indicator_rule(band="BAND", value=3, write_dim="OUT+0")
    assert rule.gate is None
    assert rule.gate_terms == ()
    assert rule.gate_bias == 1.0
    wrapped = opcode_expert_rules("OP_ADD", (rule,))
    assert len(wrapped) == 1
    w = wrapped[0]
    assert w.gate is not None
    assert w.gate.name == "OP_ADD"
    assert w.gate_weight == 1.0
    assert w.gate_bias == 0.0
    # Conditions and writes are unchanged.
    assert w.conditions == rule.conditions
    assert w.writes == rule.writes


def test_opcode_expert_rules_folds_into_existing_gate_terms():
    """A rule with existing gate_terms gets the opcode appended as
    another additive term — preserves the existing gate structure."""

    rule = multi_way_and_rule(
        conditions=(("A", 1.0), ("B", 1.0)),
        writes=(("OUT+0", 0.02),),
        gate_terms=(("MARK_AX", 1.0),),
    )
    assert rule.gate is None
    assert len(rule.gate_terms) == 1
    wrapped = opcode_expert_rules("OP_SUB", (rule,))
    assert len(wrapped) == 1
    w = wrapped[0]
    assert w.gate is None
    assert len(w.gate_terms) == 2
    assert w.gate_terms[0].dim.name == "MARK_AX"
    assert w.gate_terms[1].dim.name == "OP_SUB"
    assert w.gate_terms[1].weight == 1.0
    # gate_bias is preserved.
    assert w.gate_bias == rule.gate_bias


def test_opcode_expert_rules_preserves_existing_gate_dim():
    """A rule that already has ``gate=`` set should get the opcode
    folded into ``gate_terms`` (so both the original gate and the
    opcode flag must be present for the rule to fire)."""

    rule = step_function_rule(
        input_dim="X", threshold=0.5, write_dim="OUT+0",
        gate="MARK_AX",
    )
    assert rule.gate is not None
    assert rule.gate.name == "MARK_AX"
    wrapped = opcode_expert_rules("OP_MUL", (rule,))
    w = wrapped[0]
    # Original gate dim is preserved.
    assert w.gate is not None
    assert w.gate.name == "MARK_AX"
    # Opcode appended as an additive gate term.
    assert len(w.gate_terms) == 1
    assert w.gate_terms[0].dim.name == "OP_MUL"


def test_opcode_expert_rules_threads_through_batch():
    """All rules in a batch should be wrapped uniformly."""

    band_rules = band_range_check_rules(
        band="BAND", lo=0, hi=3, write_dim="OUT+0", name="rng",
    )
    assert len(band_rules) == 4
    wrapped = opcode_expert_rules("OP_AND", band_rules)
    assert len(wrapped) == 4
    for w in wrapped:
        assert w.gate is not None
        assert w.gate.name == "OP_AND"
        assert w.gate_bias == 0.0


def test_opcode_expert_rules_with_custom_gate_weight():
    rule = one_hot_indicator_rule(band="B", value=3, write_dim="OUT+0")
    wrapped = opcode_expert_rules("OP_X", (rule,), gate_weight=2.5)
    assert wrapped[0].gate_weight == 2.5


def test_opcode_expert_rules_returns_ffn_rule_tuple():
    rule = one_hot_indicator_rule(band="B", value=3, write_dim="OUT+0")
    wrapped = opcode_expert_rules("OP_X", (rule,))
    assert isinstance(wrapped, tuple)
    for w in wrapped:
        assert isinstance(w, FFNRule)
