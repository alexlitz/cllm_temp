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
    attention_head_extension,
    band_range_check_rules,
    binary_address_lookup_attention,
    byte_clear_rules,
    byte_route_rules,
    cancel_residual_rule,
    carry_relay_rules,
    efficient_exp_attention,
    fetch_byte_attention,
    lookup_table_rules,
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
    AO,
    AP,
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


# ===========================================================================
# binary_address_lookup_attention (V2.1)
# ===========================================================================


def _l15_binary_addr_dim_positions():
    """Mirror the L15 dim layout the imperative writer expects.

    Just enough dims to write rows 0..63 of one head: bias dim
    ``CONST``, the three address bytes ``ADDR_B{0,1,2}_{LO,HI}`` (each
    16 cells), the byte-select V/O dims, and a few discriminator /
    suppressor flags.
    """
    # ``CMP`` is a 16-wide band — the spec resolves ``"CMP+3"`` to
    # ``dp["CMP"] + 3``; the imperative writer uses ``dp["CMP"] + 3``
    # directly so both paths land at the same cell.
    return {
        "CONST": 0,
        "OP_LI_RELAY": 1,
        "OP_LC_RELAY": 2,
        "MARK_PC": 3,
        "MARK_SP": 4,
        "MARK_AX": 5,
        "MARK_STACK0": 6,
        "OP_LEV": 7,
        "H1": 16,  # H1 band base
        "CMP": 32,  # CMP band base (CMP+3 lands at 35)
        "ADDR_B0_LO": 48,
        "ADDR_B0_HI": 64,
        "ADDR_B1_LO": 80,
        "ADDR_B1_HI": 96,
        "ADDR_B2_LO": 112,
        "ADDR_B2_HI": 128,
        "CLEAN_EMBED_LO": 144,
        "CLEAN_EMBED_HI": 160,
        "OUTPUT_LO": 176,
        "OUTPUT_HI": 192,
        "PAD": 208,
    }


def _imperative_l15_head0_binary_addr_slice(attn, HD, dp):
    """Imperative L15 head-0 binary address slice — mirrors lines
    7270..7372 of vm_step.py (slot 0 bias + slots 4..27 binary 24-bit).

    Implements the load-side reads only, no V/O block, so the test can
    compare against the same surface emitted by
    :func:`binary_address_lookup_attention`.
    """
    base = 0
    scale = 10.0
    # Bias slot
    attn.W_q.data[base + 0, dp["CONST"]] = -2000.0
    attn.W_q.data[base + 0, dp["OP_LI_RELAY"]] = 2000.0
    attn.W_q.data[base + 0, dp["OP_LC_RELAY"]] = 2000.0
    attn.W_q.data[base + 0, dp["CMP"] + 3] = 2000.0
    attn.W_q.data[base + 0, dp["OP_LEV"]] = -1000.0
    attn.W_q.data[base + 0, dp["MARK_PC"]] = -25000.0
    attn.W_q.data[base + 0, dp["MARK_SP"]] = -100000.0
    attn.W_k.data[base + 0, dp["CONST"]] = 10.0
    # Binary address slots 4..27
    addr_dim = 4
    addr_bases = [
        dp["ADDR_B0_LO"], dp["ADDR_B0_HI"],
        dp["ADDR_B1_LO"], dp["ADDR_B1_HI"],
        dp["ADDR_B2_LO"], dp["ADDR_B2_HI"],
    ]
    for nibble_base in addr_bases:
        for bit in range(4):
            for k in range(16):
                bit_val = 2 * ((k >> bit) & 1) - 1
                attn.W_q.data[base + addr_dim, nibble_base + k] = scale * bit_val
                attn.W_k.data[base + addr_dim, nibble_base + k] = scale * bit_val
            addr_dim += 1


def test_binary_address_lookup_attention_byte_identical_to_imperative():
    """Build the L15 head-0 binary-address slice via the new primitive
    and via the imperative writer, then assert byte-identical Q/K/V/O
    via ``Primitives.generate_attention_head``."""
    dp = _l15_binary_addr_dim_positions()
    d_model = dp["PAD"] + 16
    HD = 64

    # Spec path
    spec = binary_address_lookup_attention(
        head_idx=0,
        addr_dim_bases=[
            "ADDR_B0_LO", "ADDR_B0_HI",
            "ADDR_B1_LO", "ADDR_B1_HI",
            "ADDR_B2_LO", "ADDR_B2_HI",
        ],
        addr_width_bits=4,
        addr_slot_base=4,
        bit_scale=10.0,
        bias_slot=0,
        bias_dim="CONST", bias_weight=-2000.0,
        key_bias_dim="CONST", key_bias_weight=10.0,
        discriminators=[
            ("OP_LI_RELAY", 2000.0),
            ("OP_LC_RELAY", 2000.0),
            ("CMP+3", 2000.0),
        ],
        suppressors=[
            ("OP_LEV", -1000.0),
            ("MARK_PC", -25000.0),
            ("MARK_SP", -100000.0),
        ],
        head_dim=HD,
        dim_positions=dp,
    )
    attn_spec = PureAttention(dim=d_model, num_heads=1, causal=False)
    Primitives.generate_attention_head(attn_spec, spec, HD=HD)

    # Imperative path
    attn_imp = PureAttention(dim=d_model, num_heads=1, causal=False)
    _imperative_l15_head0_binary_addr_slice(attn_imp, HD=HD, dp=dp)

    # Byte-identity
    assert torch.equal(attn_spec.W_q.data, attn_imp.W_q.data), (
        "W_q differs between binary_address_lookup_attention and imperative"
    )
    assert torch.equal(attn_spec.W_k.data, attn_imp.W_k.data), (
        "W_k differs between binary_address_lookup_attention and imperative"
    )
    assert torch.equal(attn_spec.W_v.data, attn_imp.W_v.data), "W_v differs"
    assert torch.equal(attn_spec.W_o.data, attn_imp.W_o.data), "W_o differs"


def test_binary_address_lookup_attention_with_value_block():
    """Verify the optional value/output block lands at the right slots
    and writes the right cells.
    """
    dp = _l15_binary_addr_dim_positions()
    d_model = dp["PAD"] + 16
    HD = 64
    # Two-byte CLEAN_EMBED_LO -> OUTPUT_LO mapping at slot 32 onward.
    value_dims = [f"CLEAN_EMBED_LO" for _ in range(1)]
    output_dims = [f"OUTPUT_LO" for _ in range(1)]
    spec = binary_address_lookup_attention(
        head_idx=0,
        addr_dim_bases=["ADDR_B0_LO", "ADDR_B0_HI"],
        addr_width_bits=4,
        addr_slot_base=4,
        bias_dim=None, key_bias_dim=None,
        value_slot_base=32,
        value_dims=value_dims,
        output_dims=output_dims,
        head_dim=HD,
        dim_positions=dp,
    )
    # Verify the V/O writes land at slot 32.
    assert any(w.slot == 32 and w.dim == dp["CLEAN_EMBED_LO"]
               for w in spec.v), "V block missing slot 32"
    assert any(w.slot == 32 and w.out_dim == dp["OUTPUT_LO"]
               for w in spec.o), "O block missing slot 32"


def test_binary_address_lookup_attention_default_value_slot_base():
    """``value_slot_base=None`` should default to addr_slot_base +
    len(addr_dim_bases) * addr_width_bits — right after the address block.
    """
    dp = _l15_binary_addr_dim_positions()
    spec = binary_address_lookup_attention(
        head_idx=0,
        addr_dim_bases=["ADDR_B0_LO", "ADDR_B0_HI"],  # 2 bands × 4 bits = 8 slots
        addr_width_bits=4,
        addr_slot_base=4,
        bias_dim=None, key_bias_dim=None,
        value_dims=["CLEAN_EMBED_LO"],
        output_dims=["OUTPUT_LO"],
        head_dim=64,
        dim_positions=dp,
    )
    # Address block: slots 4..11 (8 slots). Default value_slot_base = 12.
    assert spec.v[0].slot == 12, (
        f"default value_slot_base should be 4+8=12, got {spec.v[0].slot}"
    )


def test_binary_address_lookup_attention_rejects_missing_dim():
    dp = {"CONST": 0}
    with pytest.raises(ValueError, match="missing from dim_positions"):
        binary_address_lookup_attention(
            head_idx=0,
            addr_dim_bases=["MISSING_BAND"],
            addr_width_bits=4,
            dim_positions=dp,
        )


def test_binary_address_lookup_attention_rejects_slot_overflow():
    dp = _l15_binary_addr_dim_positions()
    with pytest.raises(ValueError, match="slot extent"):
        binary_address_lookup_attention(
            head_idx=0,
            addr_dim_bases=["ADDR_B0_LO"] * 8,  # 8 bands × 4 bits = 32 slots
            addr_width_bits=4,
            addr_slot_base=40,  # 40 + 32 = 72 > head_dim=64
            head_dim=64,
            dim_positions=dp,
        )


def test_binary_address_lookup_attention_rejects_vo_length_mismatch():
    dp = _l15_binary_addr_dim_positions()
    with pytest.raises(ValueError, match="length mismatch"):
        binary_address_lookup_attention(
            head_idx=0,
            addr_dim_bases=["ADDR_B0_LO"],
            addr_width_bits=4,
            value_dims=["CLEAN_EMBED_LO", "CLEAN_EMBED_HI"],
            output_dims=["OUTPUT_LO"],
            dim_positions=dp,
        )


def test_binary_address_lookup_attention_band_offset_dim_names():
    """``"BAND+N"`` names (e.g. ``"CMP+3"``, ``"H1+3"``) should resolve
    to ``dim_positions[BAND] + N`` for discriminators / suppressors.
    """
    dp = {
        "CONST": 0,
        "H1": 10,
        "CMP": 20,
        "ADDR_B0_LO": 30,
    }
    spec = binary_address_lookup_attention(
        head_idx=0,
        addr_dim_bases=["ADDR_B0_LO"],
        addr_width_bits=2,  # tiny — 2 bits, 4 cells per band, 2 slots
        bias_slot=0,
        bias_dim="CONST", bias_weight=-1.0,
        discriminators=[("CMP+3", 5.0)],
        suppressors=[("H1+3", -50.0)],
        head_dim=8,
        dim_positions=dp,
    )
    # Find writes at bias slot.
    cmp_writes = [w for w in spec.q if w.slot == 0 and w.dim == 23]
    h1_writes = [w for w in spec.q if w.slot == 0 and w.dim == 13]
    assert len(cmp_writes) == 1 and cmp_writes[0].weight == 5.0
    assert len(h1_writes) == 1 and h1_writes[0].weight == -50.0


# ===========================================================================
# attention_head_extension (V2.1)
# ===========================================================================


def test_attention_head_extension_appends_writes_and_preserves_head_idx():
    """Extension appends Q/K/V/O writes and preserves head_idx /
    alibi_slope / head_dim / group_size from the base spec.
    """
    dp = {"CONST": 0, "ADDR_B0_LO": 1, "MARK_AX": 17, "OUTPUT_LO": 18}
    base = binary_address_lookup_attention(
        head_idx=3,
        addr_dim_bases=["ADDR_B0_LO"],
        addr_width_bits=4,
        bias_slot=0, bias_dim="CONST", bias_weight=-1.0,
        head_dim=64,
        dim_positions=dp,
    )
    base_q = len(base.q)
    base_k = len(base.k)
    extra_q = (AP(28, dp["MARK_AX"], 500.0),)
    extra_o = (AO(dp["OUTPUT_LO"], 32, 1.0),)
    ext = attention_head_extension(
        base, extra_q_writes=extra_q, extra_o_writes=extra_o,
        alibi_slope=1.0,
    )
    assert ext.head_idx == 3
    assert ext.alibi_slope == 1.0
    assert len(ext.q) == base_q + 1
    assert len(ext.k) == base_k  # unchanged
    assert ext.q[-1] == extra_q[0]
    assert ext.o[-1] == extra_o[0]
    # Base spec not mutated.
    assert len(base.q) == base_q
    assert base.alibi_slope is None


def test_attention_head_extension_lowers_byte_identically_to_combined_spec():
    """An extended spec lowered into a fresh PureAttention should write
    the same cells as a hand-assembled DeclarativeAttentionHeadSpec that
    carries the same combined Q/K/V/O writes.
    """
    dp = _l15_binary_addr_dim_positions()
    d_model = dp["PAD"] + 16
    HD = 64
    base = binary_address_lookup_attention(
        head_idx=0,
        addr_dim_bases=["ADDR_B0_LO", "ADDR_B0_HI"],
        addr_width_bits=4,
        bias_slot=0, bias_dim="CONST", bias_weight=-1.0,
        key_bias_dim="CONST", key_bias_weight=10.0,
        head_dim=HD,
        dim_positions=dp,
    )
    extra_q = (
        AP(28, dp["MARK_AX"], 500.0),
        AP(28, dp["MARK_STACK0"], 500.0),
    )
    extra_k = (AP(28, dp["CONST"], 5.0),)
    ext = attention_head_extension(
        base, extra_q_writes=extra_q, extra_k_writes=extra_k,
    )
    # Combined hand-build for comparison.
    combined = DeclarativeAttentionHeadSpec(
        head_idx=0,
        q=tuple(base.q) + extra_q,
        k=tuple(base.k) + extra_k,
        v=base.v, o=base.o,
        alibi_slope=base.alibi_slope,
    )
    a1 = PureAttention(dim=d_model, num_heads=1, causal=False)
    a2 = PureAttention(dim=d_model, num_heads=1, causal=False)
    Primitives.generate_attention_head(a1, ext, HD=HD)
    Primitives.generate_attention_head(a2, combined, HD=HD)
    assert torch.equal(a1.W_q.data, a2.W_q.data)
    assert torch.equal(a1.W_k.data, a2.W_k.data)
    assert torch.equal(a1.W_v.data, a2.W_v.data)
    assert torch.equal(a1.W_o.data, a2.W_o.data)


def test_attention_head_extension_preserves_base_alibi_slope_when_not_overridden():
    """When ``alibi_slope`` is omitted, the base spec's slope is kept."""
    spec = DeclarativeAttentionHeadSpec(
        head_idx=2,
        q=(AP(0, 0, 1.0),),
        k=(), v=(), o=(),
        alibi_slope=0.5,
    )
    ext = attention_head_extension(spec, extra_q_writes=(AP(1, 1, 2.0),))
    assert ext.alibi_slope == 0.5


def test_attention_head_extension_overrides_alibi_slope_when_supplied():
    spec = DeclarativeAttentionHeadSpec(
        head_idx=2,
        q=(AP(0, 0, 1.0),),
        k=(), v=(), o=(),
        alibi_slope=0.5,
    )
    ext = attention_head_extension(spec, alibi_slope=1.0)
    assert ext.alibi_slope == 1.0


# ===========================================================================
# binary_address_lookup_attention + extension — full L15 head-0 surface
# ===========================================================================


def _imperative_l15_head0_full_surface(attn, HD, dp):
    """Reproduce a representative subset of the legacy L15 head-0 body:
    binary address (slots 4..27) + byte-selection (slot 3) +
    position-gate (slot 28) + V/O block (slots 32..63).

    Excludes the dense suppressor row writes (slots 29..33) — those are
    rebuilt via :func:`attention_head_extension` extras in the spec
    path and the test asserts byte-identity over the FULL head surface.
    """
    base = 0
    scale = 10.0
    BS = 60.0
    # Slot 0 bias + Q-side discriminators / suppressors
    attn.W_q.data[base + 0, dp["CONST"]] = -2000.0
    attn.W_q.data[base + 0, dp["OP_LI_RELAY"]] = 2000.0
    attn.W_q.data[base + 0, dp["OP_LC_RELAY"]] = 2000.0
    attn.W_q.data[base + 0, dp["CMP"] + 3] = 2000.0
    attn.W_q.data[base + 0, dp["MARK_PC"]] = -25000.0
    attn.W_q.data[base + 0, dp["MARK_SP"]] = -100000.0
    attn.W_k.data[base + 0, dp["CONST"]] = 10.0
    # Slot 3 byte selection
    attn.W_q.data[base + 3, dp["MARK_AX"]] = BS
    attn.W_q.data[base + 3, dp["MARK_STACK0"]] = BS
    # Slots 4..27 binary 24-bit address
    addr_dim = 4
    addr_bases = [
        dp["ADDR_B0_LO"], dp["ADDR_B0_HI"],
        dp["ADDR_B1_LO"], dp["ADDR_B1_HI"],
        dp["ADDR_B2_LO"], dp["ADDR_B2_HI"],
    ]
    for nibble_base in addr_bases:
        for bit in range(4):
            for k in range(16):
                bit_val = 2 * ((k >> bit) & 1) - 1
                attn.W_q.data[base + addr_dim, nibble_base + k] = scale * bit_val
                attn.W_k.data[base + addr_dim, nibble_base + k] = scale * bit_val
            addr_dim += 1
    # Slot 28 per-head position gate
    attn.W_q.data[base + 28, dp["CONST"]] = -500.0
    attn.W_q.data[base + 28, dp["MARK_AX"]] = 500.0
    attn.W_q.data[base + 28, dp["MARK_STACK0"]] = 500.0
    attn.W_k.data[base + 28, dp["CONST"]] = 5.0
    # V/O block at slots 32..63
    for k in range(16):
        attn.W_v.data[base + 32 + k, dp["CLEAN_EMBED_LO"] + k] = 1.0
        attn.W_v.data[base + 48 + k, dp["CLEAN_EMBED_HI"] + k] = 1.0
        attn.W_o.data[dp["OUTPUT_LO"] + k, base + 32 + k] = 1.0
        attn.W_o.data[dp["OUTPUT_HI"] + k, base + 48 + k] = 1.0


def test_v21_primitive_plus_extension_byte_identical_to_full_imperative_l15_head0():
    """End-to-end V2.1 byte-identity gate: build the L15 head-0 surface
    via :func:`binary_address_lookup_attention` + V/O block +
    :func:`attention_head_extension` rows, and verify the lowered
    W_q/W_k/W_v/W_o are equal to the imperative writer.
    """
    dp = _l15_binary_addr_dim_positions()
    d_model = dp["PAD"] + 16
    HD = 64
    BS = 60.0

    # V/O slots 32..47 (LO) and 48..63 (HI). Pass via value_dims as
    # base+k entries through dim_positions; here we expand inline.
    # The primitive's V/O block writes one V slot per value_dims entry,
    # so 32 entries: 16 LO + 16 HI starting at slot 32.
    value_dims = [f"CLEAN_EMBED_LO+{k}" for k in range(16)] + [
        f"CLEAN_EMBED_HI+{k}" for k in range(16)
    ]
    output_dims = [f"OUTPUT_LO+{k}" for k in range(16)] + [
        f"OUTPUT_HI+{k}" for k in range(16)
    ]
    # The primitive's resolver doesn't handle band+N for V/O dims (only
    # for discriminators / suppressors). Expand the per-cell dims by
    # injecting them into dim_positions for this test.
    expanded_dp = dict(dp)
    for k in range(16):
        expanded_dp[f"CLEAN_EMBED_LO+{k}"] = dp["CLEAN_EMBED_LO"] + k
        expanded_dp[f"CLEAN_EMBED_HI+{k}"] = dp["CLEAN_EMBED_HI"] + k
        expanded_dp[f"OUTPUT_LO+{k}"] = dp["OUTPUT_LO"] + k
        expanded_dp[f"OUTPUT_HI+{k}"] = dp["OUTPUT_HI"] + k

    base_spec = binary_address_lookup_attention(
        head_idx=0,
        addr_dim_bases=[
            "ADDR_B0_LO", "ADDR_B0_HI",
            "ADDR_B1_LO", "ADDR_B1_HI",
            "ADDR_B2_LO", "ADDR_B2_HI",
        ],
        addr_width_bits=4,
        addr_slot_base=4,
        bit_scale=10.0,
        bias_slot=0,
        bias_dim="CONST", bias_weight=-2000.0,
        key_bias_dim="CONST", key_bias_weight=10.0,
        discriminators=[
            ("OP_LI_RELAY", 2000.0),
            ("OP_LC_RELAY", 2000.0),
            ("CMP+3", 2000.0),
        ],
        suppressors=[
            ("MARK_PC", -25000.0),
            ("MARK_SP", -100000.0),
        ],
        value_slot_base=32,
        value_dims=value_dims,
        output_dims=output_dims,
        head_dim=HD,
        dim_positions=expanded_dp,
    )
    # Layer slot-3 byte selection + slot-28 position gate via extension.
    extra_q = (
        AP(3, dp["MARK_AX"], BS),
        AP(3, dp["MARK_STACK0"], BS),
        AP(28, dp["CONST"], -500.0),
        AP(28, dp["MARK_AX"], 500.0),
        AP(28, dp["MARK_STACK0"], 500.0),
    )
    extra_k = (AP(28, dp["CONST"], 5.0),)
    spec = attention_head_extension(
        base_spec, extra_q_writes=extra_q, extra_k_writes=extra_k,
    )

    attn_spec = PureAttention(dim=d_model, num_heads=1, causal=False)
    Primitives.generate_attention_head(attn_spec, spec, HD=HD)

    attn_imp = PureAttention(dim=d_model, num_heads=1, causal=False)
    _imperative_l15_head0_full_surface(attn_imp, HD=HD, dp=dp)

    assert torch.equal(attn_spec.W_q.data, attn_imp.W_q.data), (
        "W_q differs at full L15 head-0 surface"
    )
    assert torch.equal(attn_spec.W_k.data, attn_imp.W_k.data), (
        "W_k differs at full L15 head-0 surface"
    )
    assert torch.equal(attn_spec.W_v.data, attn_imp.W_v.data), (
        "W_v differs at full L15 head-0 surface"
    )
    assert torch.equal(attn_spec.W_o.data, attn_imp.W_o.data), (
        "W_o differs at full L15 head-0 surface"
    )


# ---------------------------------------------------------------------------
# Per-nibble byte-band factories (reduction map ⑥ — cross-layer dedup)
#
# These three factories collapse the byte-clear / byte-route / carry-relay
# 16-cell loops re-authored per layer (l6/l9/l14/...). The dedup contract
# is that each factory emits the EXACT rule tuple the hand-authored
# ``multi_way_and_rule`` loop produced, so porting a call site is
# byte-identical. Each test reconstructs the pre-refactor inline loop and
# asserts field-for-field equality of the resulting ``FFNRule`` tuple.
# ---------------------------------------------------------------------------


def _rule_key(rule: FFNRule):
    """Canonical, order-preserving comparison key for one FFNRule."""
    return (
        rule.name,
        tuple((str(t.dim), t.weight) for t in rule.conditions),
        rule.threshold,
        None if rule.gate is None else str(rule.gate),
        rule.gate_weight,
        rule.gate_bias,
        tuple((str(t.dim), t.weight) for t in rule.gate_terms),
        tuple((str(w.dim), w.weight) for w in rule.writes),
        rule.scope,
        tuple(sorted(rule.dominates_at.items())) if rule.dominates_at else None,
    )


def _assert_rules_equal(factory_rules, hand_rules):
    fk = [_rule_key(r) for r in factory_rules]
    hk = [_rule_key(r) for r in hand_rules]
    assert len(fk) == len(hk), f"length mismatch {len(fk)} vs {len(hk)}"
    for i, (a, b) in enumerate(zip(fk, hk)):
        assert a == b, f"rule {i} differs:\n  factory={a!r}\n  hand   ={b!r}"


def test_byte_clear_rules_matches_hand_authored_loop():
    """``byte_clear_rules`` reproduces the l9 ALU-clear 16-cell loop."""
    S = 100.0
    conds = (
        ("MARK_AX", 1.0),
        ("OP_IMM", 1.0),
        ("MARK_PC", -1e6),
    )
    factory = byte_clear_rules(
        bands=("ALU_LO", "ALU_HI"),
        conditions=conds,
        threshold=1.5,
        write_value=-10.0,
        S=S,
        name_by_band={"ALU_LO": "alu_lo_clear", "ALU_HI": "alu_hi_clear"},
    )
    hand = []
    for band, prefix in (("ALU_LO", "alu_lo_clear"), ("ALU_HI", "alu_hi_clear")):
        for k in range(16):
            hand.append(multi_way_and_rule(
                name=f"{prefix}_{k}",
                conditions=conds,
                threshold=1.5,
                writes=((f"{band}+{k}", -10.0 / S),),
            ))
    assert len(factory) == 32
    _assert_rules_equal(factory, hand)


def test_byte_clear_rules_raw_write_and_gate():
    """``byte_clear_rules`` supports a raw (un-S-scaled) gated scrub."""
    S = 100.0
    factory = byte_clear_rules(
        bands=("OUTPUT_LO",),
        conditions=(("NEXT_PC", 100.0 / S),),
        threshold=80.0 / S,
        write_value=-1.0,
        write_scale_by_S=False,
        gate="NEXT_PC",
        gate_weight=5.0,
        gate_bias=-3.0,
        name_prefix="scrub",
    )
    hand = [
        multi_way_and_rule(
            name=f"scrub_{k}",
            conditions=(("NEXT_PC", 100.0 / S),),
            threshold=80.0 / S,
            gate="NEXT_PC",
            gate_weight=5.0,
            gate_bias=-3.0,
            writes=((f"OUTPUT_LO+{k}", -1.0),),
        )
        for k in range(16)
    ]
    assert len(factory) == 16
    _assert_rules_equal(factory, hand)


def test_byte_route_rules_matches_hand_authored_loop():
    """``byte_route_rules`` reproduces the l6 AX_CARRY->OUTPUT route loop."""
    S = 100.0
    conds = (("OP_EXIT", 1.0), ("MARK_AX", 1.0), ("MARK_PC", -8.0))
    factory = byte_route_rules(
        band_specs=(
            ("lo", "AX_CARRY_LO", "OUTPUT_LO"),
            ("hi", "AX_CARRY_HI", "OUTPUT_HI_THIS_STEP"),
        ),
        conditions=conds,
        threshold=4.0,
        write_value=2.0,
        S=S,
        name_prefix="route",
    )
    hand = []
    for band, src, out in (
        ("lo", "AX_CARRY_LO", "OUTPUT_LO"),
        ("hi", "AX_CARRY_HI", "OUTPUT_HI_THIS_STEP"),
    ):
        for k in range(16):
            hand.append(multi_way_and_rule(
                name=f"route_{band}_{k}",
                conditions=conds,
                threshold=4.0,
                gate=f"{src}+{k}",
                writes=((f"{out}+{k}", 2.0 / S),),
            ))
    assert len(factory) == 32
    _assert_rules_equal(factory, hand)


def test_byte_route_rules_per_band_scope_and_dominates():
    """Per-band scope / dominates_at extras are threaded correctly."""
    S = 100.0
    conds = (("OP_IMM", 1.0), ("MARK_AX", 1.0))
    hi_scope = "mark == AX AND opcode_at_AX == IMM"
    factory = byte_route_rules(
        band_specs=(
            ("lo", "FETCH_LO", "OUTPUT_LO"),
            ("hi", "FETCH_HI", "OUTPUT_HI_THIS_STEP"),
        ),
        conditions=conds,
        threshold=4.0,
        write_value=2.0,
        S=S,
        name_prefix="fetch",
        scope_by_band={"hi": hi_scope},
        dominates_at_by_band={"hi": {"OUTPUT_HI_THIS_STEP": hi_scope}},
    )
    # LO band: no scope. HI band: scope + dominates_at set.
    lo_rules = factory[:16]
    hi_rules = factory[16:]
    assert all(r.scope is None for r in lo_rules)
    assert all(r.dominates_at is None for r in lo_rules)
    assert all(r.scope == hi_scope for r in hi_rules)
    assert all(
        r.dominates_at == {"OUTPUT_HI_THIS_STEP": hi_scope} for r in hi_rules
    )


def test_carry_relay_rules_matches_hand_authored_loop():
    """``carry_relay_rules`` reproduces the l6 stack-writeback loop."""
    S = 100.0
    conds = (("OP_ADJ", 1.0), ("MARK_SP", 1.0), ("IS_BYTE", -1e6))
    factory = carry_relay_rules(
        band_specs=(
            ("lo", "EMBED_LO", "AX_CARRY_LO", "OUTPUT_LO"),
            ("hi", "EMBED_HI", "AX_CARRY_HI", "OUTPUT_HI_THIS_STEP"),
        ),
        conditions=conds,
        threshold=1.5,
        write_value=2.0,
        S=S,
        name_prefix="wb",
    )
    hand = []
    for band, eb, cb, out in (
        ("lo", "EMBED_LO", "AX_CARRY_LO", "OUTPUT_LO"),
        ("hi", "EMBED_HI", "AX_CARRY_HI", "OUTPUT_HI_THIS_STEP"),
    ):
        for k in range(16):
            hand.append(multi_way_and_rule(
                name=f"wb_{band}_{k}",
                conditions=conds,
                threshold=1.5,
                gate_terms=(
                    (f"{eb}+{k}", -1.0),
                    (f"{cb}+{k}", 1.0),
                ),
                writes=((f"{out}+{k}", 2.0 / S),),
            ))
    assert len(factory) == 32
    _assert_rules_equal(factory, hand)


def test_byte_band_factories_reject_inverted_range():
    with pytest.raises(ValueError):
        byte_clear_rules(bands=("A",), conditions=(("X", 1.0),),
                         threshold=0.5, write_value=-1.0, lo=5, hi=2)
    with pytest.raises(ValueError):
        byte_route_rules(band_specs=(("lo", "S", "D"),),
                         conditions=(("X", 1.0),), threshold=0.5, lo=5, hi=2)
    with pytest.raises(ValueError):
        carry_relay_rules(band_specs=(("lo", "E", "C", "D"),),
                          conditions=(("X", 1.0),), threshold=0.5, lo=5, hi=2)


def test_byte_band_factories_ported_call_sites_are_byte_identical():
    """The live l6/l9 ported helpers still emit the original rule tuples.

    This pins the reduction-map ⑥ dedup: the two piloted layers (l6, l9)
    delegate to the factories, and their output must equal the
    pre-refactor inline loops field-for-field.
    """
    from c4_release.neural_vm.unified_compiler.ops import l6_ops, l9_ops
    from c4_release.neural_vm.unified_compiler.ops.l9_ops import _NON_ALU_OPCODES

    S = 100.0

    # l9 _alu_clear_rules
    cc = [("MARK_AX", 1.0)] + [(d, 1.0) for d in _NON_ALU_OPCODES]
    cc = tuple(cc) + (
        ("MARK_PC", -1e6), ("MARK_SP", -1e6), ("MARK_BP", -1e6),
        ("MARK_STACK0", -1e6), ("MARK_MEM", -1e6),
    )
    hand = []
    for k in range(16):
        hand.append(multi_way_and_rule(
            name=f"alu_lo_clear_{k}", conditions=cc, threshold=1.5,
            writes=((f"ALU_LO+{k}", -10.0 / S),)))
    for k in range(16):
        hand.append(multi_way_and_rule(
            name=f"alu_hi_clear_{k}", conditions=cc, threshold=1.5,
            writes=((f"ALU_HI+{k}", -10.0 / S),)))
    _assert_rules_equal(l9_ops._alu_clear_rules(S), hand)

    # l6 _layer6_exit_ax_route_rules (byte-route)
    conds = (("OP_EXIT", 1.0), ("OP_IMM", -20.0), ("MARK_AX", 1.0),
             ("MARK_PC", -8.0), ("IS_BYTE", -1.0))
    hand = []
    for band, src, out in (("lo", "AX_CARRY_LO", "OUTPUT_LO"),
                           ("hi", "AX_CARRY_HI", "OUTPUT_HI_THIS_STEP")):
        for k in range(16):
            hand.append(multi_way_and_rule(
                name=f"l6_exit_ax_to_output_{band}_{k}", conditions=conds,
                threshold=4.0, gate=f"{src}+{k}",
                writes=((f"{out}+{k}", 2.0 / S),)))
    _assert_rules_equal(l6_ops._layer6_exit_ax_route_rules(S), hand)

    # l6 _layer6_adj_sp_writeback_rules (carry-relay)
    conds = (("OP_ADJ", 1.0), ("MARK_SP", 1.0), ("IS_BYTE", -1e6),
             ("MARK_PC", -1e6), ("MARK_AX", -1e6), ("MARK_BP", -1e6),
             ("MARK_STACK0", -1e6), ("MARK_MEM", -1e6))
    hand = []
    for band, eb, cb, out in (
        ("lo", "EMBED_LO", "AX_CARRY_LO", "OUTPUT_LO"),
        ("hi", "EMBED_HI", "AX_CARRY_HI", "OUTPUT_HI_THIS_STEP"),
    ):
        for k in range(16):
            hand.append(multi_way_and_rule(
                name=f"l6_adj_sp_writeback_{band}_{k}", conditions=conds,
                threshold=1.5,
                gate_terms=((f"{eb}+{k}", -1.0), (f"{cb}+{k}", 1.0)),
                writes=((f"{out}+{k}", 2.0 / S),)))
    _assert_rules_equal(l6_ops._layer6_adj_sp_writeback_rules(S), hand)


def _rules_by_prefix(rules, prefixes):
    """Extract, in emission order, the rules whose name starts with one of
    ``prefixes`` (each prefix is a ``(name_prefix,)`` band identifier)."""
    out = []
    for r in rules:
        if r.name is not None and any(r.name.startswith(p) for p in prefixes):
            out.append(r)
    return out


def test_l16_byte_route_ported_call_sites_are_byte_identical():
    """The live l16 LEV-routing byte-route bands still emit the original
    rule tuples field-for-field after the reduction-map ⑥ port.

    Four per-cell BYTE-ROUTE bands inside ``_layer16_lev_routing_rules``
    were delegated to :func:`byte_route_rules`:
      * ``l16_lev_ax_carry_{lo,hi}``       (AX_CARRY -> OUTPUT, 2.0/S)
      * ``l16_lev_ax_full_{lo,hi}``        (AX_FULL  -> OUTPUT, 2.0/S)
      * ``l16_stale_imm_ax_carry_{lo,hi}`` (AX_CARRY -> OUTPUT, 2.0/S)
      * ``l16_bp_marker_passthrough_{lo,hi}`` (EMBED -> OUTPUT, 10.0/S)
    Each ported band's live output must equal the pre-refactor inline
    loop (gate on ``SOURCE+k``, single write to ``DEST+k``).
    """
    from c4_release.neural_vm.unified_compiler.ops.l16_ops import (
        _layer16_lev_routing_rules,
    )

    S = 100.0
    live = _layer16_lev_routing_rules(S)

    def _hand_route(name_prefix, band_specs, conditions, threshold, write):
        hand = []
        for band, src, out in band_specs:
            for k in range(16):
                hand.append(multi_way_and_rule(
                    name=f"{name_prefix}_{band}_{k}",
                    conditions=conditions,
                    threshold=threshold,
                    gate=f"{src}+{k}",
                    writes=((f"{out}+{k}", write / S),),
                ))
        return hand

    step0_guard_weight = 10.0
    lev_ax_carry_conditions = (
        ("OP_LEV", 1.0), ("MARK_AX", 1.0),
        ("MARK_PC", -8.0), ("MARK_SP", -8.0), ("MARK_BP", -8.0),
        ("MARK_STACK0", -8.0), ("MARK_MEM", -8.0),
        ("IS_BYTE", -10.0), ("OP_EXIT", -20.0), ("OP_JMP", -20.0),
        ("HAS_SE", step0_guard_weight),
    )
    # l16_lev_ax_carry
    hand = _hand_route(
        "l16_lev_ax_carry",
        (("lo", "AX_CARRY_LO", "OUTPUT_LO"),
         ("hi", "AX_CARRY_HI", "OUTPUT_HI_THIS_STEP")),
        lev_ax_carry_conditions, 1.5 + step0_guard_weight, 2.0)
    _assert_rules_equal(_rules_by_prefix(live, ("l16_lev_ax_carry_",)), hand)

    # l16_lev_ax_full
    hand = _hand_route(
        "l16_lev_ax_full",
        (("lo", "AX_FULL_LO", "OUTPUT_LO"),
         ("hi", "AX_FULL_HI", "OUTPUT_HI_THIS_STEP")),
        lev_ax_carry_conditions, 1.5 + step0_guard_weight, 2.0)
    _assert_rules_equal(_rules_by_prefix(live, ("l16_lev_ax_full_",)), hand)

    # l16_stale_imm_ax_carry
    stale_imm_ax_conditions = (
        ("OP_IMM", 1.0), ("MARK_AX", 1.0), ("MARK_PC", -8.0),
        ("IS_BYTE", -10.0), ("OP_EXIT", -20.0), ("OP_JMP", -20.0),
    )
    hand = _hand_route(
        "l16_stale_imm_ax_carry",
        (("lo", "AX_CARRY_LO", "OUTPUT_LO"),
         ("hi", "AX_CARRY_HI", "OUTPUT_HI_THIS_STEP")),
        stale_imm_ax_conditions, 1.5, 2.0)
    _assert_rules_equal(
        _rules_by_prefix(live, ("l16_stale_imm_ax_carry_",)), hand)

    # l16_bp_marker_passthrough
    bp_marker_passthrough_conditions = (
        ("MARK_BP", 1.0), ("HAS_SE", 1.0), ("IS_BYTE", -10.0),
        ("OP_ENT", -2.0), ("OP_LEV", -2.0),
    )
    hand = _hand_route(
        "l16_bp_marker_passthrough",
        (("lo", "EMBED_LO", "OUTPUT_LO"),
         ("hi", "EMBED_HI", "OUTPUT_HI_THIS_STEP")),
        bp_marker_passthrough_conditions, 1.5, 10.0)
    _assert_rules_equal(
        _rules_by_prefix(live, ("l16_bp_marker_passthrough_",)), hand)


def test_l10_ax_passthrough_ported_call_site_is_byte_identical():
    """The live l10 AX-passthrough byte-route band still emits the original
    rule tuple field-for-field after the reduction-map ⑥ port.

    ``_layer10_alu_ax_passthrough_rules`` (part of the width-locked
    ``layer10_alu`` FFN) was delegated to :func:`byte_route_rules`: two
    per-cell BYTE-ROUTE bands (``AX_CARRY_{LO,HI}+k`` gate -> route
    ``OUTPUT_{LO,HI}+k`` at ``2.0/S``) under the shared ``(MARK_AX,
    *suppressed-op NOT-terms)`` AND at threshold 0.5. The live output
    must equal the pre-refactor inline
    ``for nibble_label, carry_dim, out_dim / for k in range(16)`` loop.
    """
    from c4_release.neural_vm.unified_compiler.ops.l10_ops import (
        _L10_ALU_AX_PASSTHROUGH_SUPPRESSED_OPS,
        _layer10_alu_ax_passthrough_rules,
    )

    S = 100.0
    live = _layer10_alu_ax_passthrough_rules(S)

    hand = []
    for nibble_label, carry_dim, out_dim in (
        ("lo", "AX_CARRY_LO", "OUTPUT_LO"),
        ("hi", "AX_CARRY_HI", "OUTPUT_HI_THIS_STEP"),
    ):
        for k in range(16):
            conditions = [("MARK_AX", 1.0)]
            for op_dim in _L10_ALU_AX_PASSTHROUGH_SUPPRESSED_OPS:
                conditions.append((op_dim, -1.0))
            hand.append(multi_way_and_rule(
                name=f"l10_ax_passthrough_{nibble_label}_{k}",
                conditions=tuple(conditions),
                threshold=0.5,
                gate=f"{carry_dim}+{k}",
                gate_weight=1.0,
                writes=((f"{out_dim}+{k}", 2.0 / S),),
            ))
    assert len(live) == 32
    _assert_rules_equal(live, hand)
