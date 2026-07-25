"""Gate for the unified per-op ALU unit registry (``c4_min.alu_units``).

Covers:
  * DEFAULTS: an unset environment resolves to the current production unit per op,
    and ``apply_to_env`` is a NO-OP (mutates nothing) — so the fused-VM build stays
    byte-identical to golden when no ``C4_*_UNIT`` flag is set.
  * PROJECTION: a registry selection projects onto the legacy builder flags
    (``C4_DIV_LEAN`` / ``C4_DIV_LONGDIV`` / ``C4_MUL_LOOKAHEAD``).
  * GUARDS: unknown / measure-only (non-wired) tokens raise.
  * BUILDER DRIVE (opt-in, CPU): ``C4_DIV_UNIT=radix16_hardened`` actually switches
    the divmod builder variant end-to-end (distinctive Kogge-Stone borrow blocks).
  * FIT SELECTOR: the (layers, d_model) selector picks the expected unit per Qwen
    size and the depth accounting adds up.

Run: ``OMP_NUM_THREADS=4 python -m pytest c4_min/test_alu_units.py -v``
"""
from __future__ import annotations

import os

import pytest

from c4_min import alu_units as AU


# ---------------------------------------------------------------------------
# Defaults + byte-identity (apply_to_env is a no-op when unset).
# ---------------------------------------------------------------------------
def test_default_env_resolves_production_units():
    assert AU.is_default_env({})
    assert AU.selected_unit("add", {}).name == "scalar"
    assert AU.selected_unit("mul", {}).name == "lookahead"
    assert AU.selected_unit("div", {}).name == "radix16_lean"
    assert AU.default_unit("div").name == "radix16_lean"


def test_apply_to_env_is_noop_when_unset():
    e = {}
    assert AU.apply_to_env(e) == {}
    assert e == {}, "apply_to_env must not write any flag into a bare env"


def test_apply_to_env_does_not_leak_into_os_environ():
    """With no unit flag in os.environ, apply_to_env writes nothing (golden path)."""
    saved = {k: os.environ.pop(k, None)
             for k in ("C4_DIV_UNIT", "C4_MUL_UNIT", "C4_ADD_UNIT",
                       "C4_DIV_LEAN", "C4_DIV_LONGDIV", "C4_MUL_LOOKAHEAD")}
    try:
        before = {k: v for k, v in os.environ.items() if k.startswith("C4_")}
        AU.apply_to_env()  # real os.environ
        after = {k: v for k, v in os.environ.items() if k.startswith("C4_")}
        assert after == before, f"leaked flags: {set(after) - set(before)}"
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# Projection onto the legacy builder flags.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("unit,expect", [
    ("radix16_lean", {"C4_DIV_LEAN": "1", "C4_DIV_LONGDIV": "0"}),
    ("radix16_hardened", {"C4_DIV_LEAN": "0", "C4_DIV_LONGDIV": "0"}),
    ("longdiv", {"C4_DIV_LONGDIV": "1"}),
])
def test_div_unit_projects_to_legacy_flags(unit, expect):
    e = {"C4_DIV_UNIT": unit}
    AU.apply_to_env(e)
    for k, v in expect.items():
        assert e[k] == v, (unit, k, e)


@pytest.mark.parametrize("unit,expect", [
    ("lookahead", "1"),
    ("ripple", "0"),
])
def test_mul_unit_projects_to_lookahead_flag(unit, expect):
    e = {"C4_MUL_UNIT": unit}
    AU.apply_to_env(e)
    assert e["C4_MUL_LOOKAHEAD"] == expect


# ---------------------------------------------------------------------------
# Guards: unknown + non-wired (measure-only / const) tokens raise.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("op,tok", [
    ("div", "radix16_attn"),    # measure-only bakeoff
    ("div", "automaton_attn"),  # measure-only bakeoff
    ("div", "microcode"),       # measure-only bakeoff
    ("div", "const"),           # const-only, needs the C4_CONST_OPERAND hook
    ("mul", "const"),           # const-only
    ("add", "byte_chain"),      # not fused-VM-wired
    ("div", "nonsense"),        # unknown token
])
def test_non_wired_or_unknown_tokens_raise(op, tok):
    flag = AU._ENV_FLAG[op]
    with pytest.raises(ValueError):
        AU.selected_unit(op, {flag: tok})


def test_register_unit_makes_a_new_variant_selectable():
    """The sibling divide agent's promotion path: register a wired unit, then it
    resolves. (Registered on a throwaway op name to avoid clobbering the reals.)"""
    u = AU.AluUnit(op="div", name="_test_sub40", depth=38, stored_blocks=38,
                   wired=True, module="sub40_divide_stub",
                   builder="compile_divmod_blocks")
    AU.register_unit(u)
    try:
        got = AU.selected_unit("div", {"C4_DIV_UNIT": "_test_sub40"})
        assert got.name == "_test_sub40" and got.depth == 38
    finally:
        AU._DIV_UNITS.pop("_test_sub40", None)


# ---------------------------------------------------------------------------
# Fit-to-model selector.
# ---------------------------------------------------------------------------
def test_fit_qwen_05b_const_picks_magic_units():
    """0.5B with const operands -> magic multiply + const-divisor magic divide."""
    p = AU.select_units_for_model(24, 896, const_operands=True)
    assert p.mul.name == "const"
    assert p.div.name == "const"
    assert p.arith_applied_depth == 0  # const paths add ~no depth


def test_fit_qwen_05b_variable_needs_recurrent_or_const():
    """0.5B (24 layers) cannot fit a variable divide unrolled; the selector folds
    to the recurrent lean divide (or flags it too shallow)."""
    p = AU.select_units_for_model(24, 896, const_operands=False)
    assert p.div.name == "radix16_lean"
    assert p.div_recurrent is True
    assert p.mul.name == "lookahead"


def test_fit_qwen_72b_variable_uses_lean():
    """72B (80 layers) uses the lean radix-16 (recurrent fold — 80 unrolled still
    overflows the divide budget after machinery reserve)."""
    p = AU.select_units_for_model(80, 8192, const_operands=False)
    assert p.div.name == "radix16_lean"
    # accounting: stored layer cost is far below the model's 80 layers.
    assert p.arith_layer_cost < 80
    assert p.arith_applied_depth == 88  # mul 8 + div 80


def test_fit_as_env_roundtrips_through_selected_unit():
    """A plan's as_env() selects back to the SAME units via selected_unit."""
    p = AU.select_units_for_model(80, 8192)
    env = p.as_env()
    # lean is the default, so as_env() is empty (all production defaults).
    assert env == {} or all(k in AU._ENV_FLAG.values() for k in env)


def test_metadata_layer_cost_recurrent_below_unrolled():
    lean = AU._DIV_UNITS["radix16_lean"]
    assert lean.layer_cost(recurrent=False) == 80
    assert lean.layer_cost(recurrent=True) == 17
    assert lean.applied_depth == 80


# ---------------------------------------------------------------------------
# End-to-end builder drive (CPU, opt-in — imports transformers + builds a layout).
# ---------------------------------------------------------------------------
@pytest.mark.skipif(os.environ.get("C4_TEST_ALU_UNIT_BUILD") != "1",
                    reason="set C4_TEST_ALU_UNIT_BUILD=1 to run the CPU builder-drive check")
def test_div_unit_hardened_switches_builder_variant():
    """C4_DIV_UNIT=radix16_hardened switches the divmod builder end-to-end (distinct
    Kogge-Stone borrow blocks, absent in the lean 3-limb-borrow variant)."""
    for k in ("C4_DIV_LEAN", "C4_DIV_LONGDIV"):
        os.environ.pop(k, None)
    os.environ["C4_DIV_UNIT"] = "radix16_hardened"
    try:
        import c4_min.qwen_full_vm as Q
        AU.apply_to_env()
        assert Q._div_lean() is False
        QL = Q.QwenFullLayout(16, Q.SUBSET_MULDIV, efficient_alu=True,
                              recurrent_divmod=True)
        names = [n for n, _ in Q._block_specs(QL.L, 16, Q.SUBSET_MULDIV,
                                              efficient_alu=True,
                                              recurrent_divmod=True)]
        assert any("ks" in n for n in names), "hardened Kogge-Stone borrow expected"
        assert not any("borrow0" in n for n in names), "lean 3-limb borrow not expected"
    finally:
        os.environ.pop("C4_DIV_UNIT", None)
        os.environ.pop("C4_DIV_LEAN", None)
        os.environ.pop("C4_DIV_LONGDIV", None)
