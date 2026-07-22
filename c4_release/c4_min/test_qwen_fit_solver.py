"""Tests for the MODEL-FIT CONFIGURATOR (qwen_fit_solver).

Default tests are pure CPU ACCOUNTING (no model materialised) and MEMORY-SAFE: they
size configs via the analytic lookup-table width + the memory-light muldiv-OFF build
(peak ~4.5 GB RSS, NOT the 45 GB the real 256x256x3 table would allocate). The one
bake+verify test is @pytest.mark.slow and builds a SMALL base-subset VM (fits stock
0.5B) to confirm measure_built reads back the accounted geometry.

Run:  pytest c4_min/test_qwen_fit_solver.py            (accounting only, seconds)
      pytest c4_min/test_qwen_fit_solver.py --runslow  (adds the tiny bake+verify)
"""
from __future__ import annotations

import pytest

from c4_min import isa
from c4_min import qwen_full_vm as Q
from c4_min import qwen_fit_solver as S


def test_subset_for_ops_lights_the_right_flags():
    assert S.subset_for_ops(S.BASE_OPS) == Q.Subset(name="base")
    full = S.subset_for_ops(S.FULL)
    assert full.memory and full.cmp and full.bitwise and full.muldiv
    mul_only = S.subset_for_ops(S.BASE_OPS | S.MUL_OPS)
    assert mul_only.muldiv and not mul_only.bitwise and not mul_only.cmp


def test_full_covers_every_op_family():
    for op in (S.MEM_OPS | S.CMP_OPS | S.BITWISE_OPS | S.MULDIV_OPS | S.BASE_OPS):
        assert op in S.FULL


@pytest.mark.parametrize("mode,strat", [
    ("unrolled", "efficient-ALU-unrolled"),
    ("recurrent", "efficient-ALU-recurrent"),
])
def test_accounting_matches_fit_report_efficient(mode, strat):
    gt = {(r["subset"], r["mode"]): r for r in Q.fit_report_efficient()}
    md = frozenset(S.BASE_OPS | S.MEM_OPS | S.CMP_OPS | S.MULDIV_OPS)
    for subname, ops in (("+muldiv", md), ("full", S.FULL)):
        r = gt[(subname, mode)]
        acc = S.account(S.FitConfig(ops=ops, muldiv_strategy=strat, precision=32))
        assert acc.hidden == r["hidden_size"], (subname, mode)
        assert acc.intermediate == r["intermediate_size"], (subname, mode)
        assert acc.stored_layers == r["n_stored_layers"], (subname, mode)
        assert acc.applied_depth == r["n_applied_layers"], (subname, mode)


def test_analytic_lookup_table_matches_the_real_table_width():
    from c4_min.nibble_unified import _MDM_FN
    real_keys = sum(1 for op in (isa.MUL, isa.DIV, isa.MOD)
                    for a in range(256) for b in range(256) if _MDM_FN[op](a, b) != 0)
    assert S._mdm_lookup_width() == real_keys + 1 == 160465
    acc = S.account(S.FitConfig(ops=S.FULL, muldiv_strategy="lookup-table", precision=8))
    assert acc.intermediate == 160465
    assert acc.hidden == 1600 and acc.stored_layers == 17
    assert acc.verified


def _lut(): return S.account(S.FitConfig(ops=S.FULL, muldiv_strategy="lookup-table", precision=8))
def _alu(p=32): return S.account(S.FitConfig(ops=S.FULL, muldiv_strategy="efficient-ALU-unrolled", precision=p))
def _rec(): return S.account(S.FitConfig(ops=S.FULL, muldiv_strategy="efficient-ALU-recurrent", precision=32))
def _sub(p=8): return S.account(S.FitConfig(ops=S.FULL, muldiv_strategy="subroutine", precision=p))


def test_lookup_table_buys_zero_depth_for_huge_width():
    lut, alu = _lut(), _alu()
    assert lut.intermediate > 10 * alu.intermediate
    assert lut.stored_layers < alu.stored_layers
    assert lut.steps_per_op == 1 and alu.steps_per_op == 1


def test_efficient_alu_buys_small_width_for_large_depth():
    alu = _alu()
    assert alu.intermediate < 20000
    assert alu.stored_layers > 100


def test_recurrent_stores_fewer_than_it_applies():
    rec, unr = _rec(), _alu()
    assert rec.stored_layers < unr.stored_layers
    assert rec.applied_depth >= rec.stored_layers


def test_subroutine_buys_near_zero_width_depth_for_many_steps():
    sub, lut = _sub(), _lut()
    assert sub.steps_per_op > 1
    assert sub.intermediate < lut.intermediate
    assert sub.stored_layers <= lut.stored_layers
    assert not sub.verified


def test_subroutine_steps_scale_with_precision():
    assert _sub(8).steps_per_op < _sub(16).steps_per_op < _sub(32).steps_per_op


def test_precision_scales_efficient_alu_depth():
    assert _alu(8).applied_depth < _alu(32).applied_depth
    assert not _alu(8).verified
    assert _alu(32).verified


def test_verified_vs_estimated_labels():
    assert _lut().verified
    assert _alu(32).verified
    assert _rec().verified
    assert not S.account(S.FitConfig(ops=S.FULL, muldiv_strategy="lookup-table", precision=16)).verified
    assert not _sub(8).verified
    assert not _alu(16).verified


_BIG = dict(max_hidden=4096, max_intermediate=200000, max_depth=300)


def test_no_muldiv_subset_is_feasible_and_verified():
    ops = frozenset(S.BASE_OPS | S.MEM_OPS | S.CMP_OPS | S.BITWISE_OPS)
    res = S.fit(ops=ops, minimize="depth")
    assert res.ok and res.best.verified and res.best.steps_per_op == 1


def test_full_does_not_fit_stock_0_5b_hidden():
    res = S.fit("stock-0.5b", ops=S.FULL, minimize="depth")
    assert not res.ok
    assert res.binding == "hidden"
    assert res.relaxation and "hidden" in res.relaxation


def test_max_depth_10_infeasible_binding_is_depth():
    res = S.fit(ops=S.FULL, max_depth=10, minimize="depth")
    assert not res.ok and res.binding == "depth (stored layers)"


def test_max_width_4864_infeasible_binding_is_intermediate():
    res = S.fit(ops=S.FULL, max_width=4864, minimize="depth")
    assert not res.ok and res.binding == "intermediate"


def test_objective_min_steps_prefers_lookup_over_subroutine():
    res = S.fit(ops=S.FULL, minimize="steps", **_BIG)
    assert res.ok and res.best.steps_per_op == 1
    assert res.best.config.muldiv_strategy == "lookup-table"


def test_objective_min_depth_prefers_the_shallowest_feasible():
    res = S.fit(ops=S.FULL, minimize="depth", **_BIG)
    assert res.ok and res.best.stored_layers == 15


def test_require_buildable_filters_to_verified_only():
    res = S.fit(ops=S.FULL, minimize="depth", require_buildable=True, **_BIG)
    assert res.ok and res.best.verified
    assert all(a.verified for a in res.feasible)


def test_fits_named_needs_a_target():
    with pytest.raises(ValueError):
        S.fit(ops=S.FULL, minimize="fits-named")


def test_tradeoff_table_and_summary_render():
    res = S.fit(ops=S.FULL, minimize="depth", **_BIG)
    tbl = S.tradeoff_table(res)
    assert "hidden" in tbl and "stored" in tbl and "stp/op" in tbl
    assert "BEST" in S.best_summary(res)
    inf = S.fit("stock-0.5b", ops=S.FULL, minimize="depth")
    assert "INFEASIBLE" in S.tradeoff_table(inf)
    assert "Binding" in S.best_summary(inf)


@pytest.mark.slow
def test_bake_rejects_estimate_only_configs():
    with pytest.raises(NotImplementedError):
        S.bake(S.FitConfig(ops=S.FULL, muldiv_strategy="subroutine", precision=8))
    with pytest.raises(NotImplementedError):
        S.bake(S.FitConfig(ops=S.FULL, muldiv_strategy="efficient-ALU-unrolled", precision=16))


@pytest.mark.slow
def test_bake_base_geometry_matches_accounting():
    # base subset fits stock 0.5B -> tiny memory-safe bake; measure_built reads back
    # the ACCOUNTED (hidden, intermediate, stored, applied) geometry.
    cfg = S.FitConfig(ops=frozenset(S.BASE_OPS), code_size=24)
    acc = S.account(cfg)
    vm = S.bake(cfg)
    m = S.measure_built(vm)
    assert m["hidden_size"] == acc.hidden
    assert m["intermediate_size"] == acc.intermediate
    assert m["stored_layers"] == acc.stored_layers
    assert m["applied_depth"] == acc.applied_depth
    assert m["hidden_size"] == acc.hidden and acc.hidden < 4864  # small, memory-safe
