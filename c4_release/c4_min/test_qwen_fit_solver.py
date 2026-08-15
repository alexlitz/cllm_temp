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
    # The dense 256x256 MUL/DIV/MOD table has been REMOVED; its width survives ONLY as
    # the documented ANALYTIC constant Q._MDM_TABLE_WOULD_BE (computed tensor-free from
    # the tiny _MDM_FN truth table, never building a table).  The lookup-table strategy
    # is a pure accounting/tradeoff row that reports that hypothetical width.
    from c4_min.nibble_unified import _MDM_FN
    real_keys = sum(1 for op in (isa.MUL, isa.DIV, isa.MOD)
                    for a in range(256) for b in range(256) if _MDM_FN[op](a, b) != 0)
    assert Q._MDM_TABLE_WOULD_BE == real_keys + 1 == 160465
    assert S._mdm_lookup_width() == Q._MDM_TABLE_WOULD_BE == 160465
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


def test_full_fits_stock_0_5b_at_the_honest_subroutine_floor():
    # THE FALSELY-DEEP-DEFAULT FIX: the DEFAULT fit() now accounts each strategy at its
    # MEASURED feasible floor, so stock-0.5b FITS the full fp32 ISA at the subroutine
    # floor (14 stored, 896/896) — NOT the falsely-deep summed radix geometry that made
    # the old default report INFEASIBLE (hidden 1152 / stored 31).
    res = S.fit("stock-0.5b", ops=S.FULL, minimize="depth")
    assert res.ok
    assert res.best.stored_layers == 14
    assert res.best.hidden <= 896 and res.best.intermediate <= 4864
    assert res.best.config.muldiv_strategy == "subroutine"
    assert res.best.config._is_floor_bundle()


def test_legacy_deep_default_still_reports_the_falsely_deep_infeasibility():
    # feasible_floor=False restores the LEGACY all-levers-OFF (deep) enumeration — the
    # falsely-deep default that summed the radix-16 digit-recurrence.  Kept SELECTABLE
    # for byte-identity comparison; it reports stock-0.5b INFEASIBLE on hidden as before.
    res = S.fit("stock-0.5b", ops=S.FULL, minimize="depth", feasible_floor=False)
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
    # a genuinely-infeasible box (max_depth 10 < the shallowest floor 14) renders the
    # INFEASIBLE / Binding lines (stock-0.5b now FITS at the honest floor, so use a cap).
    inf = S.fit(ops=S.FULL, max_depth=10, minimize="depth")
    assert "INFEASIBLE" in S.tradeoff_table(inf)
    assert "Binding" in S.best_summary(inf)


# =========================================================================== #
# OPCONFIG geometry — HONEST per model-mode accounting (recurrence honesty).
#
# The KEY correctness the fix installs: a STANDARD feed-forward transformer CANNOT
# weight-tie, so its clever geometry is the SUMMED UNROLLED depth (does NOT fit
# stock 0.5B on DEPTH); only the LOOPED / Universal-Transformer variant reuses a
# few cells and fits a 0.5B-WIDTH checkpoint (labelled UT, not stock feed-forward).
# =========================================================================== #
from c4_min import opconfig as OC


def test_default_opconfig_routes_to_nibble_full_geometry():
    g = S.account_opconfig(OC.DEFAULT)
    assert g.hidden == 3008 and not g.fits_stock_0_5b   # nibble FULL blocked by WIDTH
    assert g.binding == "hidden"
    assert g.looped_transformer is False


def test_clever_standard_feedforward_unrolls_and_fails_on_depth():
    # min_params_config is a LOOPED/UT config; the STANDARD feed-forward version
    # UNROLLS -> n_layers = summed unrolled depth -> exceeds stock 0.5B's 24 layers.
    ff = OC.force_standard_feedforward(OC.min_params_config())
    g = S.account_opconfig(ff)
    assert g.looped_transformer is False
    assert g.recurrence == "unrolled"
    # summed unrolled depth = arith 11 + div 10 + mul 20 + bitwise 8 + mem 1 + triv 1
    n, fam = S.summed_unrolled_depth(ff)
    assert n == 51 and g.stored_layers == 51
    assert fam == {"arith": 11, "div": 10, "mul": 20, "bitwise": 8,
                   "memory": 1, "trivial": 1}
    # fits WIDTH (896<=896) but NOT DEPTH (51 > 24) -> does NOT fit stock 0.5B.
    assert g.hidden == 896 and g.intermediate <= S._STOCK_0_5B.intermediate
    assert not g.fits_stock_0_5b
    assert g.binding == "depth (stored layers)"


def test_clever_looped_ut_fits_0_5b_width_as_ut_not_stock():
    g = S.account_opconfig(OC.min_params_config())     # tied / looped / UT
    assert g.looped_transformer is True
    assert g.recurrence == "tied"
    assert g.stored_layers == 6                          # few reused cells
    assert g.applied_depth == 20                         # deepest single op (MUL fp128)
    assert g.fits_stock_0_5b                             # fits a 0.5B-WIDTH UT checkpoint
    assert "Universal-Transformer" in g.model_mode
    assert "NOT stock feed-forward" in g.note


def test_bf16_radix16_standard_vs_looped():
    looped = OC.min_walltime_config()
    std = OC.force_standard_feedforward(looped)
    gl = S.account_opconfig(looped)
    gs = S.account_opconfig(std)
    # LOOPED fits 0.5B width; STANDARD unrolled does not (depth).
    assert gl.fits_stock_0_5b and gl.looped_transformer
    assert not gs.fits_stock_0_5b and gs.binding == "depth (stored layers)"
    # radix-16 limb depths: arith 8 + div 8 + mul 16 + bitwise 8 + mem 1 + triv 1 = 42
    n, _ = S.summed_unrolled_depth(std)
    assert n == 42 and gs.stored_layers == 42


def test_standard_ff_unrolled_is_shallower_and_narrower_than_nibble():
    # clever-UNROLLED is BOTH narrower (896 vs 3008) AND shallower (51 vs 123) than
    # nibble FULL — the digit-extract depth is far less than the 189-block nibble
    # long division — yet still exceeds 24 layers, so it does NOT fit stock 0.5B.
    nib = S.account_opconfig(OC.DEFAULT)
    clv = S.account_opconfig(OC.force_standard_feedforward(OC.min_params_config()))
    assert clv.hidden < nib.hidden                       # narrower
    assert clv.stored_layers < nib.stored_layers         # shallower
    assert clv.stored_layers > S._STOCK_0_5B.layers       # but still > 24
    assert not clv.fits_stock_0_5b and not nib.fits_stock_0_5b


def test_opconfig_geometry_table_shows_honest_per_mode_verdict():
    configs = [
        ("nibble-fp32", OC.DEFAULT),
        ("clever-fp64-UNROLLED", OC.force_standard_feedforward(OC.min_params_config())),
        ("clever-fp64-TIED", OC.min_params_config()),
    ]
    tbl = S.opconfig_geometry_table(configs)
    assert "mode" in tbl and "std-FF" in tbl and "loop/UT" in tbl
    assert "no(depth)" in tbl        # the standard-FF clever row fails on DEPTH
    assert "UT-width" in tbl         # the looped row fits a UT-width checkpoint


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


# =========================================================================== #
# JOINT HARD-CONSTRAINT SOLVER — precision + depth + width + KV-cache budget, all
# checked at once, wired to opconfig's per-op AxisConfig.  Reports the binding
# constraint (and the axis to relax) on infeasibility, and the precision<->KV<->
# depth coupling (min_kv_precision).  See docs/SOLVER_CONSTRAINTS.md.
# =========================================================================== #

# A KV budget context that is comfortably ROOMY for the fits-all test (looped fp64,
# tiny seq/batch), and geometry caps at the 0.5B budget.
_ROOMY = dict(max_hidden=896, max_intermediate=4864, max_layers=24,
              kv_budget_bytes=256 * 1024 * 1024, seq_len=512, batch=1)


def test_precision_bytes_helper():
    assert OC.precision_bytes("int8") == 1
    assert OC.precision_bytes("fp16") == 2 == OC.precision_bytes("bf16")
    assert OC.precision_bytes("fp32") == 4
    assert OC.precision_bytes("fp64") == 8
    assert OC.precision_bytes("fp128") == 16
    with pytest.raises(OC.OpConfigError):
        OC.precision_bytes("nope")


def test_kv_formula_matches_hand_computation():
    # KV = 2 * n_layers_kv * n_heads * head_dim * seq * batch * precision_bytes.
    kv = S.kv_cache_bytes(n_layers_kv=20, n_heads=2, head_dim=64, seq_len=512,
                          batch=1, precision="fp64")
    assert kv == 2 * 20 * 2 * 64 * 512 * 1 * 8
    # int8 is 1/8 the bytes/elem of fp64 at the same geometry.
    kv8 = S.kv_cache_bytes(20, 2, 64, 512, 1, "int8")
    assert kv8 * 8 == kv


# ---- (a) a config that fits ALL FOUR hard constraints simultaneously --------
def test_solve_opconfig_fits_all_four_constraints():
    # looped/UT fp64 whole-value: hidden 896, stored 6, applied 20 — fits 0.5B width
    # + depth, roomy KV, precision fp64 valid.
    cfg = OC.min_params_config()
    con = S.FitConstraints(precision="fp64", **_ROOMY)
    r = S.solve_opconfig(cfg, con)
    assert r.fits
    assert r.binding_constraint is None and r.relax_axis is None
    # KV sized on APPLIED depth (20), NOT the 6 stored cells.
    assert r.n_layers_kv == r.geometry.applied_depth == 20
    assert r.kv_bytes == S.kv_cache_bytes(20, 2, 64, 512, 1, "fp64")
    # every slack non-negative.
    assert r.slack.layers >= 0 and r.slack.hidden >= 0
    assert r.slack.intermediate >= 0 and r.slack.kv_bytes >= 0
    assert r.slack.radix_valid is True


# ---- (b) EACH constraint individually binding -------------------------------
def test_depth_bound_binds():
    # standard-FF unrolled fp64 = 51 stored layers > max_layers 24, everything else
    # roomy -> depth binds.
    ff = OC.force_standard_feedforward(OC.min_params_config())
    con = S.FitConstraints(max_layers=24, max_hidden=4096, max_intermediate=200000,
                           kv_budget_bytes=10 ** 15, seq_len=512, batch=1)
    r = S.solve_opconfig(ff, con)
    assert not r.fits
    assert r.binding_constraint == "depth (max_layers)"
    assert "max_layers" in r.relax_axis
    assert r.slack.layers < 0
    assert r.geometry.stored_layers == 51


def test_forwards_per_step_meets_depth_bound_kv_neutral():
    # standard-FF unrolled fp64 = 51 stored layers > max_layers 24 (fails depth at
    # F=1).  Re-booking depth into the autoregressive loop with forwards_per_step=3
    # needs only ceil(51/3)=17 stored layers <= 24 -> FITS, and KV is CONSERVED
    # (the layers<->forwards trade is KV-neutral in the solver).
    ff = OC.force_standard_feedforward(OC.min_params_config())
    base = dict(max_layers=24, max_hidden=4096, max_intermediate=200000,
                kv_budget_bytes=10 ** 15, seq_len=512, batch=1)
    r1 = S.solve_opconfig(ff, S.FitConstraints(**base))                       # F=1
    r3 = S.solve_opconfig(ff, S.FitConstraints(forwards_per_step=3, **base))  # F=3
    assert not r1.fits and r1.binding_constraint == "depth (max_layers)"
    assert r3.fits and r3.binding_constraint is None       # forwards rescues depth
    assert r3.kv_bytes == r1.kv_bytes                      # KV-neutral
    assert "forwards_per_step=3" in r3.notes
    assert r3.slack.layers == 24 - 17                      # ceil(51/3)=17 effective


def test_width_hidden_bound_binds():
    # DEFAULT nibble hidden=3008 > max_hidden 896; depth/inter/KV roomy -> hidden.
    con = S.FitConstraints(max_hidden=896, max_layers=10 ** 6,
                           max_intermediate=10 ** 9, kv_budget_bytes=10 ** 15,
                           seq_len=1, batch=1)
    r = S.solve_opconfig(OC.DEFAULT, con)
    assert not r.fits and r.binding_constraint == "width (max_hidden)"
    assert r.slack.hidden < 0 and "max_hidden" in r.relax_axis


def test_width_intermediate_bound_binds():
    # DEFAULT nibble intermediate=7920 > max_intermediate 100; else roomy -> inter.
    con = S.FitConstraints(max_hidden=10 ** 6, max_layers=10 ** 6,
                           max_intermediate=100, kv_budget_bytes=10 ** 15,
                           seq_len=1, batch=1)
    r = S.solve_opconfig(OC.DEFAULT, con)
    assert not r.fits and r.binding_constraint == "width (max_intermediate)"
    assert r.slack.intermediate < 0


def test_kv_bound_binds():
    # looped fp64 fits width+depth+precision, but a huge seq/batch overruns a tiny
    # KV budget -> KV binds.
    cfg = OC.min_params_config()
    con = S.FitConstraints(precision="fp64", max_layers=24, max_hidden=896,
                           max_intermediate=4864, kv_budget_bytes=1024,
                           seq_len=8192, batch=64)
    r = S.solve_opconfig(cfg, con)
    assert not r.fits and r.binding_constraint == "kv_cache (kv_budget_bytes)"
    assert r.slack.kv_bytes < 0
    assert "lower precision" in r.relax_axis  # the KV-shrink relaxation is surfaced


def test_precision_forces_invalid_radix_binds():
    # DIV at radix 4096 needs fp32's 2^24 ceiling (r^2 = 16.7M); requiring int8
    # (ceiling 127) makes the radix INVALID -> precision binds.  (Keep every OTHER
    # op whole_value so ONLY the DIV radix is the invalid-radix bind.)
    ov = {op: dict(precision="int8", extraction="whole_value") for op in OC.ALL_OPS}
    ov["DIV"] = dict(precision="int8", radix=4096, extraction="digit_extract")
    bad = OC.OpConfig(base=OC.AxisConfig(precision="int8", extraction="whole_value"),
                      overrides=ov)
    con = S.FitConstraints(precision="int8", max_layers=10 ** 6, max_hidden=10 ** 6,
                           max_intermediate=10 ** 9, kv_budget_bytes=10 ** 15)
    r = S.solve_opconfig(bad, con)
    assert not r.fits and r.binding_constraint == "precision (invalid radix)"
    assert r.slack.radix_valid is False
    assert "DIV" in r.notes and "overflows int8" in r.notes


def test_multiple_binds_reports_most_binding_first():
    # DEFAULT nibble overshoots hidden (3008 vs 896) AND depth (123 vs 24); the
    # LARGEST relative overshoot (depth 123/24=5.1x > hidden 3008/896=3.4x) is
    # reported as THE binding constraint.
    con = S.FitConstraints(max_hidden=896, max_layers=24, max_intermediate=10 ** 9,
                           kv_budget_bytes=10 ** 15, seq_len=1, batch=1)
    r = S.solve_opconfig(OC.DEFAULT, con)
    assert not r.fits
    assert r.binding_constraint == "depth (max_layers)"   # bigger relative overshoot


# ---- (c) precision<->KV coupling: min_kv_precision picks correctly -----------
def test_min_kv_precision_fp64_wins_at_small_seq_batch():
    # TINY seq/batch: KV is negligible, so the seq-independent stored-param depth
    # dominates -> the FEW-LAYER fp64 whole-value config wins (natural mode).
    res = S.min_kv_precision(S.FitConstraints(seq_len=4, batch=1))
    assert res.best_precision == "fp64"
    # fp64 row is the FEW-layer one; int8 is the MANY-layer one.
    by = {r.precision: r for r in res.frontier}
    assert by["fp64"].stored_layers < by["int8"].stored_layers
    assert by["fp64"].bytes_per_elem == 8 and by["int8"].bytes_per_elem == 1
    assert by["fp64"].total_bytes <= by["int8"].total_bytes


def test_min_kv_precision_int8_wins_at_large_seq_batch():
    # HUGE seq/batch: KV dominates total memory, so the tiny-bytes/elem int8 config
    # wins DESPITE its many layers / greater applied depth.
    res = S.min_kv_precision(S.FitConstraints(seq_len=16384, batch=128))
    assert res.best_precision == "int8"
    by = {r.precision: r for r in res.frontier}
    # int8 has MORE applied depth than fp64 but WINS on total because of bytes/elem.
    assert by["int8"].applied_depth > by["fp64"].applied_depth
    assert by["int8"].kv_bytes < by["fp64"].kv_bytes       # 1 byte/elem << 8
    assert by["int8"].total_bytes < by["fp64"].total_bytes


def test_min_kv_precision_surfaces_the_depth_coupling():
    # The HONEST coupling: LOWER precision -> LOWER radix ceiling -> MORE applied
    # depth.  int8's DIV radix ceiling (8) is far below fp32's (4096), so int8's
    # applied depth EXCEEDS fp32's — the depth cost of dropping precision.
    res = S.min_kv_precision(S.FitConstraints(seq_len=1024, batch=1))
    by = {r.precision: r for r in res.frontier}
    assert by["int8"].max_safe_radix_div < by["fp32"].max_safe_radix_div
    assert by["int8"].applied_depth > by["fp32"].applied_depth   # more limbs=deeper
    assert by["fp64"].max_safe_radix_div is None                 # whole-value


def test_min_kv_precision_pure_kv_objective_ranks_by_depth_times_bytes():
    # objective='kv': ranks by applied_depth * bytes/elem (the seq*batch factor is
    # common).  int8 (22*1=22) beats fp64 (20*8=160) on pure per-token KV.
    res = S.min_kv_precision(S.FitConstraints(seq_len=2048, batch=8), objective="kv")
    assert res.best_precision == "int8"
    by = {r.precision: r for r in res.frontier}
    assert by["int8"].kv_bytes < by["fp64"].kv_bytes


# ---- (d) looped vs unrolled: SAME KV (applied depth), DIFFERENT stored params -
def test_looped_vs_unrolled_same_kv_diff_params():
    looped = OC.min_params_config()                       # tied/UT: stored 6
    unrolled = OC.force_standard_feedforward(looped)      # unrolled: stored 51
    con = S.FitConstraints(precision="fp64", seq_len=1024, batch=4,
                           max_layers=10 ** 6, max_hidden=10 ** 6,
                           max_intermediate=10 ** 9, kv_budget_bytes=10 ** 18)
    rl = S.solve_opconfig(looped, con)
    ru = S.solve_opconfig(unrolled, con)
    # SAME applied depth -> SAME KV (the loop unrolls into the cache; stored-cell
    # reduction does NOT shrink KV).
    assert rl.n_layers_kv == ru.n_layers_kv == 20
    assert rl.kv_bytes == ru.kv_bytes
    # DIFFERENT stored params (looped stores fewer distinct cells).
    assert rl.geometry.stored_layers == 6 and ru.geometry.stored_layers == 51
    assert rl.geometry.params_estimate < ru.geometry.params_estimate


def test_joint_result_summary_and_table_render():
    r = S.solve_opconfig(OC.min_params_config(),
                         S.FitConstraints(precision="fp64", **_ROOMY))
    assert "FITS" in r.summary()
    inf = S.solve_opconfig(OC.DEFAULT,
                           S.FitConstraints(max_hidden=896, seq_len=1, batch=1))
    assert "INFEASIBLE" in inf.summary() and inf.binding_constraint in inf.summary()
    tbl = S.min_kv_precision(S.FitConstraints(seq_len=4, batch=1)).table()
    assert "prec" in tbl and "KV" in tbl and "total" in tbl and "WIN" in tbl


def test_kv_heads_default_to_gqa_kv_head_count():
    # KV cache is keyed on the GQA KEY-VALUE heads (2 for 0.5B), NOT the 14 query
    # heads — the honest cache head count.
    con = S.FitConstraints(seq_len=128, batch=1)
    assert con.kv_heads(S.QWEN2_5_ARCH) == 2         # num_key_value_heads
    assert con.kv_head_dim(S.QWEN2_5_ARCH) == 64
    # an explicit override wins.  Pin precision=fp64 so KV bytes/elem is fp64 (the
    # config's own deepest op is MUL@fp128, which the solver otherwise sizes KV on).
    con2 = S.FitConstraints(precision="fp64", seq_len=128, batch=1, n_heads=14,
                            head_dim=64)
    assert con2.kv_heads(S.QWEN2_5_ARCH) == 14
    r = S.solve_opconfig(OC.min_params_config(), con2)
    assert r.kv_bytes == S.kv_cache_bytes(20, 14, 64, 128, 1, "fp64")


# ===========================================================================
# #913 fp32-fit levers (pack_memcam / overlap_scratch / bit_level_bitwise).
# Each lever DEFAULTS OFF, so the base accounting is byte-identical to before;
# ON, it applies a MEASURED residual/depth reduction from the real layout.
# ===========================================================================
STOCK = S.STOCK_TARGETS["stock-0.5b"]


def _fits_05b(a):
    return (a.hidden <= STOCK.hidden and a.intermediate <= STOCK.intermediate
            and a.stored_layers <= STOCK.layers)


def test_levers_default_off_is_byte_identical():
    # levers OFF -> the prior accounting (subroutine 1152/896/31, ALU 3008/7920).
    sub = S.account(S.FitConfig(ops=S.FULL, muldiv_strategy="subroutine"))
    assert (sub.hidden, sub.intermediate, sub.stored_layers) == (1152, 896, 31)
    alu = S.account(S.FitConfig(ops=S.FULL, muldiv_strategy="efficient-ALU-recurrent"))
    assert (alu.hidden, alu.intermediate) == (3008, 7920)


def test_pack_memcam_drops_the_dead_pad_to_896():
    # dropping the dead §Memory head pad collapses the subroutine hidden 1152 -> 896.
    a = S.account(S.FitConfig(ops=S.FULL, muldiv_strategy="subroutine", pack_memcam=True))
    assert a.hidden == 896 and a.hidden < 1152


def test_bit_level_bitwise_retires_the_barrel_shifter_depth():
    # the #911 depth-1 bitwise + SHL/SHR-as-subroutine retires the ~17 barrel-shift
    # blocks: subroutine stored 31 -> ~14.
    off = S.account(S.FitConfig(ops=S.FULL, muldiv_strategy="subroutine"))
    on = S.account(S.FitConfig(ops=S.FULL, muldiv_strategy="subroutine",
                               bit_level_bitwise=True))
    assert on.stored_layers < off.stored_layers
    assert off.stored_layers - on.stored_layers >= 15


def test_full_fp32_isa_FITS_stock_0_5b_with_all_levers():
    # THE #913 result: subroutine muldiv + pack_memcam + bit-level bitwise FITS the
    # stock Qwen2.5-0.5B box (hidden<=896, inter<=4864, stored<=24) at fp32.
    a = S.account(S.FitConfig(ops=S.FULL, muldiv_strategy="subroutine", precision=32,
                              pack_memcam=True, bit_level_bitwise=True))
    assert a.hidden <= 896 and a.intermediate <= 4864 and a.stored_layers <= 24
    assert _fits_05b(a)


def test_overlap_scratch_reduces_inline_alu_but_divmod_still_binds():
    # op-overlap replaces the SUM of the per-op ALU scratch with operand + MAX-family;
    # it narrows the inline-ALU residual but divmod's own scratch still exceeds 896,
    # so the inline ALU does NOT fit even fully overlapped (subroutine is the lever).
    off = S.account(S.FitConfig(ops=S.FULL, muldiv_strategy="efficient-ALU-recurrent",
                                pack_memcam=True))
    on = S.account(S.FitConfig(ops=S.FULL, muldiv_strategy="efficient-ALU-recurrent",
                               pack_memcam=True, overlap_scratch=True))
    assert on.hidden <= off.hidden          # overlap narrows the residual
    assert on.hidden > 896                  # ...but divmod scratch still binds
    assert not _fits_05b(on)


# ===========================================================================
# #916 continued — INLINE + FEED-FORWARD fit levers (attn-CAM DIV LR-dup removal,
# bit-level SHIFT, op-overlap max-op DEPTH).  All default OFF (byte-identical base).
# ===========================================================================
def _inline_ff(**kw):
    base = dict(ops=S.FULL, muldiv_strategy="efficient-ALU-unrolled", precision=32,
                code_size=24)
    base.update(kw)
    return S.account(S.FitConfig(**base))


def test_attn_cam_div_removes_the_duplicate_LR_lean_radix_residual():
    # attn_cam_div swaps the radix-16 lean-radix DIV for the R256E attention-CAM head,
    # so BOTH the ALU_* divmod scratch (1156) AND the DUPLICATE LR_* lean-radix datapath
    # (480, touched only by the removed lean-* blocks) leave the residual -> hidden drops
    # to <=896.  (Without the LR-dup removal the residual would be ~480 too wide.)
    g = S._attn_cam_div_geometry(24, False)
    assert g["lr_dup_residual"] > 0                     # the duplicate datapath is real
    off = _inline_ff(pack_memcam=True, overlap_scratch=True, bit_level_bitwise=True)
    on = _inline_ff(pack_memcam=True, overlap_scratch=True, bit_level_bitwise=True,
                    attn_cam_div=True)
    assert on.hidden < off.hidden                       # divmod+LR leave the residual


def test_bit_level_shift_retires_the_dead_barrel_shifter_residual():
    # SHL/SHR via native MUL/DIV (shift_via_mul, the fit regime) leaves the barrel-shifter
    # RESIDUAL (SH_STAGE_* 160 + TS_* 118 = 278) allocated but touched by NO block; the
    # bit_level_shift lever retires that dead residual -> hidden 960 -> 896.
    saving = S._bitlevel_shift_saving(24, True)
    assert saving >= 278
    off = _inline_ff(pack_memcam=True, overlap_scratch=True, bit_level_bitwise=True,
                     attn_cam_div=True)
    on = _inline_ff(pack_memcam=True, overlap_scratch=True, bit_level_bitwise=True,
                    bit_level_shift=True, attn_cam_div=True)
    assert on.hidden < off.hidden
    assert on.hidden <= 896


def test_inline_ff_hidden_and_inter_FIT_but_depth_binds():
    # THE #916-continued result: inline + feed-forward (no subroutine, no loop) FITS the
    # stock 0.5B WIDTH — hidden <= 896 AND inter <= 4864 — but the DIV/MOD unrolled
    # digit-recurrence (~98 layers) makes the op-overlap max-op DEPTH ~108 > 24, so it
    # does NOT fit stock DEPTH as a feed-forward (unrolled) transformer.
    a = _inline_ff(pack_memcam=True, overlap_scratch=True, bit_level_bitwise=True,
                   bit_level_shift=True, attn_cam_div=True, overlap_depth=True)
    assert a.hidden <= 896                               # WIDTH fits
    assert a.intermediate <= 4864                        # INTERMEDIATE fits
    assert a.stored_layers > 24                          # DEPTH binds (divmod unrolled)


def test_overlap_depth_is_maxop_not_sum():
    # op-overlap: only one opcode fires per step -> depth = shared pipeline + deepest
    # single op-family cascade (max-op), strictly LESS than the SUMMED unrolled depth.
    summed = _inline_ff(pack_memcam=True, overlap_scratch=True, bit_level_bitwise=True,
                        bit_level_shift=True, attn_cam_div=True)
    maxop = _inline_ff(pack_memcam=True, overlap_scratch=True, bit_level_bitwise=True,
                       bit_level_shift=True, attn_cam_div=True, overlap_depth=True)
    assert maxop.stored_layers < summed.stored_layers   # max-op < sum
    assert maxop.stored_layers > 24                      # ...still > 24 (divmod unrolled)


def test_new_levers_default_off_leave_base_accounting_unchanged():
    # every #916-continued lever defaults OFF -> the base inline-ALU accounting is
    # byte-identical to the prior pass (3008 / 7920 / 123).
    a = _inline_ff()
    assert (a.hidden, a.intermediate, a.stored_layers) == (3008, 7920, 123)


# ===========================================================================
# #916 continued — LOG-SINK DIV depth lever (§653).  Tests whether the shallow
# softmax1-reciprocal divide closes the DEPTH axis that the radix-256 R256E div
# left binding (108 > 24).  MEASURED result: it does NOT — the log-sink compiled
# feed-forward block count (127) is DEEPER than R256E (98), because the quotient
# DECODE + schoolbook VERIFY are themselves 8-nibble digit-recurrences.  Log-sink
# does RELAX intermediate (its widest FFN 2422 < R256E 3376).  All DEFAULT OFF.
# ===========================================================================
def test_logsink_geometry_is_measured_and_deeper_than_the_stage_count():
    # The §653 docstring's "~14 blocks" is the STAGE count; the compiled feed-forward
    # chain is MEASURED far deeper — the quotient decode + schoolbook verify unroll into
    # four 8-nibble MSB-first digit-recurrences (+ two schoolbook passes).
    g = S._logsink_div_geometry(24)
    assert g["blocks"] == 127                            # measured compiled block count
    assert g["fixed_blocks"] == 15                       # the genuinely-shallow ~14 stages
    assert g["decompose_blocks"] == 96                   # 4x 8-nibble digit-recurrence
    assert g["schoolbook_blocks"] == 16                  # 2x q*b product/carry/rem
    # the shallow "~14" part is a small MINORITY of the honest depth.
    assert g["fixed_blocks"] < g["decompose_blocks"]
    assert g["blocks"] > g["fixed_blocks"] * 8


def test_logsink_div_relaxes_intermediate_but_deepens_depth():
    # Swap the R256E radix-256 div for the §653 log-sink div in the inline+FF fit:
    #   * INTERMEDIATE RELAXES (widest FFN 2422 < R256E 3376) — still <= 4864.
    #   * DEPTH WORSENS (127-block log-sink cascade > 98-block R256E) — 108 -> 137.
    # Log-sink MOVES the digit-recurrence (estimate -> decode+verify); it does not remove it.
    r256e = _inline_ff(pack_memcam=True, overlap_scratch=True, bit_level_bitwise=True,
                       bit_level_shift=True, attn_cam_div=True, overlap_depth=True)
    logsink = _inline_ff(pack_memcam=True, overlap_scratch=True, bit_level_bitwise=True,
                         bit_level_shift=True, attn_cam_div=True, overlap_depth=True,
                         logsink_div=True)
    assert logsink.intermediate < r256e.intermediate     # log-sink narrower FFN
    assert logsink.intermediate <= 4864                  # still fits INTERMEDIATE
    assert logsink.hidden == r256e.hidden == 896         # hidden unchanged (fits)
    assert logsink.stored_layers > r256e.stored_layers   # ...but DEEPER
    assert logsink.stored_layers == 137 and r256e.stored_layers == 108


def test_logsink_div_does_not_close_depth_for_any_stock_model():
    # THE GOAL VERDICT: log-sink div does NOT make inline + feed-forward fit stock
    # DEPTH — not 0.5B (24), not 1.5B (28), not 7B (28).  Depth 137 > all caps.
    logsink = _inline_ff(pack_memcam=True, overlap_scratch=True, bit_level_bitwise=True,
                         bit_level_shift=True, attn_cam_div=True, overlap_depth=True,
                         logsink_div=True)
    assert logsink.hidden <= 896 and logsink.intermediate <= 4864     # WIDTH fits
    for tgt in ("stock-0.5b", "stock-1.5b", "stock-7b"):
        assert logsink.stored_layers > S.STOCK_TARGETS[tgt].layers    # DEPTH binds everywhere


def test_logsink_div_default_off_is_byte_identical():
    # logsink_div DEFAULT OFF -> the inline+FF accounting is the prior-pass R256E one.
    off = _inline_ff(pack_memcam=True, overlap_scratch=True, bit_level_bitwise=True,
                     bit_level_shift=True, attn_cam_div=True, overlap_depth=True)
    assert (off.hidden, off.intermediate, off.stored_layers) == (896, 3376, 108)
    # logsink_div is a no-op WITHOUT the attn_cam_div divmod-swap it modifies: the bare
    # base inline-ALU accounting (no attn_cam_div) is byte-identical with the flag on/off.
    on = _inline_ff(logsink_div=True)
    base = _inline_ff()
    assert (on.hidden, on.intermediate, on.stored_layers, on.applied_depth) == \
           (base.hidden, base.intermediate, base.stored_layers, base.applied_depth) == \
           (3008, 7920, 123, 123)


# ===========================================================================
# #916 continued — ADDER-HACK decode depth lever (blogspec §Position Offset).
# Tests whether the place-value nibble decode (d_j = floor(v/16^j) mod 16, each
# nibble INDEPENDENT -> depth-1) closes the DEPTH that the log-sink schoolbook
# running-remainder decode (4 x 8-deep) left binding.  MEASURED result: the
# depth-1 decode is BYTE-EXACT (fp64) but WIDTH-INFEASIBLE (~4.9e9 single-layer
# ramps: the mod-16 fold needs an unbounded floor staircase, blogspec §734); the
# bounded-width realization is the SEQUENTIAL cascade log-sink already bakes.  And
# EVEN the ideal depth-1 divmod (21) + shared base (10) = 31 > 24/28 — the
# reciprocal + q*b VERIFY multiply dominate.  All DEFAULT OFF.
# ===========================================================================
def test_adderhack_decode_geometry_is_measured():
    g = S._adderhack_div_geometry(24)
    assert g["logsink_blocks"] == 127
    assert g["decode_site_blocks"] == 100        # the 4x 8-deep running-remainder decode
    assert g["nondecode_blocks"] == 27           # reciprocal + q*b verify + correct + finalize
    assert g["decode_passes"] == 4               # q, r, d, m
    assert g["shallowest_wired_mul"] == 8         # Kogge-Stone carry-lookahead MUL
    # the depth-1 decode is byte-exact but WIDTH-infeasible (billions of ramps).
    assert g["depth1_ffn_width_ramps"] > 4_000_000_000
    # ideal depth-1 divmod << 127, but still large (reciprocal + verify dominate).
    assert g["ideal_onepass_divmod"] == 21
    assert g["bounded_divmod"] == 127            # realizable == sequential == log-sink


def test_adderhack_decode_collapses_the_decode_but_reciprocal_and_verify_dominate():
    # Apply the IDEAL depth-1 adder-hack decode: divmod cascade 127 -> 21, so the
    # inline+FF depth drops 137 -> 31.  A big cut, but STILL > 24 (0.5B) and > 28
    # (1.5B/7B): the reciprocal (8) + q*b VERIFY multiply (10) are the residual wall,
    # not the decode.
    logsink = _inline_ff(pack_memcam=True, overlap_scratch=True, bit_level_bitwise=True,
                         bit_level_shift=True, attn_cam_div=True, overlap_depth=True,
                         logsink_div=True)
    adderhack = _inline_ff(pack_memcam=True, overlap_scratch=True, bit_level_bitwise=True,
                           bit_level_shift=True, attn_cam_div=True, overlap_depth=True,
                           logsink_div=True, adderhack_decode=True)
    assert logsink.stored_layers == 137
    assert adderhack.stored_layers == 31         # 10 shared base + 21 ideal divmod
    assert adderhack.stored_layers < logsink.stored_layers   # decode collapse helps a LOT
    assert adderhack.hidden == 896 and adderhack.intermediate <= 4864   # WIDTH still fits
    # ...but DEPTH still binds every stock model (even ideal-decode).
    for tgt in ("stock-0.5b", "stock-1.5b", "stock-7b"):
        assert adderhack.stored_layers > S.STOCK_TARGETS[tgt].layers


def test_adderhack_decode_default_off_is_byte_identical():
    off = _inline_ff(pack_memcam=True, overlap_scratch=True, bit_level_bitwise=True,
                     bit_level_shift=True, attn_cam_div=True, overlap_depth=True,
                     logsink_div=True)
    assert off.stored_layers == 137              # adderhack OFF -> log-sink schoolbook depth
    # adderhack is a no-op WITHOUT logsink_div (it only rewrites the log-sink decode).
    on_nolog = _inline_ff(adderhack_decode=True)
    base = _inline_ff()
    assert (on_nolog.hidden, on_nolog.intermediate, on_nolog.stored_layers) == \
           (base.hidden, base.intermediate, base.stored_layers) == (3008, 7920, 123)


# ===========================================================================
# #916 FINAL PUSH — the three FEASIBLE levers on the two deep divmod terms.
# LEVER 1 (radix256_decode): the bounded-width decode is a max-feasible-radix
#   (radix-4096, 3-limb) cascade = 4 feasible layers/pass (NOT the infeasible
#   depth-1, NOT 8-deep radix-16).  LEVER 2 (newton_min): reciprocal Newton 2->1.
#   LEVER 3 (shallow-verify/tight-base): the q*b product is structurally required
#   for MOD and already the shallowest wired MUL; base is already 7 (no depth cut).
# MEASURED feasible result: divmod 25 + shared 10 = inline+FF 35 — WIDTH still fits
#   (896 / 4096 <= 4864) but DEPTH 35 > 28 (1.5B) > 24 (0.5B): does NOT close.
#   Every changed div sub-block is byte-exact over a large sample
#   (_div_levers_916_check).  All levers DEFAULT OFF.
# ===========================================================================
def test_final_push_geometry_is_measured_feasible():
    g = S._adderhack_div_geometry(24)
    assert g["recip_fixed"] == 8                  # Newton=2 reciprocal
    assert g["recip_fixed_newton_min"] == 6       # Newton=1 (LEVER 2) saves 2 blocks
    assert g["radix256_decode_layers_per_pass"] == 4   # max-feasible-radix (4096, 3-limb)+split
    assert g["radix_decode_ffn_width"] == 4096    # staircase width, <= 4864 INTERMEDIATE
    assert g["feasible_decode"] == 8              # 2 passes x 4 feasible layers
    # feasible divmod = recip6 + verify10 + decode8 + finalize1 = 25 (byte-exact bounded)
    assert g["feasible_divmod"] == 25
    # honest: the feasible bounded divmod (25) is BETWEEN the infeasible depth-1 ideal
    # (21) and the un-levered radix-16 log-sink (127).
    assert g["ideal_onepass_divmod"] < g["feasible_divmod"] < g["bounded_divmod"]


def test_final_push_radix_decode_lever():
    # LEVER 1: the max-feasible-radix decode cuts the divmod cascade from the log-sink
    # 127 to a bounded-width feasible depth; the 4096-wide staircase becomes the widest
    # divmod FFN (4096 <= 4864, INTERMEDIATE still fits).
    logsink = _inline_ff(pack_memcam=True, overlap_scratch=True, bit_level_bitwise=True,
                         bit_level_shift=True, attn_cam_div=True, overlap_depth=True,
                         logsink_div=True)
    radix = _inline_ff(pack_memcam=True, overlap_scratch=True, bit_level_bitwise=True,
                       bit_level_shift=True, attn_cam_div=True, overlap_depth=True,
                       logsink_div=True, radix256_decode=True)
    assert radix.stored_layers < logsink.stored_layers      # feasible decode is shallower
    assert radix.stored_layers == 37                        # shared 10 + divmod 27 (Newton=2)
    assert radix.intermediate == 4096                       # max-radix staircase FFN width
    assert radix.intermediate <= 4864                       # still fits INTERMEDIATE
    assert radix.hidden <= 896                              # WIDTH fits


def test_final_push_newton_min_lever():
    # LEVER 2: Newton 2 -> 1 saves 2 reciprocal blocks (byte-exact incl the adversarial
    # worst-sign corner; Newton=0 FAILS the refine R=512 clamp).
    radix = _inline_ff(pack_memcam=True, overlap_scratch=True, bit_level_bitwise=True,
                       bit_level_shift=True, attn_cam_div=True, overlap_depth=True,
                       logsink_div=True, radix256_decode=True)
    both = _inline_ff(pack_memcam=True, overlap_scratch=True, bit_level_bitwise=True,
                      bit_level_shift=True, attn_cam_div=True, overlap_depth=True,
                      logsink_div=True, radix256_decode=True, newton_min=True)
    assert both.stored_layers == radix.stored_layers - 2    # 2 fewer reciprocal blocks
    assert both.stored_layers == 35                         # shared 10 + divmod 25


def test_final_push_does_not_close_depth_for_any_stock_model():
    # THE GOAL VERDICT (feasible, byte-exact): all three levers applied, the inline +
    # feed-forward depth is 35 — WIDTH fits (896 / 4096 <= 4864) but DEPTH 35 exceeds
    # EVERY stock cap: 0.5B (24), 1.5B (28), 7B (28).  The divmod residual (reciprocal
    # + the structurally-required q*b product + the 2 decode passes) is the irreducible
    # feed-forward binder; no stock model closes on DEPTH.
    both = _inline_ff(pack_memcam=True, overlap_scratch=True, bit_level_bitwise=True,
                      bit_level_shift=True, attn_cam_div=True, overlap_depth=True,
                      logsink_div=True, radix256_decode=True, newton_min=True)
    assert both.hidden <= 896 and both.intermediate <= 4864     # WIDTH fits
    assert both.stored_layers == 35
    for tgt in ("stock-0.5b", "stock-1.5b", "stock-7b"):
        assert both.stored_layers > S.STOCK_TARGETS[tgt].layers  # DEPTH binds everywhere


def test_final_push_levers_default_off_are_byte_identical():
    # radix256_decode + newton_min DEFAULT OFF -> the accounting is the prior-pass one.
    off = _inline_ff(pack_memcam=True, overlap_scratch=True, bit_level_bitwise=True,
                     bit_level_shift=True, attn_cam_div=True, overlap_depth=True,
                     logsink_div=True)
    assert off.stored_layers == 137              # both levers OFF -> log-sink schoolbook
    # the levers are no-ops WITHOUT logsink_div (they only rewrite the log-sink divmod).
    base = _inline_ff()
    r_nolog = _inline_ff(radix256_decode=True)
    n_nolog = _inline_ff(newton_min=True)
    for a in (r_nolog, n_nolog):
        assert (a.hidden, a.intermediate, a.stored_layers) == \
               (base.hidden, base.intermediate, base.stored_layers) == (3008, 7920, 123)


# ===========================================================================
# #916 FINAL PUSH — BYTE-EXACTNESS of every changed div sub-block (a small,
# fast sample; the full 200k-value stress is in _div_levers_916_check.__main__).
# Confirms the depth accounting above is GROUNDED: the radix-256/-4096 decode,
# the reduced-Newton reciprocal, and the truncated q*b verify are all byte-exact.
# ===========================================================================
def test_final_push_radix_decode_is_byte_exact():
    from c4_min import _div_levers_916_check as C
    r = C.run_lever1(20000)                       # 20k values (fast)
    assert r["byte_ok"] == r["total"]             # radix-256 byte decode
    assert r["nib_ok"] == r["total"]              # radix-256 -> 8 nibbles
    assert r["hi_radix_ok"] == r["total"]         # radix-4096 (3-limb) decode


def test_final_push_newton_min_is_byte_exact():
    from c4_min import _div_levers_916_check as C
    # Newton=1 byte-exact on the random sample AND the adversarial worst-sign corner;
    # Newton=0 FAILS both (the honest minimum is 1).
    r = C.run_lever2(8000, worst_relerr=5e-7)
    assert r[1]["ok"] == r[1]["total"]            # Newton=1 exact
    assert r[2]["ok"] == r[2]["total"]            # Newton=2 exact
    assert r[0]["ok"] < r[0]["total"]             # Newton=0 NOT exact (needs >=1)
    c = C.run_lever2_worst_corner(worst_relerr=5e-7, refine_R=512)
    assert c[1]["ok"] == c[1]["total"]            # Newton=1 exact at the corner
    assert c[0]["ok"] < c[0]["total"]             # Newton=0 fails the R=512 refine clamp


def test_final_push_truncated_verify_is_byte_exact():
    from c4_min import _div_levers_916_check as C
    r = C.run_lever3(8000, newton_steps=1)
    assert r["ok"] == r["total"]                  # truncated-to-32-bit q*b byte-exact


# ===========================================================================
# #916 fp64 INLINE+FF divmod — the levers that FAIL at fp32 are byte-exact at fp64
# (2^53 mantissa clears fp32's 2^24 wall).  Precision is a FREE bake choice, so an
# fp64 fit IS goal-met.  fp64 divmod = recip5 + verify7 + decode8 + finalize1 = 21
# (vs fp32 feasible 25), inline+FF 31 (vs 35).  1.5B FITS with the decode-overlap.
# ===========================================================================
def _inline_ff_fp64(**kw):
    return _inline_ff(pack_memcam=True, overlap_scratch=True, bit_level_bitwise=True,
                      bit_level_shift=True, attn_cam_div=True, overlap_depth=True,
                      logsink_div=True, radix256_decode=True, newton_min=True, **kw)


def test_fp64_divmod_geometry_is_measured():
    g = S._fp64_divmod_geometry(24)
    assert g["fp64_recip"] == 5                   # case-split, ZERO Newton (WALL 2 clears)
    assert g["fp64_11bit_mul_depth"] == 5         # 11-bit product replaces Kogge-Stone 8 (WALL 1)
    assert g["fp64_verify"] == 7                  # mul 5 + rem 1 + correct 1
    assert g["fp64_decode"] == 8                  # 2 passes x radix-4096 4 layers
    assert g["fp64_divmod"] == 21                 # recip5 + verify7 + decode8 + finalize1
    assert g["fp64_divmod_overlap"] == 17         # decode-overlap (1.5B-only width)
    assert g["fp64_decode_ffn_width"] == 4096     # <= 4864 (2-pass fits 0.5B intermediate)
    assert g["fp64_decode_ffn_width_overlap"] == 8192   # 1.5B-only (> 4864, <= 8960)
    # fp64 is STRICTLY shallower than the fp32 feasible floor (25).
    fp32 = S._adderhack_div_geometry(24)
    assert g["fp64_divmod"] < fp32["feasible_divmod"] == 25


def test_fp64_divmod_shaves_depth_vs_fp32():
    fp32 = _inline_ff_fp64()
    fp64 = _inline_ff_fp64(fp64_divmod=True)
    assert fp32.stored_layers == 35               # shared 10 + fp32 feasible divmod 25
    assert fp64.stored_layers == 31               # shared 10 + fp64 divmod 21
    assert fp64.stored_layers == fp32.stored_layers - 4    # 4 layers (recip -1, verify -3)
    assert fp64.hidden <= 896 and fp64.intermediate <= 4864    # WIDTH still fits


def test_fp64_divmod_does_not_fit_0_5b_but_fits_1_5b():
    # 0.5B (24): fp64 divmod is 31 > 24 (conservative) and the decode-overlap that would
    # reach 27 blows the 0.5B intermediate (8192 > 4864) -> 0.5B NO FIT on either axis.
    fp64 = _inline_ff_fp64(fp64_divmod=True)
    assert fp64.stored_layers == 31 > S.STOCK_TARGETS["stock-0.5b"].layers   # 31 > 24
    fp64_ov = _inline_ff_fp64(fp64_divmod=True, fp64_decode_overlap=True)
    assert fp64_ov.stored_layers == 27            # decode-overlap floor
    assert fp64_ov.intermediate == 8192 > S.STOCK_TARGETS["stock-0.5b"].intermediate  # 8192 > 4864
    # 1.5B (28): fp64 divmod + decode-overlap = inline+FF 27 <= 28, width 8192 <= 8960 -> FITS.
    st15 = S.STOCK_TARGETS["stock-1.5b"]
    cfg = S.FitConfig(ops=S.FULL, muldiv_strategy="efficient-ALU-unrolled", precision=32,
                      code_size=24, pack_memcam=True, overlap_scratch=True,
                      bit_level_bitwise=True, bit_level_shift=True, attn_cam_div=True,
                      overlap_depth=True, logsink_div=True, radix256_decode=True,
                      newton_min=True, fp64_divmod=True, fp64_decode_overlap=True,
                      arch=st15.arch)
    a = S.account(cfg)
    assert a.hidden <= st15.hidden                # 1536 <= 1536
    assert a.intermediate <= st15.intermediate    # 8192 <= 8960
    assert a.stored_layers <= st15.layers         # 27 <= 28  -> 1.5B FITS at fp64


def test_fp64_divmod_default_off_is_byte_identical():
    # fp64_divmod + fp64_decode_overlap DEFAULT OFF -> accounting is the fp32 feasible one.
    off = _inline_ff_fp64()
    assert off.stored_layers == 35
    # the fp64 lever is a no-op WITHOUT logsink_div (it only rewrites the log-sink divmod).
    base = _inline_ff()
    fp64_nolog = _inline_ff(fp64_divmod=True)
    assert (fp64_nolog.hidden, fp64_nolog.intermediate, fp64_nolog.stored_layers) == \
           (base.hidden, base.intermediate, base.stored_layers) == (3008, 7920, 123)


def test_fp64_levers_are_byte_exact():
    from c4_min import _fp64_div_levers_916_check as F
    # LEVER A: 11-bit-chunk q*b low-32 verify (replaces Kogge-Stone 8).
    ra = F.run_lever_a(20000)
    assert ra["ok"] == ra["total"]
    # LEVER C: radix-4096 fp64 decode.
    rc = F.run_lever_c(20000)
    assert rc["ok"] == rc["total"]
    # standalone MUL (same 11-bit low-32 product).
    rm = F.run_standalone_mul(20000)
    assert rm["ok"] == rm["total"]
    # decode-overlap (q,rem in shared layers).
    ro = F.run_decode_overlap(20000)
    assert ro["ok"] == ro["total"]


def test_fp64_whole_div_gate_is_byte_exact():
    from c4_min import _fp64_div_levers_916_check as F
    # THE GATE: whole fp64 case-split div byte-exact END-TO-END with ZERO Newton
    # (recip + 11-bit verify + radix-4096 decode + correct) over a sample + boundaries.
    rg = F.run_whole_div_gate(15000, newton_steps=0, use_11bit_verify=True)
    assert rg["ok"] == rg["total"]                # div byte-exact
    assert rg["decode_ok"] == rg["total"]         # q,rem nibble decode byte-exact


# ===========================================================================
# THE FALSELY-DEEP-DEFAULT FIX — regression LOCK on the corrected defaults.
#
# The bare all-levers-OFF FitConfig reports the falsely-deep summed radix geometry
# (subroutine 1152/896/31, inline-ALU unrolled 3008/7920/123, recurrent 60/123).  The
# DEFAULT enumerator now accounts each strategy at its MEASURED, byte-exact feasible
# floor.  These tests LOCK the honest floors so they can never re-inflate — the numbers
# MATCH the two reference branches (fp32 @eed17a92, fp64 @ae22dd03) and #913.
# ===========================================================================
def _floor(strat, prec=32, ovl=False, arch=None):
    a = STOCK.arch if arch is None else arch
    return S.account(S.feasible_floor_config(S.FULL, strat, precision=prec,
                                             decode_overlap=ovl, arch=a))


def test_feasible_floor_bundles_match_the_measured_branch_floors():
    # subroutine fp32 = 14 (0.5B FIT, #913 pack_memcam + bit_level_bitwise).
    assert _floor("subroutine").stored_layers == 14
    # inline+FF fp32 = 35 (shared base 10 + feasible divmod 25), @eed17a92.
    assert _floor("efficient-ALU-unrolled").stored_layers == 35
    # inline+FF fp64 = 31 (shared base 10 + fp64 divmod 21), @ae22dd03.
    assert _floor("efficient-ALU-unrolled", prec=64).stored_layers == 31
    # inline+FF fp64 + decode-overlap = 27 (divmod 17, WIDTH 8192), @ae22dd03.
    ov = _floor("efficient-ALU-unrolled", prec=64, ovl=True)
    assert ov.stored_layers == 27 and ov.intermediate == 8192


def test_feasible_floor_bundles_reproduce_the_measured_divmod_arithmetic():
    # the floor depths are shared-base 10 + the MEASURED divmod cascade — grounded in
    # the branch geometry helpers, not asserted constants.
    fp32 = S._adderhack_div_geometry(24)
    fp64 = S._fp64_divmod_geometry(24)
    assert fp32["feasible_divmod"] == 25            # recip6 + verify10 + decode8 + fin1
    assert fp64["fp64_divmod"] == 21                # recip5 + verify7 + decode8 + fin1
    assert fp64["fp64_divmod_overlap"] == 17        # decode-overlap
    # shared base pipeline (base 7 + shared-alu 3) = 10; floor = shared + divmod.
    assert _floor("efficient-ALU-unrolled").stored_layers == 10 + fp32["feasible_divmod"]
    assert _floor("efficient-ALU-unrolled", prec=64).stored_layers == 10 + fp64["fp64_divmod"]


def test_fp64_floor_bundle_keeps_a_32bit_integer_datapath():
    # fp64 is a DTYPE choice, not a 64-bit integer width: the config precision stays 32
    # (so the linear-in-nibble P8/P16 projection never fires) and the fp64 divmod levers
    # are on.
    cfg = S.feasible_floor_config(S.FULL, "efficient-ALU-unrolled", precision=64)
    assert cfg.precision == 32 and cfg.fp64_divmod is True
    cfg_ov = S.feasible_floor_config(S.FULL, "efficient-ALU-unrolled", precision=64,
                                     decode_overlap=True)
    assert cfg_ov.fp64_divmod is True and cfg_ov.fp64_decode_overlap is True


def test_default_fit_reports_the_honest_floor_not_the_falsely_deep_sum():
    # DEFAULT fit() (feasible_floor=True) picks the honest floor; the bare deep config
    # still reports the falsely-deep 123 — the levers stay SELECTABLE, just not silent.
    res = S.fit("stock-0.5b", ops=S.FULL, minimize="depth")
    assert res.ok and res.best.stored_layers == 14
    deep = S.account(S.FitConfig(ops=S.FULL, muldiv_strategy="efficient-ALU-unrolled"))
    assert deep.stored_layers == 123               # bare deep radix still selectable
    rec = S.account(S.FitConfig(ops=S.FULL, muldiv_strategy="efficient-ALU-recurrent"))
    assert rec.stored_layers == 60                 # bare recurrent still selectable


def test_precision_projection_does_not_corrupt_the_floor_depth():
    # the pre-existing linear-in-nibble P8/P16 projection was designed for the DEEP
    # radix ALU depth; applying it on top of the FINAL overlap_depth floor drove P8 to
    # the max(1,...) clamp (a nonsense stored=1).  The floor bundles are a 32-bit
    # datapath by construction and the enumerator never emits a P8/P16 floor, so the
    # default fit's best is never the corrupted stored=1 artifact.
    res = S.fit("stock-7b", ops=S.FULL, minimize="depth")
    assert res.ok and res.best.stored_layers >= 14   # NOT the stored=1 P8 artifact
    # a P8 inline-ALU config with overlap_depth is NOT precision-projected (guarded).
    a = S.account(S.FitConfig(ops=S.FULL, muldiv_strategy="efficient-ALU-unrolled",
                              precision=8, **S._INLINE_FF_FLOOR_LEVERS))
    assert a.stored_layers == 35                     # overlap depth, NOT projected to 1


# ---- THE FIT MATRIX: {0.5B, 1.5B, 7B} x {fp32, fp64} x {subroutine, inline+FF} ------
def _fits_target(a, st):
    return (a.hidden <= st.hidden and a.intermediate <= st.intermediate
            and a.stored_layers <= st.layers)


def test_fit_matrix_subroutine_fp32_floor_fits_every_stock():
    # subroutine fp32 floor = 14 layers, hidden/inter = arch geometry -> fits all three.
    for tname in ("stock-0.5b", "stock-1.5b", "stock-7b"):
        st = S.STOCK_TARGETS[tname]
        a = _floor("subroutine", arch=st.arch)
        assert a.stored_layers == 14
        assert _fits_target(a, st), tname


def test_fit_matrix_inline_ff_fp32_fits_no_stock_on_depth():
    # inline+FF fp32 = 35 > every stock depth cap (24 / 28 / 28) -> NO stock fits fp32.
    for tname in ("stock-0.5b", "stock-1.5b", "stock-7b"):
        st = S.STOCK_TARGETS[tname]
        a = _floor("efficient-ALU-unrolled", arch=st.arch)
        assert a.stored_layers == 35
        assert not _fits_target(a, st), tname       # depth binds everywhere
        assert a.stored_layers > st.layers


def test_fit_matrix_inline_ff_fp64_0_5b_no_but_1_5b_yes_with_overlap():
    # THE #917 RE-VERIFICATION.  inline+FF at fp64-with-overlap = 27 layers:
    #   * 0.5B: NO (fp64 31 > 24; and the 27-overlap blows inter 8192 > 4864).
    #   * 1.5B: YES (27 <= 28 AND inter 8192 <= 8960 AND hidden 1536 <= 1536).
    #   * 7B:   YES (27 <= 28, inter 8192 <= 18944).
    st05, st15, st7 = (S.STOCK_TARGETS[n] for n in ("stock-0.5b", "stock-1.5b", "stock-7b"))
    # 0.5B fp64 (31) and fp64-overlap (27 but width 8192) both fail.
    a05 = _floor("efficient-ALU-unrolled", prec=64, arch=st05.arch)
    assert a05.stored_layers == 31 and not _fits_target(a05, st05)
    a05_ov = _floor("efficient-ALU-unrolled", prec=64, ovl=True, arch=st05.arch)
    assert a05_ov.stored_layers == 27 and a05_ov.intermediate == 8192
    assert not _fits_target(a05_ov, st05)           # 8192 > 4864 AND 27 > 24
    # 1.5B fp64 (31 > 28) NO; fp64-overlap (27 <= 28, 8192 <= 8960) YES.
    a15 = _floor("efficient-ALU-unrolled", prec=64, arch=st15.arch)
    assert a15.stored_layers == 31 and not _fits_target(a15, st15)   # 31 > 28
    a15_ov = _floor("efficient-ALU-unrolled", prec=64, ovl=True, arch=st15.arch)
    assert a15_ov.stored_layers == 27 and a15_ov.intermediate == 8192
    assert _fits_target(a15_ov, st15)               # 27 <= 28, 8192 <= 8960 -> FITS
    # 7B fp64-overlap also fits.
    a7_ov = _floor("efficient-ALU-unrolled", prec=64, ovl=True, arch=st7.arch)
    assert _fits_target(a7_ov, st7)


def test_1_5b_inline_ff_verdict_via_the_default_solver():
    # THE #917 verdict through the FRONT DOOR: cap depth at the inline+FF (efficient-ALU)
    # strategy and confirm the solver reports the honest inline+FF floor for 1.5B — the
    # fp64+overlap floor (27) fits 1.5B's 28-layer / 8960-inter budget, fp32 (35) does not.
    inline_ops = S.FULL
    a_fp32 = _floor("efficient-ALU-unrolled", arch=S.STOCK_TARGETS["stock-1.5b"].arch)
    a_ovl = _floor("efficient-ALU-unrolled", prec=64, ovl=True,
                   arch=S.STOCK_TARGETS["stock-1.5b"].arch)
    st15 = S.STOCK_TARGETS["stock-1.5b"]
    assert a_fp32.stored_layers == 35 > st15.layers                  # fp32 does NOT fit
    assert a_ovl.stored_layers == 27 <= st15.layers                  # fp64+overlap FITS
    assert a_ovl.intermediate <= st15.intermediate


def test_include_deep_surfaces_both_floor_and_deep_rows():
    # include_deep=True keeps the deep tradeoff rows visible (they are never the silent
    # default best, but stay in the ranked table for the honest width/depth tradeoff).
    res = S.fit(ops=S.FULL, minimize="depth", max_hidden=4096,
                max_intermediate=200000, max_depth=300, include_deep=True)
    labels = [a.config.label for a in res.feasible]
    assert any("FLOOR" in l for l in labels)         # honest floors present
    assert any("FLOOR" not in l for l in labels)     # deep tradeoff rows present
    # the default (no include_deep) best is a FLOOR bundle, not a deep row.
    res_floor = S.fit(ops=S.FULL, minimize="depth", max_hidden=4096,
                      max_intermediate=200000, max_depth=300)
    assert res_floor.best.config._is_floor_bundle()
