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
