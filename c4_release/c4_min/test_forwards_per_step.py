"""Tests for c4_min.forwards_per_step — the depth-via-forwards-per-step lever.

Covers (per the task brief):
  * the (L, F) enumeration is valid + de-duplicated;
  * the compute-conserved invariant L*F ~= D in every split;
  * KV grows with the forwards factor F;
  * the SHALLOW split fits a stock-24 budget while the DEEP (D~=51) split does not;
  * a min-KV (deep) vs vanilla-fit (shallow) Pareto example for D=51 and D=10;
  * the apply_forwards_per_step integration hook.

Pure CPU, no model built (the module never touches the build path).
"""
from __future__ import annotations

import math

import pytest

from c4_min import forwards_per_step as FPS


# ---------------------------------------------------------------------------
# (L, F) enumeration.
# ---------------------------------------------------------------------------
def test_valid_splits_cover_depth_and_are_minimal():
    D = 51
    pairs = FPS.valid_splits(D, max_layers=D)      # allow the full deep corner
    assert pairs, "expected at least one split"
    for L, F in pairs:
        assert 1 <= L <= D
        assert F == math.ceil(D / L)               # minimal forwards at that L
        assert L * F >= D                           # covers the depth
    # de-duplicated on (L, F)
    assert len(pairs) == len(set(pairs))
    # the two corners are present.
    Ls = {L for L, _ in pairs}
    assert 1 in Ls                                  # shallowest (F = D)
    assert D in Ls                                  # deepest (F = 1)
    # shallowest is L=1, F=D; deepest is L=D, F=1.
    assert (1, D) in pairs
    assert (D, 1) in pairs


def test_valid_splits_respects_max_layers():
    pairs = FPS.valid_splits(51, max_layers=24)
    assert all(L <= 24 for L, _ in pairs)
    assert max(L for L, _ in pairs) == 24           # capped at the budget


def test_valid_splits_rejects_bad_args():
    with pytest.raises(ValueError):
        FPS.valid_splits(0, max_layers=24)
    with pytest.raises(ValueError):
        FPS.valid_splits(10, max_layers=0)


# ---------------------------------------------------------------------------
# Compute-conserved invariant: L*F ~= D layer-applications per VM step.
# ---------------------------------------------------------------------------
def test_compute_conserved_every_split_D51():
    r = FPS.depth_realizations(51, max_layers=51)
    report = FPS.conserved_report(r)
    assert report, "expected splits"
    assert all(report.values()), f"compute NOT conserved: {report}"
    # every split does ~D layer-applications (D <= L*F <= D + (L-1)).
    for s in r.splits:
        assert s.layer_applications >= s.D
        assert s.layer_applications <= s.D + (s.n_layers - 1)
        assert s.slack == s.layer_applications - s.D


def test_compute_conserved_is_D_up_to_ceil_rounding():
    D = 51
    # exact-divisor splits waste ZERO compute; the ceil bound holds for all.
    assert FPS.compute_conserved(D, 51, 1)          # L=D, F=1: exactly D
    assert FPS.compute_conserved(D, 1, 51)          # L=1, F=D: exactly D
    assert FPS.compute_conserved(D, 3, 17)          # 3*17 = 51 exactly
    assert FPS.compute_conserved(D, 17, 3)          # 17*3 = 51 exactly
    # a non-divisor still conserves within the ceil slack.
    L = 10
    F = math.ceil(D / L)                            # 6 -> L*F = 60, slack 9 < L=10
    assert FPS.compute_conserved(D, L, F)
    # under-covering the depth is NOT conserved (fails the D-cover).
    assert not FPS.compute_conserved(D, 10, 5)      # 50 < 51


def test_forwards_per_step_does_not_cut_compute():
    # The DEEP and SHALLOW corners do the SAME ~D layer-applications.
    r = FPS.depth_realizations(51, max_layers=51)
    deep = r.deepest
    shallow = r.shallowest
    assert deep.forwards_per_step == 1
    assert shallow.n_layers == 1
    # both cover D with ~equal compute (deep exact, shallow exact at L=1).
    assert deep.layer_applications == 51
    assert shallow.layer_applications == 51
    # forwards-per-step re-books depth into sequence, NOT fewer FLOPs.
    assert shallow.forwards_per_step == 51
    assert deep.forwards_per_step == 1


# ---------------------------------------------------------------------------
# KV grows with the forwards factor F.
# ---------------------------------------------------------------------------
def test_kv_grows_with_forwards_factor():
    D = 51
    r = FPS.depth_realizations(D, max_layers=D)
    # KV scales with effective_seq_len = seq_len * F; for the SAME physical layer
    # count the F-inflation raises KV monotonically.  Compare across a fixed L=1
    # column by varying F directly through apply_forwards_per_step.
    base = {"n_layers": D, "seq_len": 100, "batch": 1, "precision": "fp64",
            "n_heads": 2, "head_dim": 64}
    g1 = FPS.apply_forwards_per_step(base, 1)
    # F=1 shallow-is-deep: n_layers_physical == D.
    assert g1["n_layers_physical"] == D
    assert g1["effective_seq_len"] == 100
    # Now hold the SHALLOW L=1 net and grow F: KV must grow ~linearly in F.
    shallow_seq = {"n_layers": 1, "seq_len": 100, "batch": 1, "precision": "fp64",
                   "n_heads": 2, "head_dim": 64}
    kv_by_F = []
    for F in (1, 2, 5, 10, 51):
        g = FPS.apply_forwards_per_step(shallow_seq, F)
        assert g["effective_seq_len"] == 100 * F
        kv_by_F.append(g["kv_bytes"])
    assert kv_by_F == sorted(kv_by_F)               # monotone increasing in F
    # exactly linear in F at fixed physical layers (n_layers_physical==1 here).
    assert kv_by_F[1] == 2 * kv_by_F[0]
    assert kv_by_F[4] == 51 * kv_by_F[0]


def test_deep_split_small_seq_large_layers_shallow_opposite():
    # DEEP: n_layers large (=D), seq small (F=1).  SHALLOW: n_layers small, seq large.
    r = FPS.depth_realizations(51, max_layers=51, seq_len=64)
    deep, shallow = r.deepest, r.shallowest
    assert deep.n_layers > shallow.n_layers          # deep has more physical layers
    assert deep.effective_seq_len < shallow.effective_seq_len  # deep has shorter seq
    assert deep.n_layers == 51 and deep.effective_seq_len == 64
    assert shallow.n_layers == 1 and shallow.effective_seq_len == 64 * 51


# ---------------------------------------------------------------------------
# Shallow fits stock-24; deep (D~=51) does not.
# ---------------------------------------------------------------------------
def test_full_isa_D51_shallow_fits_stock24_deep_does_not():
    v = FPS.fits_stock_vanilla(FPS.D_FULL_ISA, max_layers=24)
    assert v.D == 51
    # distinct-layers (L=D=51) does NOT fit a 24-layer stock feed-forward.
    assert v.distinct_layers.n_layers == 51
    assert not v.distinct_fits
    # a shallow forwards-per-step split DOES fit.
    assert v.shallow_fits
    assert v.shallow_fit is not None
    assert v.shallow_fit.n_layers <= 24
    # the shallowest fit is L=1 (max forwards) — fewest stored params.
    assert v.shallow_fit.n_layers == 1
    assert v.shallow_fit.forwards_per_step == 51


def test_div_D10_fits_stock24_as_distinct_layers():
    # DIV's D~=10 <= 24, so even the DISTINCT-layers stack fits a stock 24-layer.
    v = FPS.fits_stock_vanilla(FPS.D_DIV, max_layers=24)
    assert v.D == 10
    assert v.distinct_fits                           # L=10 <= 24
    assert v.shallow_fits


def test_module_stock_budget_constant():
    assert FPS.STOCK_VANILLA_MAX_LAYERS == 24


# ---------------------------------------------------------------------------
# Pareto: min-KV (deep) vs vanilla-fit (shallow) for D=51 and D=10.
# ---------------------------------------------------------------------------
def test_pareto_min_kv_is_deep_vanilla_fit_is_shallow_D51():
    r = FPS.depth_realizations(51, max_layers=51, seq_len=2048)
    front = r.pareto
    assert front, "expected a Pareto frontier"
    # the frontier is ordered DEEP -> SHALLOW.
    assert front[0].n_layers >= front[-1].n_layers
    # min-KV corner == the DEEP split (fewest tokens/step -> smallest cache).
    mk = r.min_kv()
    assert mk.forwards_per_step == 1
    assert mk.n_layers == 51
    assert mk in front                               # deep corner is non-dominated
    # min-latency corner is ALSO deep (fewest launches ~ F).
    assert min(r.splits, key=lambda s: s.launch_count).forwards_per_step == 1
    # vanilla-fit corner == the SHALLOW split (fewest stored layers) that fits a
    # stock 24-layer budget.
    vf = r.vanilla_fit(max_layers=24)
    assert vf is not None
    assert vf.n_layers <= 24
    assert vf.n_layers == 1                          # shallowest fitting == min params
    assert vf in front                               # shallow corner is non-dominated
    # the deep and shallow corners genuinely TRADE: KV TIES (both are exact-divisor
    # splits: L*F == D, so n_layers*seq_len is the SAME 51*base core), deep wins
    # strictly on LAUNCHES (min-latency), shallow wins strictly on STORED layers
    # (min-params / vanilla-fit).  This is the honest first-order KV coupling — the
    # KV difference between the pure corners is zero, not deep<shallow.
    assert mk.kv_bytes == vf.kv_bytes                # KV ties (L*F == D for both)
    assert mk.launch_count < vf.launch_count         # deep wins latency
    assert mk.n_layers > vf.n_layers                 # shallow wins stored params
    # neither corner dominates the other -> BOTH are on the Pareto frontier.


def test_pareto_D10():
    r = FPS.depth_realizations(10, max_layers=10, seq_len=2048)
    front = r.pareto
    assert front
    mk = r.min_kv()
    assert mk.forwards_per_step == 1 and mk.n_layers == 10
    vf = r.vanilla_fit(max_layers=24)                # D=10 all fits 24; shallowest=1
    assert vf is not None and vf.n_layers == 1
    # KV ties at the exact-divisor corners (L*F == D == 10); deep wins launches,
    # shallow wins stored layers.
    assert mk.kv_bytes == vf.kv_bytes
    assert mk.launch_count < vf.launch_count
    assert mk.n_layers > vf.n_layers
    # both corners on the frontier.
    assert mk in front and vf in front


def test_deepest_shallowest_accessors():
    r = FPS.depth_realizations(20, max_layers=20)
    assert r.deepest.n_layers == 20 and r.deepest.forwards_per_step == 1
    assert r.shallowest.n_layers == 1 and r.shallowest.forwards_per_step == 20
    assert r.deepest.is_distinct_layers
    assert not r.shallowest.is_distinct_layers


# ---------------------------------------------------------------------------
# KV formula agreement with the solver's kv_cache_bytes.
# ---------------------------------------------------------------------------
def test_kv_bytes_matches_solver_formula():
    # 2 (K+V) * n_layers * n_heads * head_dim * seq_len * batch * precision_bytes
    from c4_min import opconfig as OC
    n_layers, n_heads, head_dim, seq_len, batch, prec = 24, 2, 64, 2048, 4, "fp64"
    expect = (2 * n_layers * n_heads * head_dim * seq_len * batch
              * OC.precision_bytes(prec))
    assert FPS.kv_bytes(n_layers, n_heads, head_dim, seq_len, batch, prec) == expect
    # cross-check against the concurrent solver's own function if importable.
    try:
        from c4_min import qwen_fit_solver as S
    except Exception:
        return
    if hasattr(S, "kv_cache_bytes"):
        assert (FPS.kv_bytes(n_layers, n_heads, head_dim, seq_len, batch, prec)
                == S.kv_cache_bytes(n_layers, n_heads, head_dim, seq_len, batch, prec))


# ---------------------------------------------------------------------------
# apply_forwards_per_step — the integration hook.
# ---------------------------------------------------------------------------
def test_apply_forwards_per_step_identity_F1():
    geom = {"n_layers": 51, "hidden": 896, "n_heads": 2, "head_dim": 64,
            "seq_len": 2048, "batch": 1, "precision": "fp64"}
    out = FPS.apply_forwards_per_step(geom, 1)
    assert out["n_layers_physical"] == 51            # deep: physical == D
    assert out["forwards_per_step"] == 1
    assert out["effective_seq_len"] == 2048
    assert out["layer_applications"] == 51
    assert out["D"] == 51
    assert out["hidden"] == 896                       # width passed through unchanged
    # original dict not mutated.
    assert "n_layers_physical" not in geom


def test_apply_forwards_per_step_shallow():
    geom = {"n_layers": 51, "hidden": 896, "n_heads": 2, "head_dim": 64,
            "seq_len": 2048, "batch": 1, "precision": "fp64"}
    # F=13 -> physical = ceil(51/13) = 4 layers (fits stock 24), seq *13.
    out = FPS.apply_forwards_per_step(geom, 13)
    assert out["n_layers_physical"] == math.ceil(51 / 13) == 4
    assert out["forwards_per_step"] == 13
    assert out["tokens_added_per_step"] == 13
    assert out["effective_seq_len"] == 2048 * 13
    # compute conserved: physical * F >= D (~D).
    assert out["layer_applications"] >= 51
    assert FPS.compute_conserved(51, out["n_layers_physical"], 13)
    # KV re-sized on (physical layers, inflated seq).
    from c4_min import opconfig as OC
    assert out["kv_bytes"] == (2 * 4 * 2 * 64 * (2048 * 13) * 1
                               * OC.precision_bytes("fp64"))


def test_apply_forwards_per_step_defaults_for_partial_geometry():
    # only n_layers present -> defaults fill the KV context.
    out = FPS.apply_forwards_per_step({"n_layers": 10}, 2)
    assert out["n_layers_physical"] == 5
    assert out["effective_seq_len"] == FPS.DEFAULT_SEQ_LEN * 2
    assert out["kv_bytes"] > 0


def test_apply_forwards_per_step_kv_grows_with_F():
    geom = {"n_layers": 51, "seq_len": 128, "batch": 1, "precision": "fp64",
            "n_heads": 2, "head_dim": 64}
    kvs = [FPS.apply_forwards_per_step(geom, F)["kv_bytes"] for F in (1, 2, 4, 8)]
    assert kvs == sorted(kvs)                         # KV grows with F


def test_apply_forwards_per_step_rejects_bad_args():
    with pytest.raises(ValueError):
        FPS.apply_forwards_per_step({"n_layers": 10}, 0)
    with pytest.raises(KeyError):
        FPS.apply_forwards_per_step({"hidden": 896}, 2)   # missing n_layers
    with pytest.raises(ValueError):
        FPS.apply_forwards_per_step({"n_layers": 0}, 2)


# ---------------------------------------------------------------------------
# Worked-table smoke: D=51 (full ISA) and D=10 (DIV) render + have both corners.
# ---------------------------------------------------------------------------
def test_worked_tables_render():
    for D in (FPS.D_FULL_ISA, FPS.D_DIV):
        r = FPS.depth_realizations(D, max_layers=D)
        txt = r.table()
        assert f"D={D}" in txt
        assert "kv_bytes" in txt
        # both corners represented.
        assert any(s.forwards_per_step == 1 for s in r.splits)
        assert any(s.n_layers == 1 for s in r.splits)
