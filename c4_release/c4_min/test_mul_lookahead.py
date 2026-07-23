"""Gate for the CARRY-LOOKAHEAD MULTIPLY bakeoff (``mul_lookahead``).

Confirms the headline claims WITHOUT a dense DIM forward — the gadgets run on the
small residual plane via the sparse SwiGLU sim (seconds):

* the baseline is exactly the brief's numbers: 10 blocks, 11430 nz, 7 ripple rounds;
* the Kogge-Stone carry-lookahead is SHALLOWER (8 blocks vs 10) with only 3 prefix
  stages replacing the 7 serial ripples, and roughly HALVES the nz;
* both are byte-exact over the shared edge set + random incl the literal worst case
  ``0xFFFFFFFF^2`` (max carry propagation);
* every relu argument stays deep inside the fp32 exact-integer ceiling (0 fp64).

Run:  OMP_NUM_THREADS=4 PYTHONPATH=$(pwd) python -m pytest c4_min/test_mul_lookahead.py -v
"""
from __future__ import annotations

import pytest

from c4_min import mul_lookahead as ml


@pytest.fixture(scope="module")
def bakeoff():
    return ml.run_bakeoff(n_random=1500, verbose=False)


def test_baseline_matches_brief(bakeoff):
    """The baseline is the brief's stated reference: 10 blocks, 11430 nz, 7 rounds."""
    r = bakeoff["baseline_ripple"]
    assert r["depth"] == 10
    assert r["weights_nz"] == 11430
    assert r["carry_rounds"] == 7


def test_kogge_stone_is_shallower(bakeoff):
    """Carry-lookahead: fewer blocks (8 < 10) and 3 prefix stages replace 7 ripples."""
    base = bakeoff["baseline_ripple"]
    ks = bakeoff["kogge_stone_lookahead"]
    assert ks["depth"] < base["depth"]            # shallower
    assert ks["depth"] == 8
    assert ks["prefix_stages"] == 3               # ceil(log2 8) vs 7 ripple rounds
    assert ks["weights_nz"] < base["weights_nz"]  # and lighter


def test_both_byte_exact_incl_worst_case(bakeoff):
    """Byte-exact over the whole grid, INCLUDING 0xFFFFFFFF^2 (max carry chain)."""
    for name in ("baseline_ripple", "kogge_stone_lookahead"):
        r = bakeoff[name]
        assert r["byte_exact_pass"] == r["byte_exact_total"], (name, r["first_fail"])
        assert r["ffffffff_squared_ok"] is True, name


def test_fp32_discipline(bakeoff):
    """Every relu argument is fp32-exact: RELU_S*max_arg well under 2^24 (0 fp64)."""
    for name in ("baseline_ripple", "kogge_stone_lookahead"):
        r = bakeoff[name]
        assert r["tightest_relu_s_arg"] < ml.FP32_INT_MAX
        assert r["tightest_ratio_of_2p24"] < 0.01   # < 1% of the ceiling — deep headroom


def test_worst_case_explicit():
    """0xFFFFFFFF^2 directly through the Kogge-Stone plane (the carry path's point)."""
    L = ml._make_layout()
    blocks, res, _ = ml.build_kogge_stone(L, ml.DIM)
    M = 0xFFFFFFFF
    got = ml.simulate_sparse(L, blocks, res, M, M)
    assert got == (M * M) & M


def test_div_knockon_reported(bakeoff):
    """The div knock-on is quantified (QB carry reuse), honestly a wash per-iter."""
    dk = bakeoff["_div_knockon"]
    assert dk["qb_ripple_rounds_per_iter"] == 6
    assert dk["qb_ks_prefix_stages"] == 4         # ceil(log2 9)
    assert "wash" in dk["verdict"].lower()
