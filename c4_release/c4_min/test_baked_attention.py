"""Tests for the full-attention baking *completeness* demo (baked_attention.py).

Run: PYTHONPATH=<repo> python -m pytest c4_min/test_baked_attention.py
(or directly: python -m c4_min.test_baked_attention).

Load-bearing claims (BLOG_SPEC.md §"Baking Weights", line 828-836 — "I will also
show how we can simulate full attention using SwiGLU for completeness"):

  1. The five blog gadgets — SiLU-gated multiply for dot products, efficient-exp
     BOS sink for exp, real-number base-16 long division for the softmax1
     denominator's reciprocal, and a final scale-down — reproduce a REAL
     softmax1 attention layer's output on a small case, to a stated tolerance.

  2. The tolerance is controllable: the long division is near-exact and the only
     residual is the efficient-exp sink factor, which shrinks with the bias
     margin (§Efficient Exp, line 563).

  3. This is the general (mixing) case; the KV-RETRIEVAL path the VM actually
     uses (query==key, weights ~ {0,1}) already works and pays none of the
     long-division cost.  Asserted at the doc level only (see module docstring).
"""
from __future__ import annotations

import torch

from c4_min.baked_attention import (
    BakedFullAttention, BakedKV, reference_softmax1_attention,
    _mul, _reciprocal_long_division, _efficient_exp, validate,
)


def _kvs():
    return [
        BakedKV(key=torch.tensor([1., 0., 1., 0.]),
                value=torch.tensor([1., 0., 0., 0.])),
        BakedKV(key=torch.tensor([0., 1., 1., 0.]),
                value=torch.tensor([0., 1., 0., 0.])),
        BakedKV(key=torch.tensor([1., 1., 0., 1.]),
                value=torch.tensor([0., 0., 1., 1.])),
    ]


def test_gated_multiply_primitive():
    """The 6-weight SiLU-gated multiply is fp-exact for bounded operands."""
    for a in range(0, 8):
        for b in range(0, 8):
            got = float(_mul(a, b))
            assert abs(got - a * b) < 1e-6, (a, b, got)


def test_long_division_reciprocal_is_near_exact():
    """Base-16 long division computes 1/d to the fixed-point precision floor."""
    for d in [2.0, 3.0, 4.0, 7.0, 10.0, 100.0]:
        recip, layers = _reciprocal_long_division(torch.tensor(d), 12)
        assert abs(float(recip) - 1.0 / d) < 1e-7, (d, float(recip))
        assert layers == 12  # one sequential long-division layer per nibble


def test_efficient_exp_sink_approximation():
    """exp(N) via the BOS-sink approaches e^N as the bias margin grows."""
    N = torch.tensor(1.5)
    for margin, tol in [(4.0, 2e-2), (12.0, 1e-5)]:
        B = float(N) + margin
        got = float(_efficient_exp(N, B))
        assert abs(got - torch.exp(N).item()) / torch.exp(N).item() < tol, margin


def test_baked_full_attention_matches_reference_tolerance():
    """The full baked attention reproduces softmax1 attention within tolerance."""
    dim = 4
    kvs = _kvs()
    baker = BakedFullAttention(dim, kvs, precision_nibbles=12, exp_bias_margin=12.0)
    queries = [
        torch.tensor([1., 0., 1., 0.]),
        torch.tensor([0., 1., 1., 0.]),
        torch.tensor([1., 1., 1., 1.]),
        torch.tensor([0., 0., 0., 0.]),
    ]
    max_err = 0.0
    for q in queries:
        baked = baker.forward(q)
        ref = reference_softmax1_attention(q, kvs, dim)
        max_err = max(max_err, float((baked.double() - ref.double()).abs().max()))
    # near-exact: long division + sufficient sink margin.
    assert max_err < 1e-5, max_err


def test_tolerance_tightens_with_bias_margin():
    """Larger exp-bias margin => smaller residual (the approximation is the sink)."""
    dim = 4
    kvs = _kvs()
    q = torch.tensor([1., 1., 1., 1.])
    ref = reference_softmax1_attention(q, kvs, dim)

    def err_at(margin):
        b = BakedFullAttention(dim, kvs, precision_nibbles=12, exp_bias_margin=margin)
        return float((b.forward(q).double() - ref.double()).abs().max())

    e_small = err_at(4.0)
    e_large = err_at(12.0)
    assert e_large < e_small
    assert e_large < 1e-5


def test_cost_report_flags_sequential_division():
    """The cost report is honest: long division dominates and is sequential."""
    dim = 4
    baker = BakedFullAttention(dim, _kvs(), precision_nibbles=8)
    baker.forward(torch.tensor([1., 0., 1., 0.]))
    cost = baker.last_cost
    assert cost["long_division_layers_sequential"] == 8
    # long division is the largest SwiGLU-unit contributor
    assert cost["swiglu_units_longdiv"] >= cost["swiglu_units_dot"]
    assert cost["swiglu_units_longdiv"] >= cost["swiglu_units_value"]


def test_validate_runs_and_reports_tolerance():
    out = validate(verbose=False)
    assert out["max_abs_error"] < 1e-5
    assert out["cost"]["swiglu_units_total"] > 0


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
