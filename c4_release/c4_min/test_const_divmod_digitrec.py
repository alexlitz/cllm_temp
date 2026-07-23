#!/usr/bin/env python3
"""Byte-exact tests for the CONSTANT-DIVISOR digit-recurrence divmod gadget.

Runs the built SwiGLU block list through the REAL fp32 forward (``apply_spec`` /
``run_divmod_batch``) and checks BOTH quotient and remainder byte-exact against the
ISA reference (``a//b``, ``a%b``, ``b==0 -> (0,0)``) over random 32-bit dividends +
edges.  Standalone (CPU, fp32, arithmetic sim) — no full-VM build."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

import pytest

from c4_min.const_divmod_digitrec import (
    build_const_divmod_digitrec, run_divmod, run_divmod_batch, _ref_divmod,
    const_divmod_fp32_ceiling, _quot_nibbles, _rem_nibbles,
)


# fp32-EXACT divisors over the FULL 32-bit dividend range (b under the ceiling).
_FP32_EXACT_B = [2, 3, 4, 7, 8, 9, 10, 13, 16, 17, 31, 32, 100, 251, 255, 256,
                 257, 1000, 1024, 2003, 4096, 4099, 5000, 5592]


def _battery(b, in_width, n_random, seed):
    circ = build_const_divmod_digitrec(b, in_width=in_width)
    rnd = random.Random(seed)
    top = (1 << in_width) - 1
    xs = [rnd.randint(0, top) for _ in range(n_random)]
    for e in (0, 1, b - 1, b, b + 1, top):
        if 0 <= e <= top:
            xs.append(e)
    qs, rs = run_divmod_batch(circ, xs)
    for a, q, r in zip(xs, qs, rs):
        eq, er = _ref_divmod(a, b)
        assert q == eq, f"b={b} inW={in_width}: DIV a={a} got q={q} exp {eq}"
        assert r == er, f"b={b} inW={in_width}: MOD a={a} got r={r} exp {er}"
    return circ


@pytest.mark.parametrize("b", _FP32_EXACT_B)
def test_full32_byte_exact(b):
    """Both q and r byte-exact over the FULL 32-bit dividend range for every
    fp32-safe constant divisor (>= 2000 random + edges)."""
    _battery(b, 32, n_random=2000, seed=0)


@pytest.mark.parametrize("b,in_width", [(65521, 16), (40961, 16), (50021, 16),
                                        (4093, 16)])
def test_narrowed_large_b(b, in_width):
    """Large divisors (beyond the 32-bit ceiling) are byte-exact when the dividend
    is statically bounded (in_width so 2^in_width <= ~b) — the remainder then fits in
    <= 4 fp32-exact nibbles and the quotient is small.  The narrowing rescue."""
    _battery(b, in_width, n_random=2000, seed=1)


@pytest.mark.parametrize("b", [2, 3, 7, 10, 16, 100, 255, 1000, 4096])
def test_edges(b):
    """Explicit edge dividends: 0, 1, b-1, b, b+1, 2^32-1."""
    circ = build_const_divmod_digitrec(b, 32)
    for a in (0, 1, b - 1, b, b + 1, 2 * b, 0xFFFFFFFF):
        q, r = run_divmod(circ, a)
        eq, er = _ref_divmod(a, b)
        assert (q, r) == (eq, er), f"b={b} a={a}: got ({q},{r}) exp ({eq},{er})"


def test_bzero_returns_zero_zero():
    """b==0 -> (0, 0) for every dividend (ISA_SPEC 4.2), resolved at build time."""
    circ = build_const_divmod_digitrec(0, in_width=32)
    for a in (0, 1, 255, 65535, 0xFFFFFFFF, 123456789, 2 ** 31):
        assert run_divmod(circ, a) == (0, 0), f"b==0 a={a}"


def test_powers_of_two_via_general_path():
    """Powers of two need no special case — the general staircase is exact."""
    for b in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
        circ = build_const_divmod_digitrec(b, 32)
        rnd = random.Random(b)
        xs = [rnd.randint(0, 2 ** 32 - 1) for _ in range(200)]
        qs, rs = run_divmod_batch(circ, xs)
        for a, q, r in zip(xs, qs, rs):
            assert (q, r) == _ref_divmod(a, b), f"pow2 b={b} a={a}"


def test_narrowing_widths():
    """The quotient/remainder nibble widths match the provably-nonzero counts, and a
    tighter in_width narrows the quotient (fewer nonzero nibbles)."""
    # remainder width = ceil(log16 b) = nibbles to hold r in [0, b-1].
    assert _rem_nibbles(255) == 2 and _rem_nibbles(256) == 2   # r<=255 / r<=255: 2 nibbles
    assert _rem_nibbles(257) == 3                              # r<=256: needs 3 nibbles
    assert _rem_nibbles(15) == 1 and _rem_nibbles(16) == 1     # r<=14 / r<=15: 1 nibble
    assert _rem_nibbles(17) == 2                               # r<=16: needs 2 nibbles
    # quotient width shrinks with a tighter in_width (dividend bound).
    assert _quot_nibbles(7, 32) > _quot_nibbles(7, 16) >= _quot_nibbles(7, 8)
    # a full build reflects those widths.
    c32 = build_const_divmod_digitrec(1000, 32)
    c16 = build_const_divmod_digitrec(1000, 16)
    assert c16.q_nibbles < c32.q_nibbles
    assert c16.in_nibbles < c32.in_nibbles           # fewer dividend nibbles processed
    assert c16.n_blocks < c32.n_blocks               # narrowing -> fewer blocks


def test_fp32_ceiling_is_positive_and_reasonable():
    """The empirical 32-bit-dividend fp32 ceiling is ~12k (readout of r needs <=4
    nibbles for full-divmod exactness; the q-staircase holds further via the ramp's
    difference-of-relus cancellation)."""
    ceil = const_divmod_fp32_ceiling()
    assert 8000 <= ceil <= 16000


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
