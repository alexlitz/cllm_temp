"""NATIVE-fp32 VM self-emulation — the fp32-scalar MAC / dot / matvec / matmul the
byte-faithful INTEGER ``ref_interpret`` cannot run (IMM/LEA mask to 0xFF, SI/SC
store one byte, AX is a 32-bit INTEGER; no fp32 register, no FMUL/FADD).

This proves the ``ablate_matmul_buckets.part_b`` analytic count REAL: a runnable
native-fp32 VM (``c4_min.native_fp32_vm``) executes an fp32 dot/matmul VALUE-EXACT
vs numpy.float32 (tol ~1e-5 — value-faithful, NOT byte-exact-integer, which is the
right precision for an fp32 model) at a MEASURED, constant steps/MAC (K-sweep
slope): LOOP 18 / TIGHT 8 / TIGHT_REG 4 — bracketing the counted STACK 12 / TIGHT
7, with the register-allocated FMACC form BELOW 7.

CPU-only, no neural build. All checks run unconditionally (fast).

Run:  python -m pytest c4_min/selfhost/test_native_fp32_mac.py -x -q
"""
from __future__ import annotations

import numpy as np
import pytest

from c4_min.native_fp32_vm import (
    KERNELS,
    dot_mem,
    f32,
    native_fp32_interpret,
    run_dot,
    run_matmul,
    run_matvec,
)

MODES = ("loop", "tight", "tight_reg")


def _seq_dot_fp32(a, b) -> float:
    """The VM's exact accumulate order: product folded into a running fp32 sum."""
    acc = np.float32(0.0)
    for i in range(len(a)):
        acc = np.float32(acc + np.float32(np.float32(a[i]) * np.float32(b[i])))
    return float(acc)


# --------------------------------------------------------------------------- #
# value-exactness                                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", MODES)
def test_dot_value_exact_vs_numpy_fp32(mode):
    rng = np.random.default_rng(0)
    for K in range(1, 33):
        a = rng.standard_normal(K).astype(np.float32)
        b = rng.standard_normal(K).astype(np.float32)
        r, _ = run_dot(a.tolist(), b.tolist(), mode=mode)
        # EXACT vs the VM's own accumulate order (fp32), and within tol vs numpy.
        assert r == _seq_dot_fp32(a, b), (mode, K)
        assert abs(r - float(np.dot(a, b))) < 1e-4 * (1 + abs(float(np.dot(a, b))))


def test_dot_small_known_value():
    r, _ = run_dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    assert r == 32.0            # 4 + 10 + 18


@pytest.mark.parametrize("mode", MODES)
def test_matvec_value_exact(mode):
    rng = np.random.default_rng(1)
    M, K = 6, 9
    mat = rng.standard_normal((M, K)).astype(np.float32)
    vec = rng.standard_normal(K).astype(np.float32)
    y, _ = run_matvec(mat.tolist(), vec.tolist(), mode=mode)
    ref = mat @ vec
    for i in range(M):
        assert abs(y[i] - float(ref[i])) < 1e-4 * (1 + abs(float(ref[i])))


@pytest.mark.parametrize("mode", MODES)
def test_matmul_value_exact(mode):
    rng = np.random.default_rng(2)
    M, K, N = 5, 7, 4
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    C, _ = run_matmul(A.tolist(), B.tolist(), mode=mode)
    ref = A @ B
    for i in range(M):
        for j in range(N):
            assert abs(C[i][j] - float(ref[i, j])) < 1e-4 * (1 + abs(float(ref[i, j])))


def test_fp32_not_double_precision():
    """The VM holds fp32, not fp64 — a value with no exact fp32 rep is rounded."""
    r, _ = run_dot([0.1], [1.0])
    assert r == f32(0.1) != 0.1     # 0.1 is not exactly representable in fp32


# --------------------------------------------------------------------------- #
# MEASURED steps/MAC (the load-bearing result): a CONSTANT K-sweep slope       #
# --------------------------------------------------------------------------- #
def _slope(mode):
    """K-sweep marginal steps/MAC; assert value-exact + constant increment."""
    rng = np.random.default_rng(3)
    incs, prev = set(), None
    for K in range(1, 17):
        a = rng.standard_normal(K).astype(np.float32).tolist()
        b = rng.standard_normal(K).astype(np.float32).tolist()
        out = []
        _tr, steps, _h = native_fp32_interpret(KERNELS[mode](K), mem=dot_mem(a, b), out=out)
        assert abs(out[0] - _seq_dot_fp32(a, b)) < 1e-4, (mode, K)
        if prev is not None:
            incs.add(steps - prev)
        prev = steps
    assert len(incs) == 1, f"{mode} non-constant slope {incs}"
    return incs.pop()


def test_measured_steps_per_mac():
    """MEASURED (not counted) marginal steps/MAC — the headline result."""
    assert _slope("loop") == 18          # STACK / portable loop (counted ~12)
    assert _slope("tight") == 8          # unrolled, memory acc (counted ~7)
    assert _slope("tight_reg") == 4      # register-allocated FMACC (< counted 7)


def test_measured_rate_beats_draft_integer_vm():
    """Every native-fp32 rate is far below the draft integer VM's 101/MAC."""
    for mode in MODES:
        assert _slope(mode) < 101 / 4     # >4x fewer steps even for the loop form
    assert 101 / _slope("tight_reg") > 20  # register-allocated: >20x fewer steps
