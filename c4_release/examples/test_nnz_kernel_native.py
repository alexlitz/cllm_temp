"""Pytest for the NATIVIZED compiled-nnz kernel byte-exact OUTPUT block.

The kernel's `__main__` §[2]-[3] block asserts the nativized native-C integer
kernel produces the BYTE-IDENTICAL per-op OUTPUT (32-bit result + high MUL
word) as (a) the reference ArithCell/BitwiseCell forward and (b) the faithful
fp128-powl / fp64-divmod C kernel, across a mixed battery of all 17 opcodes
with full 32-bit operands + boundary edges. This wraps exactly that OUTPUT
verification (skipping the throughput/multithread timing) as a hard pytest.

Requires gcc (the kernels are gcc-compiled from embedded C). CPU-safe:
no model load, small battery, RSS well under 4 GB.
"""
from __future__ import annotations

import shutil
import tempfile

import numpy as np
import pytest

from examples import clever_compiled_nnz_kernel_native as K

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None,
                                reason="gcc required to compile the native kernel")

BATTERY_N = 20000
SEED = 20260817


def _build_and_run():
    workdir = tempfile.mkdtemp(prefix="test_nnz_nat_")
    try:
        lib_f, _, _ = K.build_kernel(workdir, K.C_SOURCE_FAITHFUL, "faithful")
        lib_n, _, _ = K.build_kernel(workdir, K.C_SOURCE_NATIVE, "native")
        rng = np.random.default_rng(SEED)
        ops, a, b, n = K.make_battery(BATTERY_N, rng)
        kres_n, khi_n = K.run_batch(lib_n, ops, a, b, n)
        kres_f, khi_f = K.run_batch(lib_f, ops, a, b, n)
        rres, rhi = K.reference_forward(ops, a, b, n)
        return ops, kres_n, khi_n, kres_f, khi_f, rres, rhi
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_nativized_output_byte_exact_vs_reference_and_faithful():
    ops, kres_n, khi_n, kres_f, khi_f, rres, rhi = _build_and_run()

    match_ref = (kres_n == rres) & (khi_n == rhi)
    match_faith = (kres_n == kres_f) & (khi_n == khi_f)
    n_ok_ref = int(match_ref.sum())
    n_ok_faith = int(match_faith.sum())

    assert n_ok_ref == BATTERY_N, (
        f"nativized vs REFERENCE cells: {n_ok_ref}/{BATTERY_N} exact "
        f"(first bad steps: {np.where(~match_ref)[0][:10].tolist()})"
    )
    assert n_ok_faith == BATTERY_N, (
        f"nativized vs FAITHFUL kernel: {n_ok_faith}/{BATTERY_N} exact "
        f"(first bad steps: {np.where(~match_faith)[0][:10].tolist()})"
    )

    # Per-op: every opcode present must be 100% exact.
    for op in sorted(K.OP_NAMES):
        idx = np.where(ops == op)[0]
        if idx.size == 0:
            continue
        ok = int(((kres_n[idx] == rres[idx]) & (khi_n[idx] == rhi[idx])).sum())
        assert ok == idx.size, f"op {K.OP_NAMES[op]}: {ok}/{idx.size} exact"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
