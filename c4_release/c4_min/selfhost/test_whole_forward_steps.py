"""Whole-forward grounded-step-count checks + the byte-exact argmax gate.

Asserts:
  1. every per-op draft-VM rate is a positive constant (the measured self-host
     cost of that op);
  2. the tiny c4vm.onnx whole-forward grounded step count is dominated by matmul
     but the non-matmul portion is a real, measured fraction (not extrapolated);
  3. the reference argmax on [1,2,42,3] is [0,1,20,3] (the byte-exact target that
     any c4-hosted run must reproduce) — via nbl_bin_interp.Graph.run itself.

CPU-only.  OMP_NUM_THREADS=4 python -m pytest \
    c4_min/selfhost/test_whole_forward_steps.py -x -q
"""
from __future__ import annotations

import tempfile
import os

import numpy as np


def _tiny_argmax():
    from c4_min import blogspec_compiler as C, export_onnx as E
    from c4_min.onnx_to_c4bin import lower_onnx_to_bin
    from c4_min.nbl_bin_interp import Graph

    d = tempfile.mkdtemp(prefix="tiny_argmax_")
    onnxp = os.path.join(d, "m.onnx")
    binp = os.path.join(d, "m.nblbin")
    model, L, _ = C.build_step_model(E.PROOF_PROG)
    model.eval()
    E.export_onnx(model, onnxp)
    lower_onnx_to_bin(onnxp, binp)
    out = Graph(binp).run(np.array([[1, 2, 42, 3]], dtype=np.int64))
    return np.argmax(out[0], axis=-1).tolist()


def test_reference_argmax_byte_exact():
    """The byte-exact target: nbl_bin_interp on [1,2,42,3] -> [0,1,20,3]."""
    assert _tiny_argmax() == [0, 1, 20, 3]


def test_per_op_rates_positive_constant():
    from c4_min.selfhost.measure_whole_forward_steps import measure_rates
    rates = measure_rates(verbose=False)
    for k, v in rates.items():
        assert v > 0, f"rate {k} not positive: {v}"
    # matmul per-MAC and per-output are the documented ~101 / ~205
    assert rates["_matmul_per_mac"] == 101
    assert rates["_matmul_per_out"] == 205


def test_whole_forward_grounded_dominated_by_matmul():
    from c4_min.selfhost.measure_whole_forward_steps import (
        measure_rates, ground_whole_forward)
    rates = measure_rates(verbose=False)
    info = ground_whole_forward(rates, verbose=False)
    assert info["argmax"] == [0, 1, 20, 3]
    assert info["n_nodes"] == 153
    # matmul dominates but non-matmul is a real measured fraction (a few %)
    assert info["matmul"] > info["nonmatmul"]
    assert info["nonmatmul"] > 0
    # whole-forward total is a bit above the matmul-only ~48-52M
    assert info["total"] > info["matmul"]
    assert 4e7 < info["total"] < 1e8


if __name__ == "__main__":
    import sys
    checks = [
        ("reference argmax [0,1,20,3]", test_reference_argmax_byte_exact),
        ("per-op rates positive/constant", test_per_op_rates_positive_constant),
        ("whole-forward grounded", test_whole_forward_grounded_dominated_by_matmul),
    ]
    passed = 0
    for name, fn in checks:
        try:
            fn()
            print(f"  OK   {name}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {name}: {e}")
    print(f"\n{passed}/{len(checks)} whole-forward checks pass")
    sys.exit(0 if passed == len(checks) else 1)
