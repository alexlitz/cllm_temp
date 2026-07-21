"""The speculation FLOP-utilization sweep (``bench_spec_flop_sweep``).

Pins the load-bearing invariants of the sweep harness:

  * the long pure-ALU/branch accumulate program decodes BYTE-EXACT vs the
    full-length ``isa.interpret`` reference at EVERY block size K (the 100%
    acceptance the sweep asserts) — this is the correctness contract that makes
    the speedup numbers meaningful;
  * the weight bundle is the expected extreme sparsity (~0.003% dense), so the
    dense-vs-sparse FLOP split is real;
  * the analytic dense-FLOP model scales linearly in B and in n_layers.

CPU-safe (the compacted base model is 7 layers / dim 960).  A GPU-guarded test
runs the forward-isolated measurement to confirm util climbs with B.
"""
from __future__ import annotations

import pytest
import torch

from c4_min import isa
from c4_min import qwen_full_vm as Q
from c4_min import qwen_lean_forward as LF
from c4_min import bench_spec_flop_sweep as BS


DEVICE = ("cuda:0" if torch.cuda.is_available() else None)
requires_cuda = pytest.mark.skipif(DEVICE is None, reason="forward sweep needs a GPU")


@pytest.fixture(scope="module")
def vm_base():
    return Q.build(code_size=24, subset=Q.SUBSET_BASE)


@pytest.fixture(scope="module")
def lean_cpu(vm_base):
    return LF.LeanQwenVM.from_full_vm(vm_base, device="cpu")


# ---------------------------------------------------------------------------
# The accumulate program is pure-ALU/branch (no JSR/ENT/LEV -> no func fallback).
# ---------------------------------------------------------------------------
def test_accum_program_is_pure_alu_branch():
    ops = {op for op, _ in BS.ACCUM_PROG}
    forbidden = {"JSR", "ENT", "LEV", "SI", "SC", "LI", "LC"}
    assert not (ops & forbidden), f"program touches out-of-slice ops: {ops & forbidden}"
    assert ops <= {"IMM", "PSH", "ADD", "JMP"}


def test_accum_runs_forever_bounded_by_max_steps():
    code = isa.assemble(BS.ACCUM_PROG)
    for n in (100, 1000, 5000):
        ref = isa.interpret(code, max_steps=n)
        assert len(ref) == n, f"accumulate loop should fill exactly {n} steps"


# ---------------------------------------------------------------------------
# BYTE-EXACT at every K vs the FULL-length reference (the sweep's acceptance gate).
# ---------------------------------------------------------------------------
def test_byte_exact_across_K_on_cpu(lean_cpu):
    """The model decodes the accumulate loop byte-for-byte identically to
    ``isa.interpret`` at every block size K — the 100%-acceptance invariant the
    sweep reports.  (On CPU so it runs anywhere; a short trace keeps it fast.)"""
    code = isa.assemble(BS.ACCUM_PROG)
    n_steps = 400
    ref = isa.interpret(code, max_steps=n_steps)
    naive = LF.run_program_lean(lean_cpu, code, max_steps=n_steps)
    assert naive["ax_trace"] == ref, "naive lean driver diverged from the reference"
    for K in (1, 8, 32, 64, 128):
        r = LF.speculative_run_lean(lean_cpu, code, block_steps=K, max_steps=n_steps)
        assert r.ax_trace == ref, f"K={K}: model trace != full reference (not byte-exact)"
        # forwards saved is exactly ceil(steps / K).
        import math
        assert r.forwards == math.ceil(n_steps / K)


# ---------------------------------------------------------------------------
# Weight sparsity is the extreme value the whole premise rests on.
# ---------------------------------------------------------------------------
def test_weights_are_extremely_sparse(lean_cpu):
    nz, tot = BS._weight_density(lean_cpu)
    density = nz / tot
    assert density < 0.001, f"weights should be <0.1% dense, got {density*100:.4f}%"
    assert tot / nz > 1000, "dense/sparse ratio should be >1000x"


# ---------------------------------------------------------------------------
# The analytic dense-FLOP model is linear in B and in n_layers.
# ---------------------------------------------------------------------------
def test_dense_flop_model_scales_linearly_in_B(lean_cpu):
    f1 = BS._forward_dense_flops(lean_cpu, 1, 7)
    f10 = BS._forward_dense_flops(lean_cpu, 10, 7)
    # attention scores/context are O(B) too (per-row S*S), so total is exactly linear in B.
    assert f10 == 10 * f1


def test_dense_flop_model_positive_and_layer_scaled(lean_cpu):
    f = BS._forward_dense_flops(lean_cpu, 4, 7)
    assert f > 0
    # ~0.45 GFLOP per single (B=1, S=7) window on this 7L/dim-960 model, so a
    # (B=4, S=7) window is ~1.8 GFLOP — a broad sanity band.
    assert 1e8 < f < 1e10


# ---------------------------------------------------------------------------
# GPU: forward-isolated util climbs with B (the overhead-bound signal).
# ---------------------------------------------------------------------------
@requires_cuda
def test_forward_isolated_util_climbs_with_B():
    vm = Q.build(code_size=24, subset=Q.SUBSET_BASE)
    vm.embed = vm.embed.to(DEVICE)
    lean = LF.LeanQwenVM.from_full_vm(vm, device=DEVICE)
    nz, tot = BS._weight_density(lean)
    code = isa.assemble(BS.ACCUM_PROG)
    r1 = BS.measure_forward_isolated(lean, code, 1, device=DEVICE, nz=nz, tot=tot,
                                     peak_tflops=27.8, warmup=2, iters=5)
    rB = BS.measure_forward_isolated(lean, code, 1024, device=DEVICE, nz=nz, tot=tot,
                                     peak_tflops=27.8, warmup=2, iters=5)
    # per-step time drops dramatically and dense-util climbs (overhead amortised).
    assert rB["ms_per_step"] < r1["ms_per_step"]
    assert rB["dense_util_pct"] > r1["dense_util_pct"] * 10


# ---------------------------------------------------------------------------
# The end-to-end sweep() runs both passes, keeps the full-driver pass BOUNDED
# (byte-exact gate + Python-driver ceiling), and lets the forward-isolated pass
# carry the K-to-VRAM curve.  This pins that structure.
# ---------------------------------------------------------------------------
@requires_cuda
def test_sweep_end_to_end_bounded_driver_and_full_forward_curve():
    res = BS.sweep(device=DEVICE, n_steps=200,
                   ks=[1, 16, 64, 256, 1024],
                   driver_steps=120, driver_max_k=64,
                   warmup=1, iters=1)
    # full-driver pass is CAPPED at driver_max_k (K=256/1024 excluded).
    driver_ks = [r["K"] for r in res["rows"]]
    assert driver_ks == [1, 16, 64], driver_ks
    # every full-driver K is BYTE-EXACT (100%% acceptance — the whole point).
    assert all(r["byte_exact"] for r in res["rows"]),         "full-driver decode diverged from the reference at some K"
    # forward-isolated pass carries the FULL K list (VRAM permitting).
    fwd_bs = [r["B"] for r in res["forward_rows"]]
    assert fwd_bs[:5] == [1, 16, 64, 256, 1024], fwd_bs
    # and shows the overhead-bound speedup: bigger B is faster per step + more util.
    f1 = next(r for r in res["forward_rows"] if r["B"] == 1)
    f1024 = next(r for r in res["forward_rows"] if r["B"] == 1024)
    assert f1024["ms_per_step"] < f1["ms_per_step"]
    assert f1024["speedup"] > 10.0
    assert f1024["dense_util_pct"] > f1["dense_util_pct"] * 5
    # the sparse/dense FLOP split is the ~31,000x sparsity ratio.
    assert res["tot"] / res["nz"] > 1000
