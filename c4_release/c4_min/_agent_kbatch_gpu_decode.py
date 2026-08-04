"""BYTE-EXACT + TIMING for the DIV-free doom pf_kbatch path (build_compact_sparse
_streaming + KBatchBoundedRunner per-row forward, where 0.034 ms/step was measured).

Proves C4_GPU_VERIFY vectorized span-decode == the per-row host decode on the battery
+ deep nested loop, then times whole drive_kbatch (forward+decode) per-row: host-loop
decode vs GPU-vectorized decode, at K in {512, 2048, 8192}.

Run:
    C4_PF_CFM=1 OMP_NUM_THREADS=4 python -m c4_min._agent_kbatch_gpu_decode \
        --device cuda:0 --K 512,2048,8192
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("C4_POS_SPARSE", "1")

import torch

from c4_min import isa
from c4_min.pf_kbatch import KBatchBoundedRunner
from c4_min.pf_speculative import draft_pf_program, set_gpu_verify
import c4_min.bench_pf_kbatch as KB
from c4_min.bench_composed_fast_path import _battery, _nested_prog


def _mem_avail_gb():
    with open("/proc/meminfo") as f:
        for ln in f:
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) / 1e6
    return 1e9


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--K", default="512,2048,8192")
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--outer", type=int, default=30)
    ap.add_argument("--inner", type=int, default=40)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args(argv)
    if _mem_avail_gb() < 25.0:
        raise SystemExit("[GUARD] <25GB -> STOP")

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    KB.runner_window = args.window
    from c4_min.compact_alloc import build_compact_sparse_streaming
    t0 = time.time()
    model, L, _ = build_compact_sparse_streaming(
        code_size=64, compute_mode="dense_kernel")
    if device != "cpu":
        model.to(device)
        model.materialize_dense(device)
    runner = KBatchBoundedRunner(model, L, window=args.window)
    print(f"[built] n_blocks={len(model.blocks)} dim={model.embed.shape[1]} "
          f"dev={device} build={time.time()-t0:.1f}s", flush=True)

    # ---- BYTE-EXACT: gpu-decode == per-row decode, battery + deep loop ----------
    print(f"\n{'='*70}\n[byte-exact] gpu-vectorized decode == per-row host decode"
          f"\n{'='*70}", flush=True)
    all_ok = True
    for name, prog, seed in _battery():
        code = prog if (prog and isinstance(prog[0], isa.Instr)) else isa.assemble(prog)
        set_gpu_verify(False)
        ref, _ = KB.drive_kbatch(model, L, runner, code, K=8, seed_mem=seed, perrow=True)
        set_gpu_verify(True)
        got, _ = KB.drive_kbatch(model, L, runner, code, K=8, seed_mem=seed, perrow=True)
        set_gpu_verify(None)
        ok = (ref == got)
        all_ok = all_ok and ok
        print(f"  {name:10s} {'OK ' if ok else 'FAIL'} steps={len(ref)}", flush=True)
        if not ok:
            print(f"    ref={ref}\n    gpu={got}", flush=True)
    deep = _nested_prog(3, 4)
    set_gpu_verify(False)
    dref, _ = KB.drive_kbatch(model, L, runner, deep, K=16, seed_mem={},
                              max_steps=60, perrow=True)
    set_gpu_verify(True)
    dgot, _ = KB.drive_kbatch(model, L, runner, deep, K=16, seed_mem={},
                              max_steps=60, perrow=True)
    set_gpu_verify(None)
    dok = (dref == dgot)
    all_ok = all_ok and dok
    print(f"  {'nested_deep':10s} {'OK ' if dok else 'FAIL'} steps={len(dref)}",
          flush=True)
    print(f"  => {'ALL BYTE-EXACT' if all_ok else 'DIVERGENCE'}", flush=True)
    if not all_ok:
        return 1

    # ---- TIMING: whole drive_kbatch (forward+decode) per-row, host vs gpu decode -
    deep = KB._nested_prog if False else _nested_prog(args.outer, args.inner)
    draft = draft_pf_program(deep, max_steps=1_000_000, mask=0xFF)
    n_steps = draft.step_count
    print(f"\n{'='*70}\n[timing] drive_kbatch per-row: host-loop decode vs GPU decode "
          f"({n_steps} steps)\n{'='*70}", flush=True)
    print(f"  {'K':>6} {'decode':>8} {'ms/step':>10} {'fwds':>6}", flush=True)
    print("  " + "-" * 36, flush=True)
    cuda = device.startswith("cuda")
    for K in [int(k) for k in args.K.split(",") if k.strip()]:
        for mode, gpu in (("host", False), ("gpu", True)):
            set_gpu_verify(gpu)
            KB.drive_kbatch(model, L, runner, deep, K=K, seed_mem={},
                            max_steps=n_steps + 10, perrow=True)   # warmup
            if cuda:
                torch.cuda.synchronize()
            best = 1e18
            nf = 0
            for _ in range(args.reps):
                t0 = time.time()
                tr, nf = KB.drive_kbatch(model, L, runner, deep, K=K, seed_mem={},
                                         max_steps=n_steps + 10, perrow=True)
                if cuda:
                    torch.cuda.synchronize()
                best = min(best, time.time() - t0)
            set_gpu_verify(None)
            print(f"  {K:>6} {mode:>8} {best/n_steps*1e3:>10.4f} {nf:>6}", flush=True)
        print("  " + "-" * 36, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
