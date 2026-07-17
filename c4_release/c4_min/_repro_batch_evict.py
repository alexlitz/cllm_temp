#!/usr/bin/env python3
"""Reproduce + validate the run_batch_gpu --evict deep-loop memory recall.

Runs the 6 block-verify SANITY deep loops (loop_sum 450/451, gcd 900/901,
rec_fib 725/726) through run_batch_gpu with evict ON and prints got vs expected.
Ground truth (block-verify): 171/153/2/4/1/2.
"""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "4")
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(_HERE)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import torch
import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
_PF.SP_INIT = 0xF0
_PFC.SP_INIT = 0xF0

from c4_min.compact_alloc import build_compact_pure_forward_model
from c4_min.sparse_forward import SparseTransformer
from c4_min.nibble_pure_forward_gpu import run_batch_gpu
from c4_min import isa
from src.compiler import compile_c
from tests.test_suite_1000 import generate_test_programs

# local bytecode->isa (avoid importing run_1096_pure_forward, which blanks
# CUDA_VISIBLE_DEVICES at import time).
_WORD = 8
_SLOT_SCALED_OPS = frozenset({isa.LEA, isa.ENT, isa.ADJ})


def _sign32(imm):
    return imm if imm < (1 << 31) else imm - (1 << 32)


def bytecode_to_isa(bytecode):
    out = []
    for word in bytecode:
        op = int(word) & 0xFF
        imm = int(word) >> 8
        if op in _SLOT_SCALED_OPS:
            out.append(isa.Instr(op, _sign32(imm) // _WORD))
        else:
            out.append(isa.Instr(op, imm & 0xFFFFFFFF))
    return out

SANITY = {450: 171, 451: 153, 900: 2, 901: 4, 725: 1, 726: 2}


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    if dev.startswith("cuda"):
        # initialise the CUDA context BEFORE the long CPU-only model build so a
        # late .to(dev) doesn't hit a reclaimed/uninitialised context.
        torch.zeros(1).to(dev)
    evict = "--no-evict" not in sys.argv
    prune_interval = 120
    for a in sys.argv:
        if a.startswith("--prune="):
            prune_interval = int(a.split("=")[1])
    t0 = time.monotonic()
    print(f"[repro] building compact+sparse model on {dev} ...", flush=True)
    base, L, _cs = build_compact_pure_forward_model(
        code_size=64, include_bitwise=False, include_divmod=True)
    sparse = SparseTransformer(base, compute_mode="dense_kernel").to(dev)
    del base
    print(f"[repro] built in {time.monotonic()-t0:.0f}s; evict={evict} "
          f"prune_interval={prune_interval}", flush=True)

    tests = generate_test_programs()
    ids = sorted(SANITY)
    codes, caps, want = [], [], []
    for idx in ids:
        src, exp, desc = tests[idx]
        code = bytecode_to_isa(compile_c(src)[0])
        codes.append(code)
        caps.append(10000)
        want.append(SANITY[idx])
    stats = {}
    t1 = time.monotonic()
    traces = run_batch_gpu(sparse, L, codes, caps, device=dev,
                           mask=0xFFFFFFFF, evict=evict,
                           prune_interval=prune_interval, stats=stats)
    dt = time.monotonic() - t1
    n_ok = 0
    print(f"\n[repro] batch of {len(ids)} ran in {dt:.1f}s "
          f"fwds={stats.get('n_forwards')} maxcache={stats.get('max_cache_size')}")
    print(f"{'id':>5} {'exp':>6} {'got':>10} {'steps':>7}  ok")
    for idx, tr, exp in zip(ids, traces, want):
        got = int(tr[-1]) & 0xFFFFFFFF if tr else None
        ok = (got == exp)
        n_ok += ok
        print(f"{idx:5d} {exp:6d} {str(got):>10} {len(tr):7d}  {'PASS' if ok else 'FAIL'}")
    print(f"\n[repro] {n_ok}/{len(ids)} match block-verify ground truth")
    return 0 if n_ok == len(ids) else 1


if __name__ == "__main__":
    raise SystemExit(main())
