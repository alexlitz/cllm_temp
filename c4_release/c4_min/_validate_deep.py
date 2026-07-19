#!/usr/bin/env python3
"""Deep-loop byte-identity validation on CPU: pick the SHORTEST deep instance per
deep cluster (tractable on CPU) + one moderate loop_sum, and prove
spec(fast) == spec(slow) == token-by-token driver, incl the deep-recursion fix
(EFF=500000, SP_INIT=0xFC).  These are the exact clusters fix-deep-recursion-cam
targets."""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "4")

import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
_PF.SP_INIT = 0xFC
_PFC.SP_INIT = 0xFC

import torch
from c4_min.compact_alloc import build_compact_pure_forward_model
from c4_min.sparse_forward import SparseTransformer
from c4_min.pf_speculative import speculative_run, spotcheck_vs_cached
from c4_min.run_corpus_stacked import bytecode_to_isa, cluster_of
from src.compiler import compile_c
from tests.test_suite_1000 import generate_test_programs

# (idx) shortest deep instance per cluster + a moderate loop_sum (462=115 steps).
IDS = [940, 700, 765, 733, 484, 462]   # gcd, rec_factorial, rec_sum, rec_fib, loop_countdown, loop_sum


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    t = time.monotonic()
    base, L, cs = build_compact_pure_forward_model(
        code_size=64, include_bitwise=False, include_divmod=True)
    sparse = SparseTransformer(base, compute_mode="sparse_mm")
    del base
    sparse = sparse.to(device)
    print(f"[deep] built in {time.monotonic()-t:.0f}s on {device}", flush=True)

    all_tests = generate_test_programs()
    ok = True
    for idx in IDS:
        s, e, d = all_tests[idx]
        code = bytecode_to_isa(compile_c(s)[0])
        t0 = time.monotonic()
        r = speculative_run(sparse, L, code, e, block_steps=48, device=device,
                            evict=True, fast=True)
        dt = time.monotonic() - t0
        # byte-identity vs token-by-token driver (evict=False for exact AR compare).
        sc = spotcheck_vs_cached(sparse, L, code, max_steps=r.step_count + 8,
                                 device=device, evict=False, block_steps=48)
        good = (r.status == "PASS" and sc["identical"] and sc["fast_matches_slow"])
        ok = ok and good
        print(f"  idx={idx:4d} [{cluster_of(d):16s}] {r.status:7s} got={r.decoded_final_ax} "
              f"exp={e} steps={r.step_count} fwds={r.forwards} spd={r.speedup:.1f}x "
              f"| identical={sc['identical']} fast==slow={sc['fast_matches_slow']} "
              f"| {dt:.0f}s  {'OK' if good else 'BAD<--'}", flush=True)
    print(f"\n=== DEEP VALIDATION {'PASS' if ok else 'FAIL'} ===", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
