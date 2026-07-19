#!/usr/bin/env python3
"""Validate the stacked runner: builds ONCE, then checks
  (a) batch=1 speculative (fast overlay) PASSes a stratified set incl deep loops,
  (b) byte-identity vs the token-by-token KV-cached driver (spotcheck),
  (c) fast overlay == slow overlay verdict,
  (d) cross-program batch (batch>1) agrees with batch=1 per-program verdicts.
"""
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
from c4_min.batched_speculative import speculative_run_batch
from c4_min.run_corpus_stacked import bytecode_to_isa, cluster_of
from src.compiler import compile_c
from tests.test_suite_1000 import generate_test_programs


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    compute = sys.argv[2] if len(sys.argv) > 2 else "sparse_mm"
    clusters = (sys.argv[3].split(",") if len(sys.argv) > 3
                else ["add", "sub", "mul", "div", "mod", "var_simple", "if_gt",
                      "func_identity", "loop_sum", "gcd", "rec_fib", "rec_sum"])
    per = int(sys.argv[4]) if len(sys.argv) > 4 else 2

    t = time.monotonic()
    base, L, cs = build_compact_pure_forward_model(
        code_size=64)
    sparse = SparseTransformer(base, compute_mode=compute)
    del base
    sparse = sparse.to(device)
    print(f"[val] built {compute} model in {time.monotonic()-t:.0f}s on {device}",
          file=sys.stderr, flush=True)

    all_tests = generate_test_programs()
    from collections import Counter
    seen = Counter()
    picks = []
    for i, (s, e, d) in enumerate(all_tests):
        cl = cluster_of(d)
        if cl in clusters and seen[cl] < per:
            picks.append(dict(idx=i, cluster=cl, description=d, expected=e,
                              code=bytecode_to_isa(compile_c(s)[0]), source=s))
            seen[cl] += 1

    # (a) batch=1 correctness + speedup
    print("\n=== (a) batch=1 speculative (fast overlay) ===")
    b1_verdict = {}
    for p in picks:
        r = speculative_run(sparse, L, p["code"], p["expected"], block_steps=64,
                            device=device, evict=True, fast=True)
        b1_verdict[p["idx"]] = r.status
        print(f"  idx={p['idx']:4d} [{p['cluster']:14s}] {r.status:7s} "
              f"got={r.decoded_final_ax} exp={r.expected} steps={r.step_count} "
              f"fwds={r.forwards} speedup={r.speedup:.1f}x")
    npass = sum(1 for v in b1_verdict.values() if v == "PASS")
    print(f"  batch=1: {npass}/{len(picks)} PASS")

    # (b)+(c) byte-identity spotcheck vs token-by-token driver (shortest 4).
    print("\n=== (b,c) byte-identity vs token-by-token driver + fast==slow ===")
    short = sorted([p for p in picks if b1_verdict[p["idx"]] == "PASS"],
                   key=lambda p: len(p["code"]))[:4]
    all_id = True
    for p in short:
        sc = spotcheck_vs_cached(sparse, L, p["code"], max_steps=6000,
                                 device=device, evict=False, block_steps=64)
        all_id = all_id and sc["identical"] and sc["fast_matches_slow"]
        print(f"  idx={p['idx']:4d} [{p['cluster']:14s}] identical={sc['identical']} "
              f"fast==slow={sc['fast_matches_slow']} "
              f"drv={sc['driver_final_ax']} spec={sc['spec_final_ax']} "
              f"steps={sc['n_steps_driver']}")
    print(f"  byte-identity: {'ALL MATCH' if all_id else 'MISMATCH!'}")

    # (d) cross-program batch vs batch=1 verdict agreement.
    print("\n=== (d) cross-program batch (batch=6) vs batch=1 ===")
    rows, sn, ss = speculative_run_batch(sparse, L, picks, block_steps=64,
                                         device=device, evict=True, fast=True,
                                         batch_cap=6)
    agree = 0
    for r in rows:
        b1 = b1_verdict.get(r["idx"])
        match = (b1 == r["status"])
        agree += int(match)
        flag = "" if match else "  <-- DISAGREE"
        print(f"  idx={r['idx']:4d} [{r['cluster']:14s}] batch={r['status']:7s} "
              f"b1={b1}{flag}")
    print(f"  batch agrees with batch=1: {agree}/{len(rows)}")
    print(f"  batch forward-count speedup: {sn} -> {ss} = {sn/ss:.1f}x")

    ok = (npass == len(picks) and all_id and agree == len(rows))
    print(f"\n=== VALIDATION {'PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
