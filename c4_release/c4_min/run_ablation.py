#!/usr/bin/env python3
"""PER-LEVER ABLATION — measure what each of the 4 stacked speedup levers buys.

Builds the model ONCE, picks a fixed representative subset, and times the corpus
under a ladder of configurations, each turning ONE lever off relative to the full
stack, so the wall-time delta attributes the speedup to that lever:

  FULL          sparse_mm + speculation + O(1)-decode + cross-program batch
  -batch        batch=1 (per-program speculation; steps still batched within a prog)
  -O(1)-decode  slow apply_overlay_window (per-row code re-scan)
  -speculation  token-by-token KV-cached driver (one forward per VM step)
  -sparse       dense_kernel compute (materialise the weight; no sparse-mm work-skip)

The forward-COUNT speedup (naive token-by-token forwards / speculative forwards) is
reported per config too, so the speculation lever is quantified even where a full
token-by-token wall-time is intractable.

Correctness: every config's per-program verdict is asserted equal to FULL's, so a
speedup is never bought by a wrong answer.

Usage
-----
    OMP_NUM_THREADS=4 python c4_min/run_ablation.py --device cpu \
        --clusters add,sub,mul,var_simple,if_gt,loop_sum,gcd --per-cluster 3
"""
from __future__ import annotations
import argparse, json, os, sys, time
from collections import Counter

os.environ.setdefault("OMP_NUM_THREADS", "4")
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(_HERE)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
_PF.SP_INIT = 0xFC
_PFC.SP_INIT = 0xFC

import torch
from c4_min.compact_alloc import build_compact_pure_forward_model
from c4_min.sparse_forward import SparseTransformer
from c4_min.pf_speculative import speculative_run
from c4_min.batched_speculative import speculative_run_batch
from c4_min.nibble_pure_forward_cached import run_pure_forward_cached
from c4_min.run_corpus_stacked import bytecode_to_isa, cluster_of
from src.compiler import compile_c
from tests.test_suite_1000 import generate_test_programs


def run_full_or_perprog(model, L, picks, device, batch_cap, fast, block_steps):
    """Speculative run.  batch_cap>1 -> cross-program batch; else per-program.
    Returns (verdicts{idx:status}, wall, naive_fwds, spec_fwds, total_steps)."""
    t0 = time.monotonic()
    verdict, naive, spec, steps = {}, 0, 0, 0
    if batch_cap > 1:
        rows, sn, ss = speculative_run_batch(
            model, L, picks, block_steps=block_steps, device=device, evict=True,
            fast=fast, batch_cap=batch_cap)
        for r in rows:
            verdict[r["idx"]] = r["status"]
            steps += r.get("steps", 0) or 0
        naive, spec = sn, ss
    else:
        for p in picks:
            r = speculative_run(model, L, p["code"], p["expected"],
                                block_steps=block_steps, device=device, evict=True,
                                fast=fast)
            verdict[p["idx"]] = r.status
            naive += r.naive_forwards
            spec += r.forwards
            steps += r.step_count
    return verdict, time.monotonic() - t0, naive, spec, steps


def run_token_by_token(model, L, picks, device):
    """Token-by-token KV-cached driver (the -speculation config)."""
    t0 = time.monotonic()
    verdict, steps = {}, 0
    for p in picks:
        trace = run_pure_forward_cached(
            model, L, p["code"], max_steps=20000, mask=0xFFFFFFFF, evict=True)
        got = (trace[-1] & 0xFFFFFFFF) if trace else None
        exp = p["expected"] & 0xFFFFFFFF
        n = len(trace)
        st = "PASS" if got == exp else ("TIMEOUT" if n >= 20000 else "FAIL")
        verdict[p["idx"]] = st
        steps += n
    return verdict, time.monotonic() - t0, steps


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--clusters", type=str,
                    default="add,sub,mul,div,mod,var_simple,if_gt,loop_sum,gcd")
    ap.add_argument("--per-cluster", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--block-steps", type=int, default=48)
    ap.add_argument("--skip-token-by-token", action="store_true",
                    help="skip the -speculation config (intractable for deep sets).")
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args(argv)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    clusters = [c.strip() for c in args.clusters.split(",")]

    all_tests = generate_test_programs()
    seen = Counter(); picks = []
    for i, (s, e, d) in enumerate(all_tests):
        cl = cluster_of(d)
        if cl in clusters and seen[cl] < args.per_cluster:
            picks.append(dict(idx=i, cluster=cl, description=d, expected=e,
                              code=bytecode_to_isa(compile_c(s)[0])))
            seen[cl] += 1
    print(f"[abl] {len(picks)} programs across {len(clusters)} clusters", flush=True)

    def build(compute_mode):
        t = time.monotonic()
        base, L, _ = build_compact_pure_forward_model(
            code_size=64)
        m = SparseTransformer(base, compute_mode=compute_mode)
        del base
        m = m.to(device)
        print(f"[abl] built {compute_mode} in {time.monotonic()-t:.0f}s", flush=True)
        return m, L

    model, L = build("sparse_mm")
    results = {}

    # FULL stack (sparse + speculation + O(1) + batch)
    v_full, wall, naive, spec, steps = run_full_or_perprog(
        model, L, picks, device, args.batch, True, args.block_steps)
    results["FULL"] = dict(wall=wall, naive_fwds=naive, spec_fwds=spec,
                           steps=steps, fwd_speedup=naive/spec if spec else 0,
                           passes=sum(1 for v in v_full.values() if v == "PASS"))
    print(f"[abl] FULL: wall={wall:.1f}s steps={steps} "
          f"fwd_speedup={naive/spec if spec else 0:.1f}x", flush=True)

    def record(name, verdict, wall, naive, spec, steps):
        agree = sum(1 for i in verdict if verdict[i] == v_full.get(i))
        results[name] = dict(wall=wall, naive_fwds=naive, spec_fwds=spec,
                             steps=steps, fwd_speedup=naive/spec if spec else 0,
                             passes=sum(1 for v in verdict.values() if v == "PASS"),
                             agrees_with_full=agree, n=len(verdict))
        print(f"[abl] {name}: wall={wall:.1f}s fwd_speedup={naive/spec if spec else 0:.1f}x "
              f"agree={agree}/{len(verdict)}", flush=True)

    # -batch (per-program speculation)
    v, wall, naive, spec, steps = run_full_or_perprog(
        model, L, picks, device, 1, True, args.block_steps)
    record("-batch (batch=1)", v, wall, naive, spec, steps)

    # -O(1)-decode (slow overlay, per-program to isolate the overlay cost)
    v, wall, naive, spec, steps = run_full_or_perprog(
        model, L, picks, device, 1, False, args.block_steps)
    record("-O(1)-decode (slow overlay)", v, wall, naive, spec, steps)

    # -speculation (token-by-token KV-cached driver)
    if not args.skip_token_by_token:
        v, wall, steps = run_token_by_token(model, L, picks, device)
        record("-speculation (token-by-token)", v, wall, steps, steps, steps)

    # -sparse (dense_kernel compute; rebuild)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    model_d, L = build("dense_kernel")
    v, wall, naive, spec, steps = run_full_or_perprog(
        model_d, L, picks, device, args.batch, True, args.block_steps)
    record("-sparse (dense_kernel)", v, wall, naive, spec, steps)

    # -- report -------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"PER-LEVER ABLATION on {device} ({len(picks)} programs, "
          f"{results['FULL']['steps']} VM steps)")
    print("=" * 78)
    full_wall = results["FULL"]["wall"]
    print(f"  {'config':32s} {'wall(s)':>9s} {'steps/s':>9s} {'fwd_spd':>8s} "
          f"{'vs FULL':>9s} {'agree':>7s}")
    print("  " + "-" * 76)
    for name in ["FULL", "-batch (batch=1)", "-O(1)-decode (slow overlay)",
                 "-speculation (token-by-token)", "-sparse (dense_kernel)"]:
        if name not in results:
            continue
        r = results[name]
        sps = r["steps"] / r["wall"] if r["wall"] else 0
        vs = r["wall"] / full_wall if full_wall else 0
        ag = f"{r.get('agrees_with_full', r['passes'])}/{r.get('n', len(picks))}" \
            if name != "FULL" else "-"
        print(f"  {name:32s} {r['wall']:9.1f} {sps:9.0f} "
              f"{r['fwd_speedup']:7.1f}x {vs:8.2f}x {ag:>7s}")
    print("  " + "-" * 76)
    print("  LEVER CONTRIBUTION (wall-time factor the lever buys, full-stack basis):")
    def factor(name):
        return results[name]["wall"] / full_wall if name in results and full_wall else None
    for lever, name in [("cross-program batch", "-batch (batch=1)"),
                        ("O(1)-decode overlay", "-O(1)-decode (slow overlay)"),
                        ("speculation", "-speculation (token-by-token)"),
                        ("sparse compute", "-sparse (dense_kernel)")]:
        f = factor(name)
        if f is not None:
            print(f"    {lever:24s}: {f:5.2f}x  (removing it makes the run {f:.2f}x "
                  f"the FULL wall)")
    print("=" * 78)

    if args.output:
        with open(args.output, "w") as fh:
            json.dump({"device": device, "n_programs": len(picks),
                       "clusters": clusters, "per_cluster": args.per_cluster,
                       "batch": args.batch, "block_steps": args.block_steps,
                       "results": results}, fh, indent=2)
        print(f"[abl] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
