#!/usr/bin/env python3
"""CHK-1 deep-loop closer — PERFECT-DRAFT SPECULATION on the SPARSE pure-forward VM.

The token-by-token pure-forward driver PASSes 725 / FAILs 0 / TIMEOUTs 104 of the
829 scored corpus (zero wrong answers).  The 104 TIMEOUTs are deep loops (gcd,
loop_sum, rec_factorial, loop_countdown) that HALT correctly (the reference VM +
the hybrid VM both reach the right answer) but are intractable one-forward-per-step
at ~5 steps/sec.

This runner closes them with BLOG_SPEC §Speculation, adapted to the pure-forward
state-machine model (``c4_min.pf_speculative``): the reference VM DRAFTS the whole
token stream (zero forwards), and the SPARSE model VERIFIES it in BLOCKS (a handful
of batched forwards) against the per-block KV cache + bounded eviction.  A program
PASSes iff the model ACCEPTS the full drafted stream (every step-query row's decoded
register state matches the draft — byte-for-byte what greedy token-by-token would
emit) AND the decoded final AX == expected.

Usage
-----
    OMP_NUM_THREADS=4 python c4_min/run_deeploop_speculative.py \
        --device cuda:0 --snapshot /tmp/chk1_snapshot.json \
        --output /tmp/spec_deeploop.json
    # narrow to one cluster / a limit for a quick check:
    OMP_NUM_THREADS=4 python c4_min/run_deeploop_speculative.py \
        --device cuda:0 --clusters loop_sum --limit 4 --spotcheck

The ``--snapshot`` JSON (the token-by-token scoreboard) supplies the 104 TIMEOUT
ids + the 725 PASSes; the runner re-runs ONLY the TIMEOUTs via speculation and
folds them into a FULL-1096 pure-forward pass fraction.  ``--spotcheck`` also runs
the token-by-token KV-cached driver on a sample and asserts byte-identity.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional

os.environ.setdefault("OMP_NUM_THREADS", "4")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(_HERE)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import re  # noqa: E402

import torch  # noqa: E402

import c4_min.nibble_pure_forward as _PF          # noqa: E402
import c4_min.nibble_pure_forward_complete as _PFC  # noqa: E402
_PF.SP_INIT = 0xF0
_PFC.SP_INIT = 0xF0

from c4_min import isa  # noqa: E402
from c4_min.compact_alloc import build_compact_pure_forward_model  # noqa: E402
from c4_min.nibble_pure_forward_complete import (  # noqa: E402
    build_pure_forward_complete_model, ref_interpret,
)
from c4_min.sparse_forward import SparseTransformer  # noqa: E402
from c4_min.pf_speculative import (  # noqa: E402
    speculative_run, draft_pf_program, spotcheck_vs_cached,
)

# `bytecode_to_isa` / `cluster_of` are inlined (NOT imported from
# run_1096_pure_forward) because that module's import-time
# `os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")` would blank the GPU for THIS
# GPU runner.  These are verbatim copies (frame-isomorphism byte re-encode).
_WORD = 8
_SLOT_SCALED_OPS = frozenset({isa.LEA, isa.ENT, isa.ADJ})


def _sign32(imm: int) -> int:
    return imm if imm < (1 << 31) else imm - (1 << 32)


def bytecode_to_isa(bytecode):
    out = []
    for word in bytecode:
        op = int(word) & 0xFF
        imm = int(word) >> 8
        if op in _SLOT_SCALED_OPS:
            simm = _sign32(imm)
            assert simm % _WORD == 0, f"unaligned {isa.NAMES.get(op)} imm {simm}"
            out.append(isa.Instr(op, simm // _WORD))
        else:
            out.append(isa.Instr(op, imm & 0xFFFFFFFF))
    return out


def cluster_of(description: str) -> str:
    base = description.split(":", 1)[0].strip()
    base = re.sub(r"_\d+$", "", base)
    base = re.sub(r"\d+$", "", base)
    base = base.rstrip("_")
    return base or "misc"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--snapshot", type=str, default=None,
                    help="token-by-token scoreboard JSON (supplies the 104 TIMEOUT "
                         "ids + the 725 PASSes for the full-1096 fold).")
    ap.add_argument("--clusters", type=str, default=None,
                    help="comma-list of clusters to run (default: all TIMEOUTs).")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--block-steps", type=int, default=48,
                    help="VM steps per batched verify forward (VRAM knob).")
    ap.add_argument("--max-steps", type=int, default=300000,
                    help="draft/step cap (the logical VM sizes the draft to the "
                         "exact halt step; this bounds a genuinely non-halting loop).")
    ap.add_argument("--no-divmod", action="store_true",
                    help="LEAN model (no DIV/MOD blocks). Default builds divmod "
                         "(gcd uses MOD) — matches the token-by-token snapshot.")
    ap.add_argument("--compute-mode", type=str, default="dense_kernel",
                    choices=["dense_kernel", "sparse_mm"])
    ap.add_argument("--no-evict", action="store_true")
    ap.add_argument("--prune-interval", type=int, default=120)
    ap.add_argument("--spotcheck", action="store_true",
                    help="ALSO run the token-by-token KV-cached driver on a sample "
                         "and assert byte-identity (proves speculation reproduces AR).")
    ap.add_argument("--spotcheck-n", type=int, default=3,
                    help="how many programs to byte-identity spot-check vs the driver.")
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--progress", type=int, default=10)
    args = ap.parse_args(argv)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[spec] CUDA not available; cpu", file=sys.stderr)
        device = "cpu"

    from src.compiler import compile_c
    from tests.test_suite_1000 import generate_test_programs
    all_tests = generate_test_programs()

    # -- pick the TIMEOUT ids from the snapshot (or all deep-loop clusters). ------
    timeout_ids: List[int] = []
    snap_summary = None
    snap_scored_total = 0
    if args.snapshot:
        snap = json.load(open(args.snapshot))
        snap_summary = snap.get("summary")
        # the scored-corpus size is a TOP-LEVEL snapshot key ("scored"/"total"),
        # NOT inside "summary" (which only holds per-status counts) — read it here.
        snap_scored_total = snap.get("scored") or snap.get("total") or 0
        timeout_ids = [r["idx"] for r in snap["results"] if r["status"] == "TIMEOUT"]
    else:
        # fall back: the four known deep-loop clusters
        deep = {"gcd", "loop_sum", "rec_factorial", "loop_countdown"}
        timeout_ids = [i for i, (s, e, d) in enumerate(all_tests)
                       if cluster_of(d) in deep]
    if args.clusters:
        keep = {c.strip() for c in args.clusters.split(",") if c.strip()}
        timeout_ids = [i for i in timeout_ids
                       if cluster_of(all_tests[i][2]) in keep]
    if args.limit is not None:
        timeout_ids = timeout_ids[:args.limit]

    include_divmod = not args.no_divmod
    t_build = time.monotonic()
    print(f"[spec:{device}] building pure-forward model (divmod={include_divmod}) ...",
          file=sys.stderr, flush=True)
    if include_divmod:
        dense_or_compact, L, cstats = build_compact_pure_forward_model(
            code_size=64, include_bitwise=False, include_divmod=True)
    else:
        dense_or_compact, L = build_pure_forward_complete_model(
            code_size=64, include_bitwise=False, include_divmod=False)
    sparse = SparseTransformer(dense_or_compact, compute_mode=args.compute_mode)
    st = sparse.stats()
    del dense_or_compact
    sparse = sparse.to(device)
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))
        vram_load = torch.cuda.memory_allocated(torch.device(device)) / 1e6
    else:
        vram_load = 0.0
    print(f"[spec:{device}] sparse model built in {time.monotonic()-t_build:.0f}s | "
          f"dim={sparse.dim} blocks={len(sparse.blocks)} "
          f"storage {st.sparse_mb:.1f}MB VRAM-load {vram_load:.0f}MB | "
          f"running {len(timeout_ids)} TIMEOUT programs via speculation",
          file=sys.stderr, flush=True)

    evict = not args.no_evict
    results: List[dict] = []
    per_cluster: Dict[str, Counter] = defaultdict(Counter)
    sum_naive = sum_spec = 0
    t0 = time.monotonic()
    for k, idx in enumerate(timeout_ids):
        source, expected, description = all_tests[idx]
        cluster = cluster_of(description)
        try:
            code = bytecode_to_isa(compile_c(source)[0])
        except Exception as exc:  # noqa: BLE001
            results.append(dict(idx=idx, cluster=cluster, status="ERROR",
                                detail=f"compile: {exc!r}"))
            per_cluster[cluster]["ERROR"] += 1
            continue
        try:
            r = speculative_run(sparse, L, code, expected,
                                block_steps=args.block_steps,
                                max_steps=args.max_steps, device=device,
                                evict=evict, prune_interval=args.prune_interval)
        except Exception as exc:  # noqa: BLE001
            results.append(dict(idx=idx, cluster=cluster, status="ERROR",
                                detail=f"spec: {exc!r}"))
            per_cluster[cluster]["ERROR"] += 1
            continue
        if device.startswith("cuda"):
            torch.cuda.synchronize(torch.device(device))
        per_cluster[cluster][r.status] += 1
        sum_naive += r.naive_forwards
        sum_spec += r.forwards
        results.append(dict(
            idx=idx, cluster=cluster, status=r.status, expected=r.expected,
            got=r.decoded_final_ax, steps=r.step_count, forwards=r.forwards,
            naive_forwards=r.naive_forwards, speedup=round(r.speedup, 2),
            accepted=r.accepted_steps, max_cache=r.max_cache_size,
            max_seq=r.max_seq_len, evicted=r.total_evicted,
            detail=r.detail, description=description))
        if args.progress and (k + 1) % args.progress == 0:
            npass = sum(1 for x in results if x["status"] == "PASS")
            print(f"[spec:{device}] {k+1}/{len(timeout_ids)} (pass {npass}) "
                  f"[{time.monotonic()-t0:.0f}s]", file=sys.stderr, flush=True)

    wall = time.monotonic() - t0
    counts = Counter(r["status"] for r in results)
    n_pass = counts.get("PASS", 0)
    total_spec = len(results)
    agg_speedup = (sum_naive / sum_spec) if sum_spec else float("inf")

    # -- byte-identity spot-check vs the token-by-token KV-cached driver ----------
    spotchecks = []
    if args.spotcheck:
        # spot-check the SHORTEST passing deep-loop programs (tractable for the
        # token-by-token driver) so byte-identity is proven affordably.
        passing = sorted([r for r in results if r["status"] == "PASS"],
                         key=lambda r: r["steps"])[:args.spotcheck_n]
        print(f"[spec:{device}] byte-identity spot-check vs token-by-token driver "
              f"on {len(passing)} programs ...", file=sys.stderr, flush=True)
        for r in passing:
            idx = r["idx"]
            source = all_tests[idx][0]
            code = bytecode_to_isa(compile_c(source)[0])
            sc = spotcheck_vs_cached(sparse, L, code,
                                     max_steps=r["steps"] + 8, device=device,
                                     evict=evict, prune_interval=args.prune_interval,
                                     block_steps=args.block_steps)
            spotchecks.append(dict(idx=idx, cluster=r["cluster"], **sc))
            print(f"  idx={idx} [{r['cluster']}] IDENTICAL={sc['identical']} "
                  f"driver_final={sc['driver_final_ax']} spec_final={sc['spec_final_ax']} "
                  f"steps={sc['n_steps_driver']}", file=sys.stderr, flush=True)

    # -- fold into the FULL-1096 pure-forward pass fraction -----------------------
    fold = None
    if snap_summary is not None:
        base_pass = snap_summary.get("PASS", 0)          # 725
        base_total = snap_scored_total                    # 829 (scored, top-level)
        # replace the 104 TIMEOUTs with their speculative verdicts.
        new_pass = base_pass + n_pass
        new_fail = counts.get("FAIL", 0)
        new_err = counts.get("ERROR", 0)
        new_timeout = counts.get("TIMEOUT", 0)
        fold = dict(
            scored_total=base_total,
            token_by_token_pass=base_pass,
            deep_loop_timeouts=len(timeout_ids),
            spec_pass=n_pass, spec_fail=new_fail, spec_error=new_err,
            spec_timeout=new_timeout,
            full_pass=new_pass,
            full_pass_frac=round(100.0 * new_pass / base_total, 2) if base_total else 0.0)

    # -- report -------------------------------------------------------------------
    print("\n" + "=" * 74)
    print(f"PERFECT-DRAFT SPECULATION — deep-loop TIMEOUT closer on {device}")
    print("=" * 74)
    print(f"  ran {total_spec} TIMEOUT programs | wall {wall:.0f}s "
          f"({wall/60:.1f} min) | model {st.sparse_mb:.1f}MB VRAM-load {vram_load:.0f}MB")
    print(f"  status: " + " ".join(f"{s}={counts.get(s,0)}"
                                    for s in ("PASS", "FAIL", "TIMEOUT", "ERROR")))
    print(f"  forward-count speedup (naive token-by-token vs speculative): "
          f"{sum_naive} -> {sum_spec} forwards = {agg_speedup:.1f}x")
    print("-" * 74)
    print("  PER-CLUSTER (of the deep-loop TIMEOUTs):")
    for cl in sorted(per_cluster):
        c = per_cluster[cl]
        n = sum(c.values())
        print(f"    {cl:16s} n={n:3d}  PASS={c.get('PASS',0):3d} "
              f"FAIL={c.get('FAIL',0):3d} TIMEOUT={c.get('TIMEOUT',0):3d} "
              f"ERROR={c.get('ERROR',0):3d}")
    if args.spotcheck:
        all_id = all(s["identical"] for s in spotchecks) if spotchecks else False
        print("-" * 74)
        print(f"  BYTE-IDENTITY vs token-by-token KV-cached driver: "
              f"{sum(1 for s in spotchecks if s['identical'])}/{len(spotchecks)} "
              f"identical {'(ALL MATCH)' if all_id else '(MISMATCH!)'}")
    if fold is not None:
        print("=" * 74)
        print(f"  FULL-1096 PURE-FORWARD FOLD:")
        print(f"    token-by-token: {fold['token_by_token_pass']} PASS / "
              f"{fold['scored_total']} scored")
        print(f"    + speculation on the {fold['deep_loop_timeouts']} TIMEOUTs: "
              f"+{fold['spec_pass']} PASS "
              f"(FAIL {fold['spec_fail']}, TIMEOUT {fold['spec_timeout']}, "
              f"ERROR {fold['spec_error']})")
        print(f"    == FULL PURE-FORWARD: {fold['full_pass']}/{fold['scored_total']} "
              f"= {fold['full_pass_frac']}%")
    print("=" * 74)

    # -- genuine fails (model argmax != draft) — REPORT, don't hide. --------------
    genuine = [r for r in results if r["status"] in ("FAIL", "ERROR")]
    if genuine:
        print("\nGENUINE FAILS / ERRORS (report, not hide):")
        for r in genuine:
            print(f"  id={r['idx']} [{r['cluster']}] {r['status']}: {r.get('detail','')}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump({
                "device": device, "wall_seconds": wall,
                "include_divmod": include_divmod,
                "compute_mode": args.compute_mode, "evict": evict,
                "block_steps": args.block_steps,
                "storage_mb": st.sparse_mb, "vram_load_mb": vram_load,
                "n_timeout_programs": len(timeout_ids),
                "status_counts": dict(counts),
                "forward_speedup": {
                    "naive_forwards": sum_naive, "spec_forwards": sum_spec,
                    "speedup": round(agg_speedup, 2)},
                "per_cluster": {cl: dict(c) for cl, c in per_cluster.items()},
                "full_1096_fold": fold,
                "spotchecks": spotchecks,
                "results": results,
            }, fh, indent=2)
        print(f"[spec:{device}] wrote {args.output}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
