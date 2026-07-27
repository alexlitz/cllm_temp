"""#748 — SELECTIVE-fp64 block analyzer.

a39ae2c runs the query-row SwiGLU FFN in fp64 on EVERY live block to fix the
``lea-addr-nib`` / nested-deep decode tie (a 1-row cuBLAS GEMV whose fp32 accum
order flips the integer decode margin).  But that fp64 is only NEEDED on the
fp-fragile blocks; the A5000 runs fp64 ~1/32 fp32 and Q=1->2 jumps the per-block
dense GEMM ~8x, so the ~11 fp-robust live blocks are paying a big fp64 tax for
nothing.

This analyzer finds the MINIMAL set of blocks that must stay fp64 to keep the
whole battery (arith/DIV/MOD/mem/JSR-LEV/branch + a deep nested loop) BYTE-EXACT
vs the all-fp64 reference, using the bounded-KV K-batch runner:

  1. reference = all-fp64 trace per battery program (a39ae2c byte-exact path).
  2. all-fp32 trace; if byte-exact everywhere -> NO block needs fp64 (report []).
  3. else greedily REPAIR: start all-fp32, add fp64 for the live block that fixes
     the most failing programs, repeat until byte-exact.  Report the fp64 set.
  4. then MINIMIZE: try dropping each fp64 block back to fp32 (re-verify) so the
     reported set is minimal (no redundant fp64 block).

The reported set is the union over ALL battery programs at ALL Ks tested (the
LEA-snap head + nested-deep are the expected members).  Prints the set + the
per-op msstep so the bench can pin it.

Run (needs a free >=18 GB CUDA card):
    OMP_NUM_THREADS=4 python -m c4_min.analyze_fp64_blocks --K 1,32 --device cuda:0
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Optional, Set, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch

from . import isa
from .pf_kbatch import KBatchBoundedRunner
from .pf_speculative import draft_pf_program
from . import bench_pf_kbatch as bk
from .bench_composed_fast_path import wait_for_gpu, _battery, _nested_prog


def _progs() -> List[Tuple[str, list, dict, int]]:
    """(name, code, seed, max_steps) for the battery + the deep nested loop."""
    out: List[Tuple[str, list, dict, int]] = []
    for name, prog, seed in _battery():
        code = prog if (prog and isinstance(prog[0], isa.Instr)) else isa.assemble(prog)
        out.append((name, code, seed, 200))
    out.append(("nested_deep", _nested_prog(3, 4), {}, 60))
    return out


def _all_traces(model, L, runner, K: int, progs=None) -> Dict[str, List[int]]:
    """Decode the given programs (default: all) with the CURRENT runner fp64
    setting.  Returns {name: ax_trace}."""
    if progs is None:
        progs = _progs()
    out: Dict[str, List[int]] = {}
    for name, code, seed, ms in progs:
        tr, _ = bk.drive_kbatch(model, L, runner, code, K=K, seed_mem=seed, max_steps=ms)
        out[name] = tr
    return out


def _mismatches(ref: Dict[str, List[int]], cur: Dict[str, List[int]]) -> List[str]:
    bad = []
    for name in ref:
        if ref[name] != cur.get(name):
            bad.append(name)
    return bad


def _live_pool_for(runner, progs, names: Set[str], max_steps_default=200) -> Set[int]:
    """Union of live blocks over the named programs (fp64 on a block a program never
    touches can't matter for that program)."""
    pool: Set[int] = set()
    for name, code, seed, ms in progs:
        if name not in names:
            continue
        draft = draft_pf_program(code, max_steps=ms, mask=0xFFFFFFFF)
        cur_pc, ops = 0, []
        for f in draft.frames:
            op = code[cur_pc].op if 0 <= cur_pc < len(code) else None
            ops.append(op)
            cur_pc = f["pc"]
        pool |= set(runner.live_union(ops))
    return pool


def analyze(model, L, runner, Ks: List[int]) -> Set[int]:
    """Return the MINIMAL fp64 block set (byte-exact over battery+nested, all Ks).

    Search is over K=min(Ks) on ONLY the failing programs (fast); the full battery
    at ALL Ks is re-verified at the end (in the bench / verify pass)."""
    progs = _progs()
    Ksearch = min(Ks)
    # reference: all-fp64 (a39ae2c byte-exact path) at every K.
    runner.set_fp64_blocks(None)
    refs = {K: _all_traces(model, L, runner, K, progs) for K in Ks}

    # 1) all-fp32 test at every K.
    runner.set_fp64_blocks(set())          # empty -> every block fp32
    total_bad: Set[str] = set()
    for K in Ks:
        total_bad |= set(_mismatches(refs[K], _all_traces(model, L, runner, K, progs)))
    print(f"[fp32-all] byte-exact except: {sorted(total_bad) or 'NONE (no fp64 needed!)'}",
          flush=True)
    if not total_bad:
        return set()

    # candidate pool + the failing subset (search only re-decodes these).
    live_pool = _live_pool_for(runner, progs, total_bad)
    fail_progs = [p for p in progs if p[0] in total_bad]
    print(f"[pool] {len(live_pool)} candidate live blocks over failing "
          f"programs {sorted(total_bad)}", flush=True)

    def bad_at(setblocks) -> Set[str]:
        runner.set_fp64_blocks(setblocks)
        return set(_mismatches({p[0]: refs[Ksearch][p[0]] for p in fail_progs},
                               _all_traces(model, L, runner, Ksearch, fail_progs)))

    # 2) greedy repair over the failing programs at Ksearch.
    fp64: Set[int] = set()
    remaining = set(live_pool)
    while True:
        bad = bad_at(fp64)
        if not bad:
            break
        best, best_fix = None, -1
        for b in sorted(remaining):
            fixed = len(bad) - len(bad_at(fp64 | {b}))
            if fixed > best_fix:
                best_fix, best = fixed, b
        if best is None or best_fix <= 0:
            print("[greedy] no single-block progress; using full pool", flush=True)
            fp64 = set(live_pool)
            break
        fp64.add(best); remaining.discard(best)
        print(f"[greedy] +fp64 block {best} (repaired {best_fix}); "
              f"fp64 now {sorted(fp64)}", flush=True)

    # 3) minimize: drop each fp64 block if the set stays byte-exact without it.
    for b in sorted(fp64):
        if not bad_at(fp64 - {b}):
            fp64 = fp64 - {b}
            print(f"[minimize] dropped redundant block {b}; fp64 now {sorted(fp64)}",
                  flush=True)

    # 4) FINAL full-battery re-verify at ALL Ks with the reduced set.
    print(f"[reverify] full battery+nested at K={Ks} with fp64={sorted(fp64)} ...",
          flush=True)
    runner.set_fp64_blocks(fp64)
    for K in Ks:
        bad = _mismatches(refs[K], _all_traces(model, L, runner, K, progs))
        print(f"  K={K}: {'BYTE-EXACT' if not bad else 'FAIL '+str(bad)}", flush=True)
        if bad:
            raise SystemExit(f"[reverify] selective-fp64 NOT byte-exact at K={K}: {bad}")
    return fp64


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--K", type=str, default="1,32")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--code-size", type=int, default=64)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--min-free-gb", type=float, default=18.0)
    ap.add_argument("--stable-s", type=float, default=60.0)
    ap.add_argument("--no-wait", action="store_true")
    args = ap.parse_args(argv)
    bk.runner_window = args.window

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    idx = int(device.split(":")[1]) if (device.startswith("cuda") and ":" in device) else 0
    if device.startswith("cuda") and not args.no_wait:
        wait_for_gpu(idx, min_free_gb=args.min_free_gb, stable_s=args.stable_s)

    os.environ["C4_POS_SPARSE"] = "1"
    from .compact_alloc import build_compact_sparse_streaming
    t0 = time.time()
    model, L, _ = build_compact_sparse_streaming(
        code_size=args.code_size, compute_mode="dense_kernel")
    if device != "cpu":
        model.to(device)
        model.materialize_dense(device)
    print(f"[built] n_blocks={len(model.blocks)} dim={model.embed.shape[1]} "
          f"dev={device} build={time.time()-t0:.1f}s", flush=True)

    runner = KBatchBoundedRunner(model, L, window=args.window)
    Ks = [int(k) for k in args.K.split(",") if k.strip()]
    t1 = time.time()
    fp64 = analyze(model, L, runner, Ks)
    print(f"[analyze] {time.time()-t1:.0f}s", flush=True)
    print(f"\n{'='*72}\n[RESULT] minimal fp64 block set (byte-exact over battery+nested "
          f"at K={Ks}):\n  {sorted(fp64)}  ({len(fp64)}/{runner.n_blocks} blocks)\n"
          f"{'='*72}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
