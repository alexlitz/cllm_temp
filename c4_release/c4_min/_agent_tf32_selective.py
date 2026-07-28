"""#751 — TF32-SELECTIVE-fp64: the minimal fp64 block set that keeps the battery
byte-exact under GLOBAL TF32 at K=32.

_agent_tf32_deep found TF32 is byte-exact at K=1 everywhere, but at K=32 flips `mod`
and `nested_deep` (a few nibbles).  TF32 is a GLOBAL backend flag (can't be set
per-block), but a block routed through fp64 is UNAFFECTED by it.  So the byte-exact
TF32 policy = fp64 on the (lea-addr-nib + the deep MOD/nested-fragile) blocks, TF32 on
the rest.  This greedily finds that minimal fp64 set under TF32-ON at K=32.

Run:  OMP_NUM_THREADS=4 python -m c4_min._agent_tf32_selective --device cuda:0
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Set, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch

from . import isa
from .pf_kbatch import KBatchBoundedRunner
from .pf_speculative import draft_pf_program
from . import bench_pf_kbatch as bk
from .bench_composed_fast_path import _nested_prog


def _battery() -> List[Tuple[str, list, dict, int]]:
    B = [
        ("add", [("IMM", 12), ("PSH", 0), ("IMM", 30), ("ADD", 0), ("HALT", 0)], {}, 40),
        ("mul", [("IMM", 12), ("PSH", 0), ("IMM", 7), ("MUL", 0), ("HALT", 0)], {}, 40),
        ("div", [("IMM", 100), ("PSH", 0), ("IMM", 7), ("DIV", 0), ("HALT", 0)], {}, 200),
        ("mod", [("IMM", 100), ("PSH", 0), ("IMM", 7), ("MOD", 0), ("HALT", 0)], {}, 200),
        ("li", [("IMM", 8), ("LI", 0), ("HALT", 0)], {8: 123}, 40),
        ("jsr_lev", [("JSR", 3), ("IMM", 5), ("HALT", 0), ("ENT", 0), ("IMM", 7), ("LEV", 0)], {}, 40),
    ]
    out = [(n, isa.assemble(p), s, ms) for n, p, s, ms in B]
    out.append(("nested_deep", _nested_prog(3, 4), {}, 60))
    return out


def _traces(model, L, runner, progs, K):
    return {n: bk.drive_kbatch(model, L, runner, c, K=K, seed_mem=s, max_steps=ms)[0]
            for n, c, s, ms in progs}


def _bad(ref, cur):
    return [n for n in ref if ref[n] != cur.get(n)]


def _live_pool(runner, progs, names):
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--code-size", type=int, default=64)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=18.0)
    ap.add_argument("--stable-s", type=float, default=20.0)
    a = ap.parse_args(argv)
    bk.runner_window = a.window

    device = a.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    idx = int(device.split(":")[1]) if (device.startswith("cuda") and ":" in device) else 0
    if device.startswith("cuda") and not a.no_wait:
        from .bench_composed_fast_path import wait_for_gpu
        wait_for_gpu(idx, min_free_gb=a.min_free_gb, stable_s=a.stable_s)

    os.environ["C4_POS_SPARSE"] = "1"
    from .compact_alloc import build_compact_sparse_streaming
    t0 = time.time()
    model, L, _ = build_compact_sparse_streaming(code_size=a.code_size, compute_mode="dense_kernel")
    if device != "cpu":
        model.to(device); model.materialize_dense(device)
    runner = KBatchBoundedRunner(model, L, window=a.window, selective_fp64=True)
    bnames = list(getattr(L, "_block_names", []))
    base_fp64 = set(bi for bi, kb in enumerate(runner.kblocks) if kb.b.fp64_ffn)
    K = a.K
    print(f"[built] blocks={len(model.blocks)} dim={model.embed.shape[1]} "
          f"base_fp64={sorted(bnames[b] for b in base_fp64)} build={time.time()-t0:.1f}s", flush=True)

    progs = _battery()

    # fp64 reference (TF32 does not affect fp64; ref is precision ground truth).
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runner.set_fp64_blocks(None)
    ref = _traces(model, L, runner, progs, K)

    # TF32 ON globally for the rest.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    def bad_at(fp64set, names=None):
        runner.set_fp64_blocks(fp64set)
        sub = [p for p in progs if names is None or p[0] in names]
        cur = {p[0]: bk.drive_kbatch(model, L, runner, p[1], K=K, seed_mem=p[2],
                                     max_steps=p[3])[0] for p in sub}
        return [n for n in {p[0] for p in sub} if ref[n] != cur.get(n)]

    fp64 = set(base_fp64)
    bad = bad_at(fp64)
    print(f"[tf32 selective, K={K}] baseline fp64={sorted(bnames[b] for b in fp64)} "
          f"-> flips {bad}", flush=True)
    if not bad:
        print("  ALREADY BYTE-EXACT with base selective-fp64 under TF32.", flush=True)
        return 0

    pool = _live_pool(runner, progs, set(bad)) - fp64
    print(f"[pool] {len(pool)} candidate blocks over failing {bad}", flush=True)
    while True:
        bad = bad_at(fp64, names=set(bad) if bad else None)
        bad = bad_at(fp64)
        if not bad:
            break
        best, best_fix = None, -1
        for b in sorted(pool):
            fixed = len(bad) - len(bad_at(fp64 | {b}, names=set(bad)))
            if fixed > best_fix:
                best_fix, best = fixed, b
        if best is None or best_fix <= 0:
            print("[greedy] no single-block progress; forcing full pool to fp64", flush=True)
            fp64 |= pool
            break
        fp64.add(best); pool.discard(best)
        print(f"[greedy] +fp64 block {best} ({bnames[best] if best < len(bnames) else best}) "
              f"(fixed {best_fix}); fp64 now {sorted(fp64)}", flush=True)

    # minimize.
    for b in sorted(fp64 - base_fp64):
        if not bad_at(fp64 - {b}):
            fp64 = fp64 - {b}
            print(f"[minimize] dropped {b}", flush=True)

    runner.set_fp64_blocks(fp64)
    final_bad = bad_at(fp64)
    extra = sorted(fp64 - base_fp64)
    print(f"\n{'='*74}\n[RESULT] TF32-byte-exact fp64 set at K={K}:\n"
          f"  fp64 blocks = {sorted(fp64)} ({len(fp64)}/{runner.n_blocks})\n"
          f"  = base {sorted(base_fp64)} + TF32-extra {extra} "
          f"({[bnames[b] if b < len(bnames) else b for b in extra]})\n"
          f"  final verify: {'BYTE-EXACT' if not final_bad else 'FAIL ' + str(final_bad)}\n"
          f"{'='*74}", flush=True)
    print(f"total wall: {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
