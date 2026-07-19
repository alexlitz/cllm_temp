#!/usr/bin/env python3
"""Run the deep-tail rec_fib programs (>512 steps) via the VALIDATED block-verify
(pf_speculative.speculative_run) — the O(steps/block_steps)-forward path, so the
8369-step fib(12) is a few hundred forwards, not 8369.

Runs EXACTLY the given ids (default: the 13 deep rec_fib ids >512 steps), each
byte-identical to the token-by-token KV-cached driver (block-verify only ACCEPTS
what the model itself decodes).  Reports got vs expected + wall.
"""
from __future__ import annotations
import argparse, json, os, sys, time
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
from c4_min import isa
from c4_min.compact_alloc import build_compact_pure_forward_model
from c4_min.sparse_forward import SparseTransformer
from c4_min.pf_speculative import speculative_run
from src.compiler import compile_c
from tests.test_suite_1000 import generate_test_programs

_WORD = 8
_SS = frozenset({isa.LEA, isa.ENT, isa.ADJ})


def _s32(i):
    return i if i < (1 << 31) else i - (1 << 32)


def b2i(bc):
    out = []
    for w in bc:
        op = int(w) & 0xFF
        im = int(w) >> 8
        out.append(isa.Instr(op, _s32(im) // _WORD if op in _SS else im & 0xFFFFFFFF))
    return out


# the 13 deep rec_fib ids (>512 ref steps) — marked DEEP by the bulk fast runner.
DEEP_IDS = [734, 747, 728, 732, 737, 745, 743, 738, 727, 730, 735, 741, 749]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ids", default=None, help="comma list; default the 13 deep ids")
    ap.add_argument("--block-steps", type=int, default=48)
    ap.add_argument("--prune-interval", type=int, default=120)
    ap.add_argument("--max-steps", type=int, default=300000)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    dev = args.device
    if dev.startswith("cuda"):
        torch.zeros(1).to(dev)     # init CUDA context before the long CPU build
    ids = DEEP_IDS if args.ids is None else [int(x) for x in args.ids.split(",")]

    t0 = time.monotonic()
    print(f"[deep-tail:{dev}] building divmod sparse model ...", flush=True)
    base, L, _cs = build_compact_pure_forward_model(
        code_size=64, include_bitwise=False, include_divmod=True)
    sparse = SparseTransformer(base, compute_mode="dense_kernel").to(dev)
    del base
    print(f"[deep-tail:{dev}] built in {time.monotonic()-t0:.0f}s "
          f"blocks={len(sparse.blocks)} | running {len(ids)} deep programs", flush=True)

    tests = generate_test_programs()
    results = []
    n_ok = 0
    trun = time.monotonic()
    for idx in ids:
        src, exp, desc = tests[idx]
        code = b2i(compile_c(src)[0])
        t1 = time.monotonic()
        r = speculative_run(sparse, L, code, exp, block_steps=args.block_steps,
                            max_steps=args.max_steps, device=dev, evict=True,
                            prune_interval=args.prune_interval)
        dt = time.monotonic() - t1
        ok = (r.status == "PASS")
        n_ok += ok
        results.append(dict(idx=idx, cluster="rec_fib", status=r.status,
                            expected=r.expected, got=r.decoded_final_ax,
                            steps=r.step_count, forwards=r.forwards,
                            speedup=round(r.speedup, 1), max_cache=r.max_cache_size,
                            wall=round(dt, 1), detail=r.detail, description=desc))
        print(f"  id={idx:4d} steps={r.step_count:5d} exp={r.expected:4d} "
              f"got={r.decoded_final_ax} {r.status:5s} fwds={r.forwards} "
              f"speedup={r.speedup:.0f}x cache={r.max_cache_size} dt={dt:.0f}s "
              f":: {desc}", flush=True)
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
    wall = time.monotonic() - trun
    print(f"\n[deep-tail] {n_ok}/{len(ids)} PASS | run wall {wall:.0f}s "
          f"({wall/60:.1f} min)", flush=True)
    if args.output:
        json.dump({"device": dev, "wall_seconds": wall, "results": results},
                  open(args.output, "w"), indent=2)
        print(f"[deep-tail] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
