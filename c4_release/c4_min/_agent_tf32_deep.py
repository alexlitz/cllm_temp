"""#751 — TF32 byte-exactness on the DEEP path (K=1,32 incl DIV/MOD + nested loop).

Confirms the quick verdict (_agent_tf32_quick: TF32 byte-exact, bf16 flips) holds on
the recurrent DIV/MOD megablock and a deep nested loop at K=32.  TF32-ONLY (skips the
expensive bf16 flip-search) so it finishes in a few minutes.

Run:  OMP_NUM_THREADS=4 python -m c4_min._agent_tf32_deep --device cuda:0 --K 1,32
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch

from . import isa
from .pf_kbatch import KBatchBoundedRunner
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--K", default="1,32")
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
    names = list(getattr(L, "_block_names", []))
    fp64 = [bi for bi, kb in enumerate(runner.kblocks) if kb.b.fp64_ffn]
    print(f"[built] blocks={len(model.blocks)} dim={model.embed.shape[1]} "
          f"fp64={[names[b] for b in fp64]} build={time.time()-t0:.1f}s", flush=True)

    Ks = [int(k) for k in a.K.split(",") if k.strip()]
    progs = _battery()

    for K in Ks:
        print(f"\n{'='*70}\nK={K}\n{'='*70}", flush=True)
        # fp64 ref.
        runner.set_fp64_blocks(None)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        ref = _traces(model, L, runner, progs, K)

        # fp32 + TF32 (selective fp64).
        runner.set_fp64_blocks(set(fp64))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        cur = _traces(model, L, runner, progs, K)
        bad = _bad(ref, cur)
        print(f"  [tf32 K={K}] {'BYTE-EXACT (all: ' + ','.join(ref.keys()) + ')' if not bad else 'FLIP: ' + str(bad)}",
              flush=True)
        for n in bad:
            print(f"      {n}: ref={ref[n]}  got={cur[n]}", flush=True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    print(f"\ntotal wall: {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
