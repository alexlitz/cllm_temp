"""#741 — measure the fused pos-sparse ms/step at each fusion level, byte-exact.

Levels:
  0. POSITION-SPARSE eager (baseline, ``PositionSparseRunner``, the #745 number).
  1. PER-BLOCK FUSION  (``FusedPosSparseRunner``: KV-concat + gate/up-concat GEMMs).
  2. WHOLE-STEP CUDA GRAPH (``PosSparseStepGraph``: one graph launch/op-class).

Byte-exact gate: the LEVEL-1 fused runner is driven through the SAME
``verify_composition`` (full-238 dense-over-positions == fused pos-sparse) as the
#745 bench, and the LEVEL-2 graph output is L-inf checked vs the eager LEVEL-1 forward.

Run (needs a free >=18 GB CUDA card):
    python -m c4_min.bench_pos_sparse_fused --S 300,900
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from typing import Dict, List, Optional

os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch

from . import isa
from .pos_sparse_forward import PositionSparseRunner
from .pos_sparse_fused import FusedPosSparseRunner, PosSparseStepGraph
from .bench_composed_fast_path import wait_for_gpu, _OP_MIX, _OP_LABEL


def _time_fn(fn, n, warmup, cuda):
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    if cuda:
        torch.cuda.synchronize()
    return (time.time() - t0) / n * 1e3


# ---------------------------------------------------------------------------
# BYTE-EXACT gate: reuse the #745 verify_composition but with the FUSED runner
# swapped in for the composed forward.
# ---------------------------------------------------------------------------
def verify_fused(model, L, verbose=True) -> bool:
    """Full-238 dense-over-positions == LEVEL-1 fused pos-sparse, per-step AX, over
    the #745 battery + the deep nested loop.  Uses the fused runner in place of the
    plain PositionSparseRunner inside the composed path."""
    from .bench_pos_sparse_composed import PosSparseComposed, _drive, _battery, \
        _nested_prog
    from .pf_speculative import draft_pf_program

    # Build a composed wrapper whose runner is the FUSED runner.
    composed = PosSparseComposed(model, L, direct_cam=True)
    composed.runner = FusedPosSparseRunner(model, L)
    ok = True
    for name, prog, seed in _battery():
        code = prog if (prog and isinstance(prog[0], isa.Instr)) else isa.assemble(prog)
        base, _ = _drive(model, L, code, composed=None, seed_mem=seed)
        comp, _ = _drive(model, L, code, composed=composed, seed_mem=seed)
        match = base == comp
        ok = ok and match
        if verbose:
            print(f"  {name:10s} {'OK ' if match else 'FAIL'} steps={len(base)}",
                  flush=True)
            if not match:
                print(f"    BASE={base}\n    COMP={comp}", flush=True)
    deep_steps = 60
    deep = _nested_prog(3, 4)
    draft = draft_pf_program(deep, max_steps=deep_steps, mask=0xFF)
    draft_ax = [f["ax"] & 0xFF for f in draft.frames]
    comp, _ = _drive(model, L, deep, composed=composed, seed_mem={}, max_steps=deep_steps)
    n = min(len(comp), len(draft_ax))
    dmatch = comp[:n] == draft_ax[:n] and n > 0
    ok = ok and dmatch
    if verbose:
        print(f"  {'nested_deep':10s} {'OK ' if dmatch else 'FAIL'} steps={n}",
              flush=True)
        if not dmatch:
            print(f"    DRAFT={draft_ax[:20]}\n    COMP ={comp[:20]}", flush=True)
    return ok


def bench_at_S(model, L, S: int, *, n=30, warmup=8, use_graphs=True):
    dev = model.embed.device
    cuda = (dev.type == "cuda")
    eager = PositionSparseRunner(model, L)          # LEVEL 0 (baseline)
    fused = FusedPosSparseRunner(model, L)          # LEVEL 1
    D = model.embed.shape[1]
    x0 = torch.zeros(1, S, D, device=dev, dtype=model.embed.dtype)

    print(f"\n{'='*84}\n[bench] S={S}  (n={n}, warmup={warmup})\n{'='*84}", flush=True)

    graph = None
    graph_status: Dict = {}
    if use_graphs and cuda:
        graph = PosSparseStepGraph(fused, S, dev)

    per_op: Dict[str, Dict[int, float]] = {"eager": {}, "fused": {}, "graph": {}}
    print(f"\n  {'op':>5} {'live':>5} {'L0 eager':>11} {'L1 fused':>11} "
          f"{'L2 graph':>11} {'L0/L2':>8}", flush=True)
    print("  " + "-" * 62, flush=True)
    for op in _OP_MIX:
        live = fused.live_index[op]

        def f_eager(op=op):
            with torch.no_grad():
                return eager.forward(x0, op)

        def f_fused(op=op):
            with torch.no_grad():
                return fused.forward(x0, op)

        t_eager = _time_fn(f_eager, n, warmup, cuda)
        t_fused = _time_fn(f_fused, n, warmup, cuda)
        per_op["eager"][op] = t_eager
        per_op["fused"][op] = t_fused
        t_graph = None
        if graph is not None:
            key = graph.op_to_key[op]
            if key not in graph.graphs and key not in graph_status:
                graph_status[key] = graph.try_capture(key)
            if graph.graphs.get(key) is not None:
                t_graph = graph.replay_ms(key, iters=n, warmup=warmup)
                per_op["graph"][op] = t_graph
        lbl = _OP_LABEL.get(op, str(op))
        gtxt = f"{t_graph:9.3f}" if t_graph is not None else "  (no-graph)"
        spd = f"{t_eager/max(t_graph,1e-9):7.1f}x" if t_graph else "     -"
        print(f"  {lbl:>5} {len(live):>5} {t_eager:9.3f}  {t_fused:9.3f}  "
              f"{gtxt:>11} {spd:>8}", flush=True)

    def mix(times):
        return sum(times[op] * w for op, w in _OP_MIX.items() if op in times)

    m_eager = mix(per_op["eager"])
    m_fused = mix(per_op["fused"])
    best_graph = {op: per_op["graph"].get(op, per_op["fused"][op]) for op in _OP_MIX}
    m_graph = mix(best_graph)
    print(f"\n  --- WEIGHTED OP-MIX ms/step at S={S} ---", flush=True)
    print(f"    L0 POSITION-SPARSE (eager)      : {m_eager:9.3f} ms/step  (1.0x)",
          flush=True)
    print(f"    L1 per-block fused              : {m_fused:9.3f} ms/step  "
          f"({m_eager/max(m_fused,1e-9):.2f}x vs L0)", flush=True)
    if per_op["graph"]:
        print(f"    L2 whole-step CUDA graph        : {m_graph:9.3f} ms/step  "
              f"({m_eager/max(m_graph,1e-9):.2f}x vs L0)   <-- MEGAKERNEL", flush=True)

    div = isa.DIV
    print(f"\n  --- DIV focus ({len(fused.live_index[div])}-block divmod span) ---",
          flush=True)
    print(f"    L0 eager : {per_op['eager'][div]:9.3f} ms/step", flush=True)
    print(f"    L1 fused : {per_op['fused'][div]:9.3f} ms/step  "
          f"({per_op['eager'][div]/max(per_op['fused'][div],1e-9):.2f}x)", flush=True)
    if div in per_op["graph"]:
        print(f"    L2 graph : {per_op['graph'][div]:9.3f} ms/step  "
              f"({per_op['eager'][div]/max(per_op['graph'][div],1e-9):.2f}x)", flush=True)

    if graph is not None:
        n_ok = sum(1 for k in graph.distinct_keys if graph.graphs.get(k) is not None)
        print(f"\n  CUDA graphs captured {n_ok}/{len(graph.distinct_keys)} "
              f"distinct op-class shapes", flush=True)
        for k in sorted(graph.distinct_keys, key=len):
            ops = ",".join(_OP_LABEL.get(o, str(o)) for o in graph._seen[k])
            st = "ok" if graph.graphs.get(k) is not None else graph_status.get(k, "?")
            print(f"    live={len(k):3d}  [{ops}]  -> {st}", flush=True)
        graph.graphs.clear()
        del graph
    return {"S": S, "eager": m_eager, "fused": m_fused, "graph": m_graph,
            "div_eager": per_op["eager"][div], "div_fused": per_op["fused"][div],
            "div_graph": per_op["graph"].get(div)}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--S", type=str, default="300,900")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--code-size", type=int, default=64)
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=18.0)
    ap.add_argument("--stable-s", type=float, default=60.0)
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args(argv)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[fused] CUDA unavailable; cpu", file=sys.stderr)
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

    if not args.no_verify:
        print("\n[verify] full-238 == LEVEL-1 per-block-fused pos-sparse:", flush=True)
        ok = verify_fused(model, L)
        print(f"[verify] {'ALL BYTE-EXACT' if ok else 'DIVERGENCE'}", flush=True)
        if not ok:
            print("[verify] fused NOT byte-exact -> abort", flush=True)
            return 1

    results = []
    for S in (int(s) for s in args.S.split(",") if s.strip()):
        results.append(bench_at_S(model, L, S, n=args.n,
                                  use_graphs=(not args.no_graphs)))
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    print(f"\n{'='*84}\n[self-emulation wall] fused pos-sparse ms/step x 23M steps\n"
          f"{'='*84}", flush=True)
    for r in results:
        best = r["graph"] if r.get("graph") else r["fused"]
        hrs = best * 23e6 / 1e3 / 3600.0
        print(f"  S={r['S']:5d}: best {best:.3f} ms/step x 23M = {hrs:8.1f} hr",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
