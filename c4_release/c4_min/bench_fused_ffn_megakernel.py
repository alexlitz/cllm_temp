"""#758 — bench the FUSED GATHER-GATE-SCATTER megakernel on the DIV/MOD FFN tail.

Measures, HONESTLY, the fused per-block PARALLEL Triton gather-gate-scatter chain
(``fused_ffn_megakernel.GraphedTritonBlockGSChain``) vs the prior dense-SwiGLU CUDA
graph (``pf_kbatch.GraphedFFNChain``) and the eager dense chain, on the DIV step's
186-block passthrough-FFN tail; and the END-TO-END composed DIV step (attention
prefix + FFN tail) so Amdahl is visible.

Reports: FFN-tail ms (dense-graphed / megakernel), end-to-end DIV-step ms, the
attn-prefix-vs-FFN-tail split (is the FFN the bottleneck?), the realized speedup vs
the 1429x FLOP-cut ceiling, and the K-scaling (flat == launch/replay-bound; linear ==
bandwidth-bound).

BYTE-EXACT: run ``c4_min.bench_pf_kbatch --K ... --graph`` with
``C4_FUSED_FFN_MEGAKERNEL=1`` for the full-battery decode-level verify (this bench is
timing + L-inf only).

Run (needs a free >=18 GB CUDA card):
    OMP_NUM_THREADS=4 C4_FUSED_FFN_MEGAKERNEL=1 python -m c4_min.bench_fused_ffn_megakernel
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional

os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch

from . import isa
from .step_block_skip import build_live_index
from .pf_kbatch import KBatchBoundedRunner, GraphedFFNChain
from .bench_composed_fast_path import wait_for_gpu
from .bench_pf_kbatch import _make_timing_stream
from .fused_ffn_megakernel import (
    GraphedTritonBlockGSChain, TritonBlockGSChain, analyze_chain, verify_chain_linf,
)


def _dense(sw):
    for a in ("dense_resident", "dense"):
        v = getattr(sw, a, None)
        if v is not None:
            return v
    return sw.csr.to_dense()


def _time(fn, n=50, w=8):
    for _ in range(w):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1e3


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--K", type=str, default="1,8,32,128")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--code-size", type=int, default=64)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--base-s", type=int, default=900)
    ap.add_argument("--op", type=str, default="DIV")
    ap.add_argument("--min-free-gb", type=float, default=18.0)
    ap.add_argument("--stable-s", type=float, default=60.0)
    ap.add_argument("--no-wait", action="store_true")
    args = ap.parse_args(argv)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[fused-ffn] CUDA unavailable; abort", file=sys.stderr)
        return 1
    idx = int(device.split(":")[1]) if ":" in device else 0
    if device.startswith("cuda") and not args.no_wait:
        wait_for_gpu(idx, min_free_gb=args.min_free_gb, stable_s=args.stable_s)

    os.environ["C4_POS_SPARSE"] = "1"
    from .compact_alloc import build_compact_sparse_streaming
    t0 = time.time()
    model, L, _ = build_compact_sparse_streaming(
        code_size=args.code_size, compute_mode="dense_kernel")
    model.to(device)
    model.materialize_dense(device)
    runner = KBatchBoundedRunner(model, L, window=args.window, selective_fp64=True)
    li = build_live_index(model, L)
    op = getattr(isa, args.op)
    div_live = sorted(li[op])
    seg = [bi for bi in div_live if runner.kblocks[bi].b.is_passthrough]
    prefix = [bi for bi in div_live if bi not in set(seg)]
    D = model.embed.shape[1]
    dt = model.embed.dtype
    print(f"[built] {args.op}: {len(div_live)} live blocks "
          f"({len(seg)} passthrough-FFN + {len(prefix)} attn), D={D}, "
          f"build={time.time()-t0:.1f}s", flush=True)

    st = analyze_chain(runner.kblocks, seg)
    dense_flops = gather_flops = 0
    for bi in seg:
        Wu = _dense(model.blocks[bi].ffn.W_up)
        Wg = _dense(model.blocks[bi].ffn.W_gate)
        Wd = _dense(model.blocks[bi].ffn.W_down)
        dff, d = Wu.shape
        dense_flops += 2 * (d * dff + d * dff + dff * d)
        gather_flops += 2 * (int((Wu != 0).sum()) + int((Wg != 0).sum())
                             + int((Wd != 0).sum()))
    print(f"[structure] hidden-units={st['n_units']:,} nnz(up/gate/down)="
          f"{st['nnz_up']}/{st['nnz_gate']}/{st['nnz_down']} reads/unit="
          f"{st['reads_per_unit']:.2f} (max {st['max_read']})", flush=True)
    print(f"[FLOP-cut] dense FFN={dense_flops:,}/row  gather={gather_flops:,}/row "
          f"-> {dense_flops/gather_flops:.0f}x fewer FLOPs (the ceiling; memory/"
          f"launch-bound realizes << this)", flush=True)

    # L-inf honesty (residual-value divergence vs dense chain).
    linf = verify_chain_linf(runner.kblocks, seg, 8, D, device, dt, n_trials=4)
    print(f"[L-inf] fused vs dense residual = {linf:.2e} (multi-read/atomic accum "
          f"order; << the integer decode margin -> byte-exact at DECODE)", flush=True)

    Ks = [int(k) for k in args.K.split(",") if k.strip()]
    print(f"\n{'='*84}\n[FFN-tail]  {args.op} 186-block passthrough-FFN chain: dense "
          f"vs megakernel\n{'='*84}", flush=True)
    print(f"  {'K':>5} {'eager':>9} {'dense-graph':>12} {'megakernel':>11} "
          f"{'mega/dense':>11} {'mega/eager':>11}", flush=True)
    tail = {}
    for K in Ks:
        xr = torch.randn(1, K, D, device=device, dtype=dt) * 0.01

        def eager():
            out = xr
            for bi in seg:
                out = runner.kblocks[bi].b._ffn_qrow(out)
            return out
        ms_e = _time(eager, 15)
        gd = GraphedFFNChain(runner.kblocks, seg, K, D, device, dt)
        gd.try_capture()
        ms_g = _time(lambda: gd.run(xr))
        gm = GraphedTritonBlockGSChain(runner.kblocks, seg, K, D, device, dt)
        gm.try_capture()
        ms_m = _time(lambda: gm.run(xr))
        tail[K] = (ms_e, ms_g, ms_m)
        print(f"  {K:>5} {ms_e:>8.3f}m {ms_g:>11.3f}m {ms_m:>10.3f}m "
              f"{ms_g/ms_m:>10.2f}x {ms_e/ms_m:>10.2f}x", flush=True)

    print(f"\n{'='*84}\n[end-to-end] composed {args.op} step (attn prefix + FFN tail), "
          f"forward_span_graphed\n{'='*84}", flush=True)
    print(f"  {'K':>5} {'S':>6} {'prefix':>9} {'dense-e2e':>10} {'mega-e2e':>10} "
          f"{'e2e-speed':>10} {'FFN-frac':>9}", flush=True)
    for K in Ks:
        x0, q_idxs, S = _make_timing_stream(model, L, K, args.base_s, op, args.window)
        ops = [op] * len(q_idxs)
        q_t = torch.tensor(q_idxs, device=device, dtype=torch.long)
        col = x0[0, :, int(L.IS_STORE)]
        allstore = torch.nonzero(col != 0).flatten()

        def pref():
            buf = x0.clone()
            for bi in prefix:
                buf = runner.kblocks[bi].forward(buf, q_idxs, all_store=allstore,
                                                 q_t=q_t, inplace=True)
            return buf
        ms_pref = _time(pref, 15)
        os.environ["C4_FUSED_FFN_MEGAKERNEL"] = "0"
        runner._ffn_graphs = {}
        runner._graph_disabled = set()
        ms_de2e = _time(lambda: runner.forward_span_graphed(x0, ops, q_idxs), 15)
        os.environ["C4_FUSED_FFN_MEGAKERNEL"] = "1"
        runner._ffn_graphs = {}
        runner._graph_disabled = set()
        ms_me2e = _time(lambda: runner.forward_span_graphed(x0, ops, q_idxs), 15)
        ffn_frac = tail[K][1] / (ms_pref + tail[K][1]) * 100
        print(f"  {K:>5} {S:>6} {ms_pref:>8.3f}m {ms_de2e:>9.3f}m {ms_me2e:>9.3f}m "
              f"{ms_de2e/ms_me2e:>9.2f}x {ffn_frac:>8.0f}%", flush=True)

    print(f"\n{'='*84}\n[K-scaling] megakernel (flat==launch/replay-bound; "
          f"linear==bandwidth-bound)\n{'='*84}", flush=True)
    prev = None
    for K in [1, 2, 8, 32, 128, 256]:
        xr = torch.randn(1, K, D, device=device, dtype=dt) * 0.01
        gm = GraphedTritonBlockGSChain(runner.kblocks, seg, K, D, device, dt)
        if not gm.try_capture():
            continue
        ms = _time(lambda: gm.run(xr), 80)
        r = f"x{ms/prev:.2f}/2x-K" if prev else ""
        print(f"  K={K:>4}: {ms:6.3f}ms  {r}", flush=True)
        prev = ms
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
