#!/usr/bin/env python3
"""_agent_dense_vs_sparse_probe.py — lever 3 ceiling: is the sparse-CSR fused mega chain
kernel-efficiency-bound? Compare it against a DENSE-GEMM equivalent of the SAME dead-FFN
chain (up/gate/down as dense [Dff,D]@[D,C] cuBLAS GEMMs), and against the theoretical
roofline. Also measures the achieved GB/s of both so we know how far from peak we are.

If the dense chain (which streams the FULL dense weights, ~all zeros) is FASTER than the
compacted sparse chain, the sparse per-nnz gather is the waste and opcode-segmented dense
GEMMs are the lever. If sparse is faster (fewer bytes), compaction is already tight and
the floor is HBM.

We build a small representative slice (a subset of dead blocks) to bound VRAM, and scale.
"""
from __future__ import annotations
import os, time, json
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
import c4_min.nibble_pure_forward_cached as _PFCa
_PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = 0xFC

from c4_min.lib_neural import build_lib_model_streaming
from c4_min.pf_speculative import _frozen_skip_cut
from c4_min.tight_attn_compose import install_composed, uninstall_composed
from c4_min.fused_megablock import MegaBlockChain, _dense_of
import torch.nn.functional as Fn

COMPOSED = ["C4_DEAD_BLOCK_FUSION", "C4_DIRECT_CAM_BATCHED", "C4_DIRECT_LOCAL_CAM",
            "C4_FLASH_ATTN", "C4_BANDED_LOCAL_ATTN", "C4_FUSED_MEGABLOCK",
            "C4_DIRECT_CAM_VEC"]


def _cuda_time_ms(fn, reps=20, warmup=6, dev=None):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(dev)
    s = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    e = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        s[i].record(); fn(); e[i].record()
    torch.cuda.synchronize(dev)
    ts = sorted(s[i].elapsed_time(e[i]) for i in range(reps))
    keep = ts[2:-2] if len(ts) > 6 else ts
    return sum(keep) / len(keep)


def main():
    for f in COMPOSED:
        os.environ[f] = "1"
    dev = torch.device("cuda:0")
    t0 = time.time()
    model, L, _ = build_lib_model_streaming(code_size=256, recurrent_divmod=True,
                                            addr32=True, compute_mode="dense_kernel")
    model = model.to(dev)
    D = model.dim
    install_composed(model, verbose=False)
    cut = _frozen_skip_cut(model)
    dead = [b for b in range(cut, len(model.blocks))
            if getattr(model.blocks[b].attn, "_dead_block_fused", False)]
    model.materialize_dense(device=str(dev))
    print(f"[built] blocks={len(model.blocks)} dim={D} dead={len(dead)} {time.time()-t0:.1f}s",
          flush=True)

    # SPARSE chain time (graphed) at a chunk.
    C = 32768
    hq = torch.zeros(1, C, D, device=dev)
    chain = MegaBlockChain(model, dev, dead, block_k=64)
    _ = chain.run_graphed(hq); torch.cuda.synchronize(dev)
    ms_sparse = _cuda_time_ms(lambda: chain.run_graphed(hq), 25, 8, dev)
    print(f"\n[SPARSE mega chain] {ms_sparse:.4f} ms  {ms_sparse*1e3/C:.5f} us/step", flush=True)

    # DENSE equivalent: for each distinct dead ffn, dense up/gate/down GEMMs.
    # Build dense weights ONCE (dedup by id). Represent the chain as a list of
    # (Wu[Dff,D], Wg[Dff,D], Wd[D,Dff], b_up, b_gate) and run:
    #   h = silu(Wu@x + b_up) * (Wg@x + b_gate);  x = x + Wd@h
    # over a [D,C] residual. This streams the FULL dense weights each block.
    cache = {}
    dense_ffns = []
    total_dense_w_bytes = 0
    for b in dead:
        ffn = model.blocks[b].ffn
        key = id(ffn)
        d = cache.get(key)
        if d is None:
            Wu = _dense_of(ffn.W_up).to(dev).float()
            Wg = _dense_of(ffn.W_gate).to(dev).float()
            Wd = _dense_of(ffn.W_down).to(dev).float()
            bu = ffn.b_up.to(dev).float()
            bg = ffn.b_gate.to(dev).float()
            d = (Wu, Wg, Wd, bu, bg)
            cache[key] = d
            total_dense_w_bytes += (Wu.numel() + Wg.numel() + Wd.numel()) * 4
        dense_ffns.append(d)
    n_distinct = len(cache)
    print(f"[DENSE] distinct ffns={n_distinct} dense-weight VRAM={total_dense_w_bytes/1e9:.2f} GB",
          flush=True)

    x = torch.zeros(D, C, device=dev)   # [D,C] residual

    def _dense_chain():
        y = x
        for (Wu, Wg, Wd, bu, bg) in dense_ffns:
            up = Wu @ y + bu.view(-1, 1)
            gt = Wg @ y + bg.view(-1, 1)
            h = Fn.silu(up) * gt
            y = y + Wd @ h
        return y

    # graph-capture the dense chain for a fair replay comparison
    def _cap(fn):
        s = torch.cuda.Stream(device=dev)
        s.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(s):
            for _ in range(3):
                with torch.no_grad():
                    fn()
        torch.cuda.current_stream(dev).wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            with torch.no_grad():
                fn()
        return g

    try:
        gd = _cap(_dense_chain)
        ms_dense = _cuda_time_ms(lambda: gd.replay(), 20, 6, dev)
        print(f"[DENSE mega chain] {ms_dense:.4f} ms  {ms_dense*1e3/C:.5f} us/step", flush=True)
        # weight-stream bytes per replay (dense weights streamed once each, if not L2-resident)
        dense_w_stream = total_dense_w_bytes
        print(f"   dense weight-stream/replay ~ {dense_w_stream/1e9:.2f} GB "
              f"-> ach BW {dense_w_stream/(ms_dense*1e-3)/1e9:.0f} GB/s", flush=True)
        speedup = ms_sparse / ms_dense
        print(f"\n  sparse/dense ratio = {speedup:.3f}x  "
              f"{'DENSE FASTER (sparse gather is the waste; lever 3 wins)' if speedup>1.1 else 'SPARSE FASTER/equal (compaction already tight)'}",
              flush=True)
        out = {"chunk": C, "sparse_ms": ms_sparse, "dense_ms": ms_dense,
               "sparse_us_step": ms_sparse*1e3/C, "dense_us_step": ms_dense*1e3/C,
               "sparse_over_dense": speedup, "dense_weight_GB": dense_w_stream/1e9,
               "n_distinct_ffns": n_distinct}
    except torch.cuda.OutOfMemoryError:
        print("[DENSE] OOM building dense chain -- dense-weight VRAM too large", flush=True)
        out = {"chunk": C, "sparse_ms": ms_sparse, "dense": "OOM"}

    uninstall_composed(model)
    json.dump(out, open("/tmp/dense_vs_sparse.json", "w"), indent=2, default=str)
    print("[written] /tmp/dense_vs_sparse.json", flush=True)


if __name__ == "__main__":
    main()
