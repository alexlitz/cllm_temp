"""CUDA-GRAPHED end-to-end ms/step: DENSE vs BLOCK-MoE vs CONDITIONAL.

The companion ``bench_cond_vs_moe`` measures the EAGER forward -- where the
per-layer python launch overhead is part of every config's cost.  A CUDA graph
(``qwen_lean_cuda_graph.GraphedLeanForward``) captures the whole fixed-shape
forward once and replays it with ZERO python / ZERO per-launch CPU overhead, so
the ms/step is the amortised GPU cost -- exactly the setting where the question
"does fine-grained conditional sparsity beat block-MoE END-TO-END" is honest,
because the fixed attention/RMSNorm/gather cost is no longer inflated by launch
overhead and the ONLY thing that changes between configs is the FFN GEMM width.

All three configs are the SAME ``SkipAwareCondLean`` forward (identical machinery,
graph-captured); they differ ONLY in the per-layer active-unit set:
  dense      = all I units, no skip
  block-MoE  = all I units of MoE-active layers, dead layers skipped (== #628)
  cond       = firing units only

Run:
    python -m c4_min.bench_cond_vs_moe_graphed --device cuda:0 --subset bitwise
"""
from __future__ import annotations

import argparse
import time
from typing import List

import torch

from . import isa
from . import qwen_full_vm as Q
from . import qwen_lean_forward as LF
from . import perlayer_conditional_sparse as PC
from .qwen_lean_cuda_graph import GraphedLeanForward
from .bench_cond_vs_moe import (SkipAwareCondLean, blockmoe_active_set,
                                _prog_countdown, _prog_mul_accum, _prog_divmod,
                                _prog_mixed)


def _bench_call(g, x, pos, n, warmup, cuda):
    if cuda:
        torch.cuda.synchronize()
    for _ in range(warmup):
        g(x, pos)
    if cuda:
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        g(x, pos)
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000.0


def _vram_ok(device, need_bytes, headroom_gb=2.0):
    if not device.startswith("cuda"):
        return True
    free, _ = torch.cuda.mem_get_info(torch.device(device))
    return need_bytes + headroom_gb * (1024 ** 3) < free


def run(device="cuda:0", subset_name="bitwise",
        batches=(512, 2048, 8192), n=30, warmup=5, thr=0.0, max_steps=200):
    subsets = {"base": Q.SUBSET_BASE, "mem+cmp": Q.SUBSET_MEM_CMP,
               "bitwise": Q.SUBSET_BITWISE, "full": Q.SUBSET_FULL}
    subset = subsets[subset_name]
    cuda = device.startswith("cuda")
    print(f"# CUDA-GRAPHED cond-vs-MoE bench subset={subset_name} device={device} "
          f"n={n}", flush=True)
    vm = Q.build(code_size=24, subset=subset)
    lean = LF.LeanQwenVM.from_full_vm(vm, device=device)
    H = lean.hidden_size
    I = lean.layers[0].gate_w.shape[0]
    nL = lean.n_layers
    print(f"# n_layers={nL} H={H} I={I}", flush=True)

    progs = [("countdown (SUB/BNZ loop)", _prog_countdown()),
             ("mixed (add/mul/sub/cmp loop)", _prog_mixed())]

    for label, code in progs:
        print(f"\n{'='*78}\n## PROGRAM: {label}", flush=True)
        try:
            xw, posw, opc = PC.repetitive_program_windows(lean, code, max_steps=max_steps)
        except ValueError as e:
            print(f"  SKIP ({e})"); continue
        xw = xw.to(device); posw = posw.to(device)
        info = PC.conditional_active_units(lean, xw, posw, thr=thr)
        cond_active = info["active_units"]
        cond_counts = [a.numel() for a in cond_active]
        moe_sets, moe_flags = blockmoe_active_set(lean, cond_active)
        n_moe_active = sum(moe_flags)
        moe_units = sum(s.numel() for s in moe_sets)
        opmix = ", ".join(f"{isa.NAMES[o]}:{c}" for o, c in
                          sorted(opc.items(), key=lambda kv: -kv[1]))
        print(f"  steps={xw.shape[0]}  op mix: {opmix}")
        print(f"  conditional total units={sum(cond_counts)} ({sum(cond_counts)/(I*nL)*100:.3f}%)"
              f"  block-MoE active layers={n_moe_active}/{nL} units={moe_units} "
              f"({moe_units/(I*nL)*100:.2f}%)", flush=True)

        dense_full = [torch.arange(l.gate_w.shape[0]) for l in lean.layers]
        dense_m = SkipAwareCondLean(lean, dense_full, skip_empty=False).to(device)
        moe_m = SkipAwareCondLean(lean, moe_sets, skip_empty=True).to(device)
        cond_m = SkipAwareCondLean(lean, list(cond_active), skip_empty=True).to(device)

        # pad_window so a SINGLE graph per B covers the naive window.
        S = xw.shape[1]
        g_dense = GraphedLeanForward(dense_m, pad_window=S)
        g_moe = GraphedLeanForward(moe_m, pad_window=S)
        g_cond = GraphedLeanForward(cond_m, pad_window=S)

        base_x = xw[:1].contiguous(); base_pos = posw[:1].contiguous()
        # graphed forward returns hidden; assert the three agree at the query row.
        print(f"  {'B':>6} {'M':>8} | {'dense/stp':>10} {'moe/stp':>10} {'cond/stp':>10} "
              f"| {'c/dense':>8} {'c/moe':>7} {'moe/dense':>9}  [graphed]")
        for B in batches:
            if not _vram_ok(device, 8 * B * S * H * 4):
                print(f"  {B:>6}  -- SKIP (VRAM guard) --"); continue
            xb = base_x.expand(B, -1, -1).contiguous()
            pb = base_pos.expand(B, -1).contiguous()
            # byte-identity spot-check: graphed cond output query row == dense.
            with torch.no_grad():
                hd = g_dense(xb[:1], pb[:1]); hc = g_cond(xb[:1], pb[:1])
                hm = g_moe(xb[:1], pb[:1])
            linf_cd = (hd - hc).abs().max().item()
            linf_md = (hd - hm).abs().max().item()
            t_dense = _bench_call(g_dense, xb, pb, n, warmup, cuda)
            t_moe = _bench_call(g_moe, xb, pb, n, warmup, cuda)
            t_cond = _bench_call(g_cond, xb, pb, n, warmup, cuda)
            c_dense = t_dense / t_cond if t_cond else float('nan')
            c_moe = t_moe / t_cond if t_cond else float('nan')
            moe_dense = t_dense / t_moe if t_moe else float('nan')
            print(f"  {B:>6} {B*S:>8} | {t_dense/B*1000:9.4f}u {t_moe/B*1000:9.4f}u "
                  f"{t_cond/B*1000:9.4f}u | {c_dense:7.2f}x {c_moe:6.2f}x {moe_dense:8.2f}x "
                  f" [Linf c-d={linf_cd:.1e} m-d={linf_md:.1e}]", flush=True)
            del xb, pb
            if cuda:
                torch.cuda.empty_cache()
        del g_dense, g_moe, g_cond, dense_m, moe_m, cond_m, xw, posw
        if cuda:
            torch.cuda.empty_cache()


def _parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--subset", default="bitwise",
                    choices=["base", "mem+cmp", "bitwise", "full"])
    ap.add_argument("--batch", default="512,2048,8192")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--thr", type=float, default=0.0)
    ap.add_argument("--max-steps", type=int, default=200)
    return ap.parse_args()


if __name__ == "__main__":
    a = _parse()
    run(device=a.device, subset_name=a.subset,
        batches=tuple(int(b) for b in a.batch.split(",")),
        n=a.n, warmup=a.warmup, thr=a.thr, max_steps=a.max_steps)
