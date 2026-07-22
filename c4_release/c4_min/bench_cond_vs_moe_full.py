"""FULL-model (I=160465 divmod megablock) CONDITIONAL vs BLOCK-MoE net-win bench.

The FULL dense model is 51.6 GB and cannot live on a 24 GB GPU -- but the honest
conditional-vs-blockMoE question on the FULL model is really about the L9 divmod
MEGABLOCK (the only place block-MoE and conditional differ meaningfully):

  * NON-DIVMOD program: L9 fires ~14/160465 units -> BOTH block-MoE and conditional
    skip it (block-MoE routes the whole layer to identity == #628; conditional keeps
    ~14 dead units).  The two configs are then IDENTICAL on the megablock; their only
    difference is the small non-megablock layers (block-MoE keeps them dense,
    conditional prunes to the ~2 k firing union) -> the SAME ~1.6x small-layer win
    measured on mem+cmp.

  * DIVMOD program: L9 FIRES.  block-MoE runs the WHOLE 160465-unit layer (its route
    is at BLOCK granularity -- once the op is DIV/MOD it keeps the entire megablock);
    conditional keeps only the ~500 firing units.  THIS is the net-new win: within
    the active megablock, conditional does far less work than block-MoE.

This bench builds the FULL VM once (CPU, RSS-watchdog guarded), then for the busiest
layer (L9) times the FFN GEMM under three configs at spec batch sizes:
    dense  = 160465 units (streamed to GPU, one layer ~1.9 GB, fits)
    moe    = 160465 units on a divmod step / 0 (skipped) on a non-divmod step
    cond   = firing units only (~500 divmod / ~14 non-divmod, i.e. skipped)
The dense FFN weights are freed immediately after; only active blocks persist.

Run (RAM-guarded; needs ~54 GB host RAM free):
    python -m c4_min.bench_cond_vs_moe_full --device cuda:0 --max-rss 60
"""
from __future__ import annotations

import argparse
import time
from typing import List

import torch
import torch.nn.functional as F

from . import isa
from . import qwen_full_vm as Q
from . import qwen_lean_forward as LF
from . import perlayer_conditional_sparse as PC
from ._build_guard import RSSWatchdog


def _bench(fn, n, warmup, cuda):
    if cuda:
        torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000.0


def _vram_ok(device, need_gb, headroom_gb=2.0):
    if not device.startswith("cuda"):
        return True
    free, _ = torch.cuda.mem_get_info(torch.device(device))
    return need_gb + headroom_gb < free / (1024 ** 3)


def _prog_countdown():
    return isa.assemble([("IMM", 60), ("PSH", 0), ("IMM", 1), ("SUB", 0),
                         ("BNZ", 1), ("HALT", 0)])


def _prog_divmod():
    return isa.assemble([("IMM", 100), ("PSH", 0), ("IMM", 7), ("MOD", 0),
                         ("PSH", 0), ("IMM", 1), ("ADD", 0), ("PSH", 0),
                         ("IMM", 1), ("SUB", 0), ("BNZ", 1), ("HALT", 0)])


@torch.no_grad()
def _time_layer_ffn(lean_cpu, li, cond_idx, device, B, S, n, warmup, cuda):
    """Time layer-``li`` FFN GEMM (dense vs block-MoE vs conditional) at M=B*S.

    Returns (dense_ms, moe_ms, cond_ms, k_cond, I).  block-MoE on an ACTIVE layer
    keeps ALL I units (== dense weights, same GEMM), so moe_ms == dense_ms on a
    divmod step; the block-MoE saving comes ONLY from SKIPPING the whole layer on a
    non-divmod step (cond does the same via a ~14-unit block).
    """
    l = lean_cpu.layers[li]
    I = l.gate_w.shape[0]
    H = l.gate_w.shape[1]
    M = B * S
    xn = torch.randn(M, H, device=device, dtype=l.gate_w.dtype)

    # dense / block-MoE-active: stream the full [I,H] gate/up + [H,I] down to GPU.
    need_gb = 3 * I * H * 4 / (1024 ** 3) + M * I * 4 / (1024 ** 3)
    dense_ms = float("nan")
    if _vram_ok(device, need_gb):
        gate = l.gate_w.to(device); up = l.up_w.to(device); down = l.down_w.to(device)

        def dense_fwd():
            return F.linear(F.silu(F.linear(xn, gate)) * F.linear(xn, up), down)
        dense_ms = _bench(dense_fwd, n, warmup, cuda)
        del gate, up, down
        if cuda:
            torch.cuda.empty_cache()
    moe_ms = dense_ms  # block-MoE keeps the WHOLE layer when the op fires it

    # conditional: only the firing units.
    idx = cond_idx.cpu().long()
    if idx.numel() == 0:
        cond_ms = 0.0  # skipped entirely (identical to block-MoE's skip route)
    else:
        gate_c = l.gate_w[idx].contiguous().to(device)
        up_c = l.up_w[idx].contiguous().to(device)
        down_c = l.down_w[:, idx].contiguous().to(device)

        def cond_fwd():
            return F.linear(F.silu(F.linear(xn, gate_c)) * F.linear(xn, up_c), down_c)
        cond_ms = _bench(cond_fwd, n, warmup, cuda)
        del gate_c, up_c, down_c
    del xn
    if cuda:
        torch.cuda.empty_cache()
    return dense_ms, moe_ms, cond_ms, idx.numel(), I


def run(device="cuda:0", batches=(512, 2048, 8192), n=20, warmup=5, max_rss=60.0):
    cuda = device.startswith("cuda")
    print(f"# FULL cond-vs-MoE bench device={device} n={n} warmup={warmup} "
          f"RSS-guard={max_rss}GB", flush=True)
    wd = RSSWatchdog(max_gb=max_rss, hard_abort=True).start()
    try:
        print("# building FULL fused VM (I=160465, ~54 GB CPU RSS) ...", flush=True)
        vm = Q.build(code_size=24, subset=Q.SUBSET_FULL)
        lean_cpu = LF.LeanQwenVM.from_full_vm(vm, device="cpu")
    finally:
        pass
    H = lean_cpu.hidden_size
    I = lean_cpu.layers[0].gate_w.shape[0]
    nL = lean_cpu.n_layers
    dense_gb = sum(l.gate_w.numel() + l.up_w.numel() + l.down_w.numel()
                   for l in lean_cpu.layers) * 4 / (1024 ** 3)
    print(f"# n_layers={nL} H={H} I={I}  DENSE FFN wt = {dense_gb:.1f} GB "
          f"-> dense model INFEASIBLE on 24GB GPU", flush=True)
    wd.stop()  # build done; the dense weights live on CPU (fine)

    # locate the megablock layer (max I-firing on a divmod op).
    xdm, pdm = PC.op_window(lean_cpu, isa.MOD)
    info_mod = PC.conditional_active_units(lean_cpu, xdm, pdm.unsqueeze(0), thr=0.0)
    mega_li = int(max(range(nL), key=lambda i: info_mod["active_units"][i].numel()))
    print(f"# megablock layer = L{mega_li} "
          f"(MOD fires {info_mod['active_units'][mega_li].numel()}/{I} of its units)", flush=True)

    for label, code in (("countdown (no divmod: megablock DEAD)", _prog_countdown()),
                        ("divmod   (MOD loop: megablock FIRES)", _prog_divmod())):
        print(f"\n{'='*78}\n## PROGRAM: {label}", flush=True)
        xw, posw, opc = PC.repetitive_program_windows(lean_cpu, code, max_steps=200)
        info = PC.conditional_active_units(lean_cpu, xw, posw, thr=0.0)
        cond_active = info["active_units"]
        cond_counts = [a.numel() for a in cond_active]
        # block-MoE: active layer keeps all I; dead layer keeps 0.
        moe_active_layers = [i for i, c in enumerate(cond_counts) if c > 0]
        moe_units = sum(I if c > 0 else 0 for c in cond_counts)
        mega_cond = cond_counts[mega_li]
        mega_moe = I if mega_cond > 0 else 0
        opmix = ", ".join(f"{isa.NAMES[o]}:{k}" for o, k in
                          sorted(opc.items(), key=lambda kv: -kv[1]))
        print(f"  steps={xw.shape[0]}  ops[{opmix}]")
        print(f"  conditional total units = {sum(cond_counts)}/{I*nL} "
              f"({sum(cond_counts)/(I*nL)*100:.5f}% of dense)")
        print(f"  block-MoE active layers = {len(moe_active_layers)}/{nL}; "
              f"units = {moe_units}/{I*nL} ({moe_units/(I*nL)*100:.5f}% of dense)")
        print(f"  MEGABLOCK L{mega_li}: dense={I}  block-MoE={mega_moe}  "
              f"conditional={mega_cond}  "
              f"-> cond keeps {(mega_cond/mega_moe*100) if mega_moe else 0:.4f}% of MoE's megablock",
              flush=True)

        print(f"  MEGABLOCK L{mega_li} FFN GEMM ms/call (dense-streamed vs block-MoE vs cond):")
        print(f"    {'B':>6} {'M':>8} {'dense':>10} {'block-MoE':>10} {'cond':>10} "
              f"{'c/moe':>8} {'moe/dense':>9}")
        S = xw.shape[1]
        for B in batches:
            dms, mms, cms, k, _ = _time_layer_ffn(
                lean_cpu, mega_li, cond_active[mega_li], device, B, S, n, warmup, cuda)
            # block-MoE megablock cost: 0 if skipped (non-divmod), else dense.
            moe_cost = mms if mega_moe > 0 else 0.0
            cond_cost = cms
            c_over_moe = (moe_cost / cond_cost) if (cond_cost and moe_cost) else float("nan")
            m_over_d = (moe_cost / dms) if (dms == dms and dms) else float("nan")
            dstr = f"{dms:9.3f}m" if dms == dms else "   OOM   "
            mstr = f"{moe_cost:9.3f}m" if mega_moe > 0 else "  0(skip) "
            cstr = f"{cond_cost:9.4f}m" if cond_cost else "  0(skip) "
            print(f"    {B:>6} {B*S:>8} {dstr} {mstr} {cstr} "
                  f"{c_over_moe:7.1f}x {m_over_d:8.2f}x", flush=True)
        del xw, posw
        if cuda:
            torch.cuda.empty_cache()

    print("\n# NOTE: on a NON-divmod program both block-MoE AND conditional skip the "
          "megablock\n#       (identical); their only difference is the small "
          "non-megablock layers\n#       (the ~1.6x measured on mem+cmp).  On a DIVMOD "
          "program block-MoE must run\n#       the WHOLE megablock while conditional runs "
          "only the firing units -> the\n#       large c/moe factor above is the NET-NEW "
          "fine-grained win over block-MoE.", flush=True)


def _parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", default="512,2048,8192")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--max-rss", type=float, default=60.0)
    return ap.parse_args()


if __name__ == "__main__":
    a = _parse()
    run(device=a.device, batches=tuple(int(b) for b in a.batch.split(",")),
        n=a.n, warmup=a.warmup, max_rss=a.max_rss)
