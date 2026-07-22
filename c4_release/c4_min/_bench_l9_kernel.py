"""Focused L9 (divmod megablock) dense-vs-conditional FFN GEMM timing — the
load-bearing TASK-5 kernel comparison for the FULL subset.

L9 is the [160465,1600] base-16 long-division megablock.  For a non-divmod step
only ~14-663 of its 160465 units fire; a divmod step fires ~2185.  This times the
DENSE L9 FFN GEMM (streamed onto the GPU) vs the CONDITIONAL (active-units-only)
GEMM at the real speculation batch shapes, and reports where dense OOMs (the
[M,160465] intermediate) so the memory story is explicit.

===========================================================================
MEASURED (RTX A5000, cuda:0, fp32, TF32 OFF, n=20 warmup=5).  L9 FFN GEMM
ms/call, dense-streamed vs conditional (active-units-only):

  B      M       dense/L9      countdown/L9(k=502)   divmod/L9(k=2185)   speedup
  512    3584    483.83 ms     1.28 ms               6.12 ms             379x / 79x
  2048   14336   OOM(26 GB)    5.45 ms               23.80 ms            dense OOM
  4096   28672   OOM(51 GB)    10.43 ms              46.26 ms            dense OOM
  8192   57344   OOM(103 GB)   22.70 ms              92.65 ms            dense OOM
  16384  114688  OOM(206 GB)   42.48 ms              187.29 ms           dense OOM

The dense L9 GEMM materialises an [M,160465] intermediate (26-206 GB at
B>=2048) and cannot run on a 24 GB GPU past B=512; the conditional GEMM
(gather the ~502/2185 active units => a [k,1600] dense block) does the
work in a few ms and scales linearly.  This is the CONDITIONAL win the
static BSR could never see: the divmod megablock's WEIGHTS are all nonzero
(BSR is 100% dense there), but its ACTIVATIONS are ~0 for a non-divmod step.
===========================================================================
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from . import isa
from . import qwen_full_vm as Q
from . import qwen_lean_forward as LF
from . import perlayer_conditional_sparse as PC


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


def _fits(device, need_gb, headroom=2.0):
    free, _ = torch.cuda.mem_get_info(torch.device(device))
    return need_gb + headroom < free / (1024 ** 3)


def run(device="cuda:0", batches=(512, 2048, 4096, 8192), n=20, warmup=5):
    cuda = device.startswith("cuda")
    print(f"# L9 kernel bench device={device} n={n} warmup={warmup}", flush=True)
    vm = Q.build(code_size=24, subset=Q.SUBSET_FULL)
    lean = LF.LeanQwenVM.from_full_vm(vm, device="cpu")
    l9 = lean.layers[9]
    I, H = l9.gate_w.shape
    S = 7
    print(f"# L9 gate/up=[{I},{H}] down=[{H},{I}]  dense L9 FFN = "
          f"{(2*I*H + H*I)*4/(1024**3):.2f} GB", flush=True)

    # active-unit sets for a non-divmod (countdown) and a divmod program.
    prog = {
        "countdown(L9 dead)": isa.assemble([
            ("IMM", 60), ("PSH", 0), ("IMM", 1), ("SUB", 0), ("BNZ", 1), ("HALT", 0)]),
        "divmod(L9 fires)": isa.assemble([
            ("IMM", 100), ("PSH", 0), ("IMM", 7), ("MOD", 0), ("PSH", 0),
            ("IMM", 1), ("ADD", 0), ("PSH", 0), ("IMM", 1), ("SUB", 0),
            ("BNZ", 1), ("HALT", 0)]),
    }
    l9_active = {}
    for label, code in prog.items():
        xw, posw, _ = PC.repetitive_program_windows(lean, code, max_steps=150)
        info = PC.conditional_active_units(lean, xw, posw, thr=0.0)
        l9_active[label] = info["active_units"][9]
        print(f"# {label}: L9 active = {l9_active[label].numel()}/{I}", flush=True)

    # dense L9 weights (streamed to GPU).
    gate = l9.gate_w.to(device); up = l9.up_w.to(device); down = l9.down_w.to(device)
    print(f"\n{'B':>6} {'M':>8} | {'dense/L9':>11} | " +
          " | ".join(f"{lab.split('(')[0]+'/L9':>18}" for lab in prog) + " | speedups")
    for B in batches:
        M = B * S
        xn = torch.randn(M, H, device=device, dtype=gate.dtype)
        # dense needs ~3 live [M,I] intermediates (silu(gate), up, product) + the
        # down output — guard for ~3x the [M,I] footprint plus headroom.
        need_dense = 3 * M * I * 4 / (1024 ** 3)
        if _fits(device, need_dense, headroom=1.5):
            dms = _bench(lambda: F.linear(F.silu(F.linear(xn, gate)) * F.linear(xn, up), down),
                         n, warmup, cuda)
            dstr = f"{dms:10.3f}m"
        else:
            dms = float("nan"); dstr = f"OOM({need_dense:.0f}GB)"
        cells = []
        sps = []
        for label in prog:
            idx = l9_active[label].cpu().long()
            gc = gate[idx].contiguous(); uc = up[idx].contiguous(); dc = down[:, idx].contiguous()
            cms = _bench(lambda: F.linear(F.silu(F.linear(xn, gc)) * F.linear(xn, uc), dc),
                         n, warmup, cuda)
            cells.append(f"{cms:10.4f}m(k={idx.numel()})")
            sps.append(f"{label.split('(')[0]}:{(dms/cms):.0f}x" if dms == dms else f"{label.split('(')[0]}:n/a")
            del gc, uc, dc
        print(f"{B:>6} {M:>8} | {dstr:>11} | " + " | ".join(f"{c:>18}" for c in cells) +
              " | " + " ".join(sps), flush=True)
        del xn
        torch.cuda.empty_cache()


def _parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", default="512,2048,4096,8192")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    return ap.parse_args()


if __name__ == "__main__":
    a = _parse()
    run(device=a.device, batches=tuple(int(b) for b in a.batch.split(",")),
        n=a.n, warmup=a.warmup)
