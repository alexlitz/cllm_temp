"""FULL-subset (I=160465, muldiv) CONDITIONAL block sparsity bench.

The FULL dense model is ~51.6 GB (18 layers x 3 x [160465,1600] fp32) and does NOT
fit on a 24 GB GPU — so the DENSE baseline cannot even be materialised on cuda:0.
That is itself the headline: the CONDITIONAL per-layer block model (active units
only, ~470 units total => a few MB) is the ONLY runnable form of the FULL VM on a
single 24 GB GPU.

This bench, cuda:0, VRAM-guarded:
  * builds the dense weights on CPU (they fit in host RAM),
  * measures the per-op / per-program active-unit fraction (esp. whether MOD/DIV
    fires the L9 divmod megablock and everything else silu-gates it to 0),
  * builds the CONDITIONAL block model on GPU (fits) and times its full forward,
  * for a FAIR dense-vs-conditional KERNEL comparison on the busiest layer (L9,
    the [160465,1600] divmod megablock), streams the dense L9 gate/up/down onto
    the GPU one at a time and times the dense L9 FFN GEMM vs the conditional L9
    GEMM at the same batch — the dense number a 56 GB GPU would get per-forward.

Run:
    python -m c4_min.bench_conditional_full --device cuda:0
"""
from __future__ import annotations

import argparse
import time
from typing import Dict, List

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


def _vram_ok(device, need_gb, headroom_gb=3.0):
    if not device.startswith("cuda"):
        return True
    free, _ = torch.cuda.mem_get_info(torch.device(device))
    return need_gb + headroom_gb < free / (1024 ** 3)


def _prog_countdown(n=60):
    return isa.assemble([("IMM", n), ("PSH", 0), ("IMM", 1), ("SUB", 0),
                         ("BNZ", 1), ("HALT", 0)])


def _prog_divmod():
    return isa.assemble([("IMM", 100), ("PSH", 0), ("IMM", 7), ("MOD", 0),
                         ("PSH", 0), ("IMM", 1), ("ADD", 0), ("PSH", 0),
                         ("IMM", 1), ("SUB", 0), ("BNZ", 1), ("HALT", 0)])


@torch.no_grad()
def _busiest_layer_ffn(lean_cpu, active_L, li, device, B, S, n, warmup, cuda):
    """Stream dense layer-``li`` gate/up/down to GPU and time its FFN GEMM at M=B*S
    vs the conditional (active-units-only) GEMM.  Returns (dense_ms, cond_ms, k, I)."""
    l = lean_cpu.layers[li]
    I = l.gate_w.shape[0]
    H = l.gate_w.shape[1]
    M = B * S
    xn = torch.randn(M, H, device=device, dtype=l.gate_w.dtype)

    # dense: stream the full [I,H] gate/up + [H,I] down onto the GPU.
    need_gb = 3 * I * H * 4 / (1024 ** 3) + M * I * 4 / (1024 ** 3)
    dense_ms = float("nan")
    if _vram_ok(device, need_gb):
        gate = l.gate_w.to(device); up = l.up_w.to(device); down = l.down_w.to(device)

        def dense_fwd():
            return F.linear(F.silu(F.linear(xn, gate)) * F.linear(xn, up), down)
        dense_ms = _bench(dense_fwd, n, warmup, cuda)
        del gate, up, down
        torch.cuda.empty_cache()

    # conditional: only the active units (index on CPU where the weights live, then move).
    idx = active_L.cpu().long()
    gate_c = l.gate_w[idx].contiguous().to(device)
    up_c = l.up_w[idx].contiguous().to(device)
    down_c = l.down_w[:, idx].contiguous().to(device)

    def cond_fwd():
        return F.linear(F.silu(F.linear(xn, gate_c)) * F.linear(xn, up_c), down_c)
    cond_ms = _bench(cond_fwd, n, warmup, cuda)
    del gate_c, up_c, down_c, xn
    torch.cuda.empty_cache()
    return dense_ms, cond_ms, idx.numel(), I


def run(device="cuda:0", batches=(512, 2048, 8192, 16384), n=20, warmup=5, thr=0.0):
    cuda = device.startswith("cuda")
    print(f"# FULL conditional bench device={device} n={n} warmup={warmup}")
    print("# building FULL fused VM (I=160465) ...", flush=True)
    vm = Q.build(code_size=24, subset=Q.SUBSET_FULL)
    lean_cpu = LF.LeanQwenVM.from_full_vm(vm, device="cpu")
    H = lean_cpu.hidden_size
    I = lean_cpu.layers[0].gate_w.shape[0]
    nL = lean_cpu.n_layers
    dense_gb = sum(l.gate_w.numel() + l.up_w.numel() + l.down_w.numel()
                   for l in lean_cpu.layers) * 4 / (1024 ** 3)
    print(f"# n_layers={nL} H={H} I={I}  DENSE FFN weights = {dense_gb:.1f} GB "
          f"(cuda:0={torch.cuda.mem_get_info(torch.device(device))[1]/(1024**3):.0f}GB) "
          f"-> dense model {'FITS' if dense_gb < 20 else 'DOES NOT FIT on GPU'}")

    # ---- per-op active-unit fraction (does MOD/DIV fire L9?) -----------------
    print("\n## per-OP active units (total across layers; L9=divmod megablock)")
    for opn in ("IMM", "ADD", "SUB", "MUL", "MOD", "DIV", "PSH", "BNZ", "EQ"):
        op = isa.BY_NAME[opn]
        x, pos = PC.op_window(lean_cpu, op)
        info = PC.conditional_active_units(lean_cpu, x, pos.unsqueeze(0), thr=thr)
        c = [a.numel() for a in info["active_units"]]
        print(f"  {opn:4s}: total={sum(c):7d} ({sum(c)/(I*nL)*100:.5f}%)  "
              f"L9={c[9]:7d}/{I}  L13={c[13]:6d}")

    # ---- programs: repetitive (no divmod) vs divmod --------------------------
    for label, code in (("countdown (no divmod: L9 dead)", _prog_countdown()),
                        ("divmod (MOD loop: L9 fires)", _prog_divmod())):
        print(f"\n## PROGRAM: {label}")
        xw, posw, opc = PC.repetitive_program_windows(lean_cpu, code, max_steps=200)
        info = PC.conditional_active_units(lean_cpu, xw, posw, thr=thr)
        act = info["active_units"]
        c = [a.numel() for a in act]
        opmix = ", ".join(f"{isa.NAMES[o]}:{k}" for o, k in
                          sorted(opc.items(), key=lambda kv: -kv[1]))
        print(f"  steps={xw.shape[0]} ops[{opmix}]")
        print(f"  per-layer active: {c}")
        print(f"  total={sum(c)}/{I*nL} = {sum(c)/(I*nL)*100:.5f}% of dense; "
              f"L9 active={c[9]}/{I} = {c[9]/I*100:.5f}%")

        # conditional full forward on GPU (fits).
        cond = PC.ConditionalBlockLean(lean_cpu, act).to(device)
        S = xw.shape[1]
        base_x = xw[:1].to(device).contiguous()
        base_pos = posw[:1].to(device).contiguous()
        print(f"  CONDITIONAL full-forward (GPU): {'B':>6} {'M':>8} {'cond/fwd':>10} {'cond/stp':>10}")
        for B in batches:
            if not _vram_ok(device, 40 * B * S * H * 4 / (1024 ** 3), headroom_gb=2.0):
                print(f"    {B:>6}  -- SKIP (VRAM guard) --")
                continue
            xb = base_x.expand(B, -1, -1).contiguous()
            pb = base_pos.expand(B, -1).contiguous()
            tc = _bench(lambda: cond.forward(xb, q_positions=pb), n, warmup, cuda)
            print(f"                                  {B:>6} {B*S:>8} {tc:9.3f}m {tc/B*1000:9.4f}u")
            del xb, pb
            torch.cuda.empty_cache()

        # busiest-layer (L9) dense-vs-conditional KERNEL comparison.
        print(f"  BUSIEST-LAYER L9 FFN GEMM (dense-streamed vs conditional), S={S}:")
        print(f"    {'B':>6} {'M':>8} {'dense/L9':>10} {'cond/L9':>10} {'speedup':>9} {'k':>7}")
        for B in batches:
            dms, cms, k, _I = _busiest_layer_ffn(
                lean_cpu, act[9], 9, device, B, S, n, warmup, cuda)
            sp = (dms / cms) if (cms and dms == dms) else float("nan")
            dstr = f"{dms:9.3f}m" if dms == dms else "   OOM   "
            print(f"    {B:>6} {B*S:>8} {dstr} {cms:9.3f}m {sp:8.1f}x {k:>7}")
        del cond
        torch.cuda.empty_cache()


def _parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", default="512,2048,8192,16384")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--thr", type=float, default=0.0)
    return ap.parse_args()


if __name__ == "__main__":
    a = _parse()
    run(device=a.device, batches=tuple(int(b) for b in a.batch.split(",")),
        n=a.n, warmup=a.warmup, thr=a.thr)
