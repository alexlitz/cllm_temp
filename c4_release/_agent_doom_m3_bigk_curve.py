#!/usr/bin/env python3
"""MILESTONE 3: big-K speculative ms/step curve on the CFM lean model.

The deterministic-render model is: draft the whole deterministic stretch for free,
verify B steps per lean forward (big-K).  This sweeps K (block_steps) on a long
deterministic loop and reports effective ms/step — the fast-path headline for the
NON-divmod fraction of doom's render (the divmod fraction is the separate lean wall).

Uses SUBSET_MEM_CMP (fast, fits) for the lever measurement; the same big-K batching
applies to the muldiv model (11.9 GB) with the per-forward cost scaled by its 102 vs
10 layers (measured separately in M2).
"""
from __future__ import annotations

import argparse
import time
import warnings

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--ks", default="16,32,64,128,256")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    from c4_min import isa, qwen_full_vm as Q, qwen_lean_forward as LF
    dev = torch.device(args.device)
    vm = Q.build(code_size=24, subset=Q.SUBSET_MEM_CMP)
    vm.embed = vm.embed.to(dev)
    lean = LF.LeanQwenVM.from_full_vm(vm, device=dev)
    del vm
    print(f"[built] CFM lean {lean.n_layers}L hidden={lean.hidden_size}", flush=True)

    # a long deterministic countdown (no I/O): the render analogue.  4n+2 steps.
    n = args.steps
    prog = isa.assemble([("IMM", 250), ("PSH", 0), ("IMM", 1), ("SUB", 0),
                         ("BNZ", 1), ("HALT", 0)])

    # naive per-step baseline (a handful of steps to get ms/step; O(1) window)
    _ = LF.run_program_lean(lean, prog, max_steps=20)   # warmup
    if dev.type == "cuda": torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    rn = LF.run_program_lean(lean, prog, max_steps=200)
    if dev.type == "cuda": torch.cuda.synchronize(dev)
    naive_ms = (time.perf_counter() - t0) / max(rn["steps"], 1) * 1e3
    print(f"\n  naive per-step (fresh window)     : {naive_ms:8.3f} ms/step\n", flush=True)

    print(f"  {'K':>5} {'forwards':>9} {'fewer':>7} {'ms/step':>9} {'wall_s':>8} "
          f"{'exact':>6}", flush=True)
    print("  " + "-" * 52, flush=True)
    best = None
    for K in [int(k) for k in args.ks.split(",")]:
        _ = LF.speculative_run_lean(lean, prog, block_steps=K, max_steps=50)  # warmup
        if dev.type == "cuda": torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        r = LF.speculative_run_lean(lean, prog, block_steps=K, max_steps=n)
        if dev.type == "cuda": torch.cuda.synchronize(dev)
        wall = time.perf_counter() - t0
        ms = wall / max(r.steps, 1) * 1e3
        print(f"  {K:>5} {r.forwards:>9} {r.speedup:>6.1f}x {ms:>9.3f} {wall:>8.2f} "
              f"{'OK' if r.exact else 'FAIL':>6}", flush=True)
        if best is None or ms < best[1]:
            best = (K, ms, r.steps)
    print(f"\n  BEST: K={best[0]}  {best[1]:.3f} ms/step over {best[2]} steps  "
          f"({naive_ms/best[1]:.1f}x vs naive; {best[1]/1.0:.2f}x the 1 ms mark)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
