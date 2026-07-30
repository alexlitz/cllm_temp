#!/usr/bin/env python3
"""MILESTONE 2: does the CFM LEAN model run byte-exact + what is the REAL ms/step?

Builds the CFM lean model (code_from_memory=True, fixed D) and:
  (a) verifies byte-exactness on a small battery via run_program_lean (naive) +
      speculative_run_lean (big-K) -- proves the CFM forward decodes correctly.
  (b) measures REAL ms/step: naive per-step, big-K speculative, and the
      graphed bounded-KV driver, on a deterministic loop (the closest analogue
      to doom's deterministic render).

This is on the SUBSET the program needs.  doom needs SUBSET_FULL (muldiv) which
is a 13 GB model; the small battery here uses SUBSET_MEM_CMP (fast) to prove the
CFM forward + fast levers, then reports the doom-specific walls separately.
"""
from __future__ import annotations

import argparse
import time
import warnings

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--subset", default="mem_cmp", choices=["mem_cmp", "full", "muldiv"])
    ap.add_argument("--spin-steps", type=int, default=2000)
    ap.add_argument("--block-k", type=int, default=64)
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    from c4_min import isa
    from c4_min import qwen_full_vm as Q
    from c4_min import qwen_lean_forward as LF

    dev = torch.device(args.device)
    subset = {"full": Q.SUBSET_FULL, "muldiv": Q.SUBSET_MULDIV,
              "mem_cmp": Q.SUBSET_MEM_CMP}[args.subset]
    recurrent = args.subset in ("full", "muldiv")

    t0 = time.perf_counter()
    # recurrent_divmod folds the 8 long-division iterations into ONE reused body,
    # so the muldiv model STORES ~39 distinct layers (applied 102x) not ~262.  Build
    # on CPU, then extract the DISTINCT layers to the GPU (build_lean_recurrent) — the
    # stock from_full_vm materialises 102 GPU copies of the 39 distinct layers (~31 GB)
    # and OOMs a 24 GB card; the recurrent extractor keeps ONE copy per distinct layer.
    vm = Q.build(code_size=24, subset=subset, recurrent_divmod=recurrent)
    if recurrent:
        from _agent_lean_recurrent import build_lean_recurrent
        lean = build_lean_recurrent(vm, device=str(dev))
    else:
        vm.embed = vm.embed.to(dev)
        lean = LF.LeanQwenVM.from_full_vm(vm, device=dev)
    del vm
    import gc
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    build_s = time.perf_counter() - t0
    if dev.type == "cuda":
        print(f"[vram] {torch.cuda.memory_allocated(dev)/1e9:.2f} GB allocated", flush=True)
    print(f"[built] CFM lean: {lean.n_layers}L {lean.n_heads}h hidden={lean.hidden_size} "
          f"code_from_memory={lean.code_from_memory} subset={subset.name} "
          f"build={build_s:.1f}s", flush=True)

    # -- (a) byte-exactness battery ------------------------------------------
    battery = [
        ("add", [("IMM", 100), ("PSH", 0), ("IMM", 27), ("ADD", 0), ("HALT", 0)]),
        ("mul", [("IMM", 12), ("PSH", 0), ("IMM", 7), ("MUL", 0), ("HALT", 0)]),
        ("div", [("IMM", 100), ("PSH", 0), ("IMM", 7), ("DIV", 0), ("HALT", 0)]),
        ("cmp_lt", [("IMM", 3), ("PSH", 0), ("IMM", 5), ("LT", 0), ("HALT", 0)]),
        ("loop20", [("IMM", 20), ("PSH", 0), ("IMM", 1), ("SUB", 0),
                    ("BNZ", 1), ("HALT", 0)]),
        ("si_li", [("IMM", 5), ("PSH", 0), ("IMM", 0x23), ("SI", 0),
                   ("IMM", 5), ("LI", 0), ("HALT", 0)]),
        # a function call (JSR/ENT/LEV) -- doom is function-heavy.
        ("jsr", [("JSR", 3), ("IMM", 5), ("HALT", 0), ("ENT", 0),
                 ("IMM", 7), ("LEV", 0)]),
    ]
    print("\n[verify] naive + big-K speculative byte-exact vs isa.interpret:", flush=True)
    all_ok = True
    for name, prog in battery:
        code = isa.assemble(prog)
        r = LF.run_program_lean(lean, code, max_steps=2000)
        spec = LF.speculative_run_lean(lean, code, block_steps=args.block_k,
                                       max_steps=2000)
        ok = r["exact"] and spec.exact
        all_ok = all_ok and ok
        print(f"  {name:8s} naive={'OK' if r['exact'] else 'FAIL'} "
              f"spec={'OK' if spec.exact else 'FAIL'} "
              f"steps={r['steps']} spec_forwards={spec.forwards} "
              f"spec_speedup={spec.speedup:.1f}x", flush=True)
    print(f"[verify] {'ALL BYTE-EXACT' if all_ok else 'DIVERGENCE'}", flush=True)

    # -- (b) real ms/step on a deterministic loop ----------------------------
    # a never-halting-until-N spin countdown: the closest small analogue to
    # doom's deterministic render (a long deterministic stretch, no I/O).
    n = args.spin_steps
    spin = isa.assemble([("IMM", 200), ("PSH", 0), ("IMM", 1), ("SUB", 0),
                         ("BNZ", 1), ("HALT", 0)])
    # NAIVE per-step (one lean forward per VM step, fresh window)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    rn = LF.run_program_lean(lean, spin, max_steps=n)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    naive_ms = (time.perf_counter() - t0) / max(rn["steps"], 1) * 1e3

    # BIG-K speculative (batched verify, block_steps rows per forward)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    rs = LF.speculative_run_lean(lean, spin, block_steps=args.block_k, max_steps=n)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    spec_wall = time.perf_counter() - t0
    spec_ms = spec_wall / max(rs.steps, 1) * 1e3

    print(f"\n[ms/step] deterministic countdown, {rn['steps']} steps:", flush=True)
    print(f"  naive per-step (fresh window) : {naive_ms:8.3f} ms/step", flush=True)
    print(f"  big-K speculative (K={args.block_k})   : {spec_ms:8.3f} ms/step "
          f"({rs.forwards} forwards, {rs.speedup:.1f}x fewer, exact={rs.exact})",
          flush=True)
    print(f"  big-K wall for {rs.steps} steps      : {spec_wall:.3f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
