#!/usr/bin/env python3
"""WALL #3 DOOM FORM on the RECURRENT full-ISA MUL/DIV/MOD model (wall#2's path).

The non-recurrent SUBSET_BITWISE MUL is independently broken in the CFM lean
forward (a bare 12*7 decodes 7, PC stuck) — a wall#2-family divmod/MUL decode
issue ORTHOGONAL to the stack-depth wall.  The wall#2 fix was validated on the
RECURRENT divmod build (SUBSET_MULDIV recurrent, ~11.9 GB), so the real a*b+c*d
doom form must be shown there.

Builds that model (CPU bake -> GPU distinct-layer extract, per _agent_lean_recurrent)
and proves the depth-2 a*b+c*d / a*b-c*d / stored-fixed-point-product terms
byte-exact through the FIXED (stack_depth=True default) lean drivers vs isa.interpret.
"""
from __future__ import annotations

import argparse
import gc
import warnings

import torch


def _programs(isa):
    P = []
    P.append(("doom_a*b+c*d = 12*7+6*5 = 114", [
        ("IMM", 12), ("PSH", 0), ("IMM", 7), ("MUL", 0),
        ("PSH", 0),
        ("IMM", 6), ("PSH", 0), ("IMM", 5), ("MUL", 0),
        ("ADD", 0), ("HALT", 0)]))
    P.append(("doom_a*b-c*d = 9*8-5*6 = 42", [
        ("IMM", 9), ("PSH", 0), ("IMM", 8), ("MUL", 0),
        ("PSH", 0),
        ("IMM", 5), ("PSH", 0), ("IMM", 6), ("MUL", 0),
        ("SUB", 0), ("HALT", 0)]))
    # the raycaster >> FP form surrogate: (a*b) / d.  20*7=140, /8 = 17 (depth-2).
    P.append(("doom_(a*b)/d = (20*7)/8 = 17", [
        ("IMM", 20), ("PSH", 0), ("IMM", 7), ("MUL", 0),   # 140
        ("PSH", 0), ("IMM", 8), ("DIV", 0),                # 140 / 8 = 17
        ("HALT", 0)]))
    P.append(("store_doomterm mem[50]=(3*4+2*5)=22", [
        ("IMM", 0x50), ("PSH", 0),
        ("IMM", 3), ("PSH", 0), ("IMM", 4), ("MUL", 0),
        ("PSH", 0),
        ("IMM", 2), ("PSH", 0), ("IMM", 5), ("MUL", 0),
        ("ADD", 0),
        ("SI", 0),
        ("IMM", 0x50), ("LI", 0), ("HALT", 0)]))
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--block-k", type=int, default=16)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    from c4_min import isa
    from c4_min import qwen_full_vm as Q
    from c4_min import qwen_lean_forward as LF
    from _agent_lean_recurrent import build_lean_recurrent

    dev = torch.device(args.device)
    print(f"[build] SUBSET_MULDIV recurrent CFM lean -> {dev} (~11.9 GB) ...", flush=True)
    vm = Q.build(code_size=32, subset=Q.SUBSET_MULDIV, recurrent_divmod=True)
    lean = build_lean_recurrent(vm, device=str(dev))
    del vm
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        print(f"[vram] {torch.cuda.memory_allocated(dev)/1e9:.2f} GB", flush=True)
    print(f"[built] lean {lean.n_layers}L {lean.n_heads}h cfm={lean.code_from_memory} "
          f"eff_alu={lean.efficient_alu}", flush=True)

    ok = total = 0
    for name, prog in _programs(isa):
        code = isa.assemble(prog)
        ref = isa.interpret(code, max_steps=2000)
        rn = LF.run_program_lean(lean, code, max_steps=2000)
        rs = LF.speculative_run_lean(lean, code, block_steps=args.block_k, max_steps=2000)
        exact = rn["exact"] and rs.exact
        ok += exact
        total += 1
        print(f"\n== {name} ==", flush=True)
        print(f"  isa.interpret last AX = {ref[-1] if ref else None}", flush=True)
        print(f"  lean naive exact={rn['exact']}  last={rn['ax_trace'][-1] if rn['ax_trace'] else None}"
              f"  steps={rn['steps']}", flush=True)
        print(f"  lean spec  exact={rs.exact}  last={rs.ax_trace[-1] if rs.ax_trace else None}"
              f"  (fwd={rs.forwards}, {rs.speedup:.1f}x)", flush=True)

    print(f"\n[RESULT] {ok}/{total} REAL doom-form MUL/DIV terms byte-exact through lean drivers",
          flush=True)
    return 0 if ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
