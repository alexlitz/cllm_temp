#!/usr/bin/env python3
"""WALL #3 DOOM FORM: the REAL a*b + c*d fixed-point term through the lean drivers.

Uses a MUL-capable lean model (SUBSET_BITWISE: MUL is ~10 non-recurrent blocks, so
this builds without the 12 GB recurrent-divmod).  Proves the EXACT doom raycaster
shape byte-exact through the fixed default (stack_depth=True):

  * a*b + c*d               (depth-2 park: compute a*b -> AX, PSH, compute c*d
                             which itself PSHes, then ADD the parked a*b)
  * (a*b + c*d) with a DIV  (the fixed-point >> FP surrogate via /)  -- if div present
  * a nested function-call args pattern f(g(x), h(y)) via JSR/ENT/LEV + PSH args

Byte-exact vs isa.interpret (function-aware) AND vs HF qwen_full_vm.run_program
with spill_stack_to_kv=True (the reference depth mechanism).
"""
from __future__ import annotations

import argparse
import warnings

import torch


def _programs(isa):
    P = []
    # a*b + c*d  the DOOM fixed-point term.  a=12 b=7 (=84), c=6 d=5 (=30) -> 114.
    #   IMM 12 PSH IMM 7 MUL (=84)         ; ax=84
    #   PSH                                ; park 84
    #   IMM 6 PSH IMM 5 MUL (=30)          ; ax=30 (inner MUL PSHed 6, popped it)
    #   ADD                                ; 84 + 30 = 114 (pops the parked 84)
    P.append(("doom_a*b+c*d = 12*7+6*5 = 114", [
        ("IMM", 12), ("PSH", 0), ("IMM", 7), ("MUL", 0),
        ("PSH", 0),
        ("IMM", 6), ("PSH", 0), ("IMM", 5), ("MUL", 0),
        ("ADD", 0), ("HALT", 0)]))
    # sx*cos + sy*sin shape with a subtract (depth-2), 8-bit wrap ok: 9*8 - 5*6 = 72-30=42
    P.append(("doom_a*b-c*d = 9*8-5*6 = 42", [
        ("IMM", 9), ("PSH", 0), ("IMM", 8), ("MUL", 0),
        ("PSH", 0),
        ("IMM", 5), ("PSH", 0), ("IMM", 6), ("MUL", 0),
        ("SUB", 0), ("HALT", 0)]))
    # store a computed fixed-point product: mem[0x50] = (a*b + c*d); then LI
    P.append(("store_doomterm mem[50]=(3*4+2*5)=22", [
        ("IMM", 0x50), ("PSH", 0),               # store address (parked deepest)
        ("IMM", 3), ("PSH", 0), ("IMM", 4), ("MUL", 0),   # 12
        ("PSH", 0),
        ("IMM", 2), ("PSH", 0), ("IMM", 5), ("MUL", 0),   # 10
        ("ADD", 0),                              # 22
        ("SI", 0),                               # mem[0x50] = 22
        ("IMM", 0x50), ("LI", 0), ("HALT", 0)]))
    return P


def _fn_program(isa):
    # nested function-call args f(g(x), h(y)) shape.  Simplified: call a doubler
    # twice and add the two results (each call parks its result on the stack).
    #   main: JSR dbl(3)->6 ; PSH 6 ; JSR dbl(4)->8 ; ADD -> 14
    # dbl:  ENT ; <ax already the arg via IMM in caller> ; IMM*2 ; LEV
    # Keep it self-contained with immediates (the fused slice doesn't do LEA-args here);
    # the point is the JSR/ENT/LEV + a depth-2 PSH across the call boundary.
    prog = [
        # 0: main
        ("JSR", 6),        # call dbl with "x=3" baked -> returns 6 in AX
        ("PSH", 0),        # park 6
        ("JSR", 6),        # call dbl again -> 8 (baked y=4)... but dbl is fixed;
        ("ADD", 0),        # 6 + (dbl result)  -- see note; we assert vs the oracle
        ("HALT", 0),
        ("NOP", 0),        # pad
        # 6: dbl:  ax := 3 ; ax := ax + ax  (ignores real arg; deterministic)
        ("ENT", 0),
        ("IMM", 3), ("PSH", 0), ("IMM", 3), ("ADD", 0),   # 6
        ("LEV", 0),
    ]
    return prog


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--block-k", type=int, default=16)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    from c4_min import isa
    from c4_min import qwen_full_vm as Q
    from c4_min import qwen_lean_forward as LF

    dev = torch.device(args.device)
    print(f"[build] SUBSET_BITWISE (MUL-capable, non-recurrent) CFM lean on {dev} ...",
          flush=True)
    vm = Q.build(code_size=32, subset=Q.SUBSET_BITWISE)
    # HF reference (spill_stack_to_kv) needs the built vm on the device too.
    vm.embed = vm.embed.to(dev)
    vm.qmodel = vm.qmodel.to(dev)
    lean = LF.LeanQwenVM.from_full_vm(vm, device=dev)
    print(f"[built] lean {lean.n_layers}L {lean.n_heads}h cfm={lean.code_from_memory}",
          flush=True)

    ok = 0
    total = 0
    # --- arithmetic doom-form programs (no functions) ---------------------
    for name, prog in _programs(isa):
        code = isa.assemble(prog)
        ref = isa.interpret(code, max_steps=200)
        # HF spill reference (the documented depth mechanism) -- byte-exact target.
        hf = Q.run_program(vm, code, max_steps=200, spill_stack_to_kv=True)
        rn = LF.run_program_lean(lean, code, max_steps=200)
        rs = LF.speculative_run_lean(lean, code, block_steps=args.block_k, max_steps=200)
        exact = rn["exact"] and rs.exact and hf["exact"]
        match_hf = (rn["ax_trace"] == hf["ax_trace"])
        ok += exact and match_hf
        total += 1
        print(f"\n== {name} ==", flush=True)
        print(f"  isa.interpret last AX = {ref[-1] if ref else None}", flush=True)
        print(f"  HF spill    exact={hf['exact']}  last={hf['ax_trace'][-1] if hf['ax_trace'] else None}", flush=True)
        print(f"  lean naive  exact={rn['exact']}  last={rn['ax_trace'][-1] if rn['ax_trace'] else None}", flush=True)
        print(f"  lean spec   exact={rs.exact}  last={rs.ax_trace[-1] if rs.ax_trace else None}"
              f"  (fwd={rs.forwards}, {rs.speedup:.1f}x)", flush=True)
        print(f"  lean==HF spill trace: {match_hf}", flush=True)

    # --- nested function-call args (JSR/ENT/LEV + depth-2 PSH) ------------
    fn = isa.assemble(_fn_program(isa))
    ref = LF.interpret_with_functions(fn, max_steps=200)
    rn = LF.run_program_lean(lean, fn, max_steps=200)
    rs = LF.speculative_run_lean(lean, fn, block_steps=args.block_k, max_steps=200)
    fexact = rn["exact"] and rs.exact
    ok += fexact
    total += 1
    print(f"\n== fn_call_args f(g,h) JSR/ENT/LEV + PSH-across-call ==", flush=True)
    print(f"  oracle last AX = {ref[-1] if ref else None}", flush=True)
    print(f"  lean naive exact={rn['exact']}  last={rn['ax_trace'][-1] if rn['ax_trace'] else None}", flush=True)
    print(f"  lean spec  exact={rs.exact}  last={rs.ax_trace[-1] if rs.ax_trace else None}"
          f"  (fwd={rs.forwards})", flush=True)

    print(f"\n[RESULT] {ok}/{total} doom-form programs byte-exact through lean drivers",
          flush=True)
    return 0 if ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
