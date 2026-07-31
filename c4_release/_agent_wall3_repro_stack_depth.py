#!/usr/bin/env python3
"""WALL #3 REPRO: the #702 1-slot STACK0 depth wall on the lean CFM drivers.

Builds a fast SUBSET_MEM_CMP CFM lean model (ADD/SUB/PSH/mem — no divmod, so it
builds in seconds) and demonstrates:

  (1) the STOCK lean drivers (run_program_lean / draft_program_lean /
      speculative_run_lean) DIVERGE on depth-2+ stack expressions
      (e.g. 10 + (3+4) decodes 10 not 17; ((a+b)+c)+d wrong);
  (2) the STACK-AWARE drivers (qwen_lean_stack_driver.run_program_lean_stack /
      speculative_run_lean_stack) are BYTE-EXACT on the same programs.

This isolates the bug + the fix WITHOUT the 12 GB muldiv build.  The a*b+c*d
doom form (needs MUL) is verified separately on SUBSET_BITWISE / SUBSET_FULL.
"""
from __future__ import annotations

import argparse
import warnings

import torch


def _asm(isa, prog):
    return isa.assemble(prog)


# depth-2/3/4 stack expressions (the doom shape: compute, PSH, compute, ADD).
def _programs(isa):
    # 10 + (3 + 4): PSH 10, then (3+4) which itself PSHes -> 2 deep, then ADD.
    #   IMM 10 PSH  IMM 3 PSH  IMM 4 ADD (=7)  ADD (=17)
    P = []
    P.append(("depth2_10+(3+4)=17", [
        ("IMM", 10), ("PSH", 0),
        ("IMM", 3), ("PSH", 0), ("IMM", 4), ("ADD", 0),   # inner = 7 (pops 1 deep)
        ("ADD", 0),                                        # 10 + 7 = 17 (pops 2-deep park)
        ("HALT", 0)]))
    # ((a+b)+c)+d  left-assoc, 4-deep park pattern via nested inner exprs.
    #   a=1 b=2 c=3 d=4 -> 10.  Build as 1 + (2 + (3 + 4)) to force depth 4.
    P.append(("depth4_1+(2+(3+4))=10", [
        ("IMM", 1), ("PSH", 0),
        ("IMM", 2), ("PSH", 0),
        ("IMM", 3), ("PSH", 0), ("IMM", 4), ("ADD", 0),   # 7  (depth->2 after pop)
        ("ADD", 0),                                        # 2+7=9  (depth->1)
        ("ADD", 0),                                        # 1+9=10 (depth->0)
        ("HALT", 0)]))
    # a*b + c*d  DOOM FORM using ADD-only surrogate (no MUL in mem_cmp): use the
    # SAME 2-deep park shape a's product would need: (p) PSH (q) where q is itself
    # a 2-push sub-expr, then ADD.  p=20 q=(6+8)=14 -> 34.
    P.append(("doomform_20+(6+8)=34", [
        ("IMM", 20), ("PSH", 0),
        ("IMM", 6), ("PSH", 0), ("IMM", 8), ("ADD", 0),   # 14
        ("ADD", 0),                                        # 34
        ("HALT", 0)]))
    # 3-deep ADD chain reduced from a parked stack: PSH 5, PSH 6, PSH 7, ADD, ADD
    #   after IMM 8: stack=[5,6,7], ax=8;  ADD: 7+8=15 stack=[5,6];
    #   ADD: 6+15=21 stack=[5]; ADD: 5+21=26.
    P.append(("triple_park_5,6,7,+8_chain=26", [
        ("IMM", 5), ("PSH", 0),
        ("IMM", 6), ("PSH", 0),
        ("IMM", 7), ("PSH", 0),
        ("IMM", 8),
        ("ADD", 0), ("ADD", 0), ("ADD", 0),
        ("HALT", 0)]))
    # store of a COMPUTED (depth-2) value: mem[0x40] = (3 + 4); then LI it.
    #   IMM 0x40 PSH  IMM 3 PSH IMM 4 ADD (=7)  SI  ... IMM 0x40 LI -> 7
    P.append(("store_computed_mem[40]=(3+4)=7", [
        ("IMM", 0x40), ("PSH", 0),
        ("IMM", 3), ("PSH", 0), ("IMM", 4), ("ADD", 0),   # value = 7
        ("SI", 0),                                         # mem[0x40] = 7
        ("IMM", 0x40), ("LI", 0),
        ("HALT", 0)]))
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--block-k", type=int, default=64)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    from c4_min import isa
    from c4_min import qwen_full_vm as Q
    from c4_min import qwen_lean_forward as LF
    from c4_min import qwen_lean_stack_driver as SD

    dev = torch.device(args.device)
    print(f"[build] SUBSET_MEM_CMP CFM lean on {dev} ...", flush=True)
    vm = Q.build(code_size=24, subset=Q.SUBSET_MEM_CMP)
    vm.embed = vm.embed.to(dev)
    lean = LF.LeanQwenVM.from_full_vm(vm, device=dev)
    del vm
    print(f"[built] {lean.n_layers}L {lean.n_heads}h hidden={lean.hidden_size} "
          f"cfm={lean.code_from_memory}", flush=True)

    progs = _programs(isa)
    legacy_ok = fixed_ok = 0
    for name, prog in progs:
        code = _asm(isa, prog)
        ref = isa.interpret(code, max_steps=200)
        want = ref[-1] if ref else None
        # LEGACY 1-slot path (stack_depth=False) — should FAIL depth>1
        rn0 = LF.run_program_lean(lean, code, max_steps=200, stack_depth=False)
        rs0 = LF.speculative_run_lean(lean, code, block_steps=args.block_k,
                                      max_steps=200, stack_depth=False)
        # FIXED default path (stack_depth=True, the new default)
        rn = LF.run_program_lean(lean, code, max_steps=200)
        rs = LF.speculative_run_lean(lean, code, block_steps=args.block_k, max_steps=200)
        legacy_pass = rn0["exact"] and rs0.exact
        fixed_pass = rn["exact"] and rs.exact
        legacy_ok += legacy_pass
        fixed_ok += fixed_pass
        print(f"\n== {name}  (want last AX={want}) ==", flush=True)
        print(f"  LEGACY naive: exact={rn0['exact']}  ax_last={rn0['ax_trace'][-1] if rn0['ax_trace'] else None}", flush=True)
        print(f"  LEGACY spec : exact={rs0.exact}  ax_last={rs0.ax_trace[-1] if rs0.ax_trace else None}", flush=True)
        print(f"  FIXED  naive: exact={rn['exact']}  ax_last={rn['ax_trace'][-1] if rn['ax_trace'] else None}", flush=True)
        print(f"  FIXED  spec : exact={rs.exact}  ax_last={rs.ax_trace[-1] if rs.ax_trace else None}"
              f"  (fwd={rs.forwards}, {rs.speedup:.1f}x)", flush=True)

    print(f"\n[SUMMARY] legacy 1-slot pass {legacy_ok}/{len(progs)}  "
          f"| FIXED default pass {fixed_ok}/{len(progs)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
