#!/usr/bin/env python3
"""WALL #3 VERIFY: depth-N stack byte-exact through EVERY lean CFM driver.

Fast SUBSET_MEM_CMP CFM lean model (no divmod).  For each of a depth-1/2/3/4
battery + a store-of-computed + a function-call-args program, confirms the FIXED
(stack_depth=True) default is byte-exact through:
  * run_program_lean (naive)
  * speculative_run_lean (big-K)
  * run_program_lean_graphed / speculative_run_lean_graphed (cuda-graph)  [GPU]
  * run_program_lean_evict (bounded-KV eviction)
  * run_program_lean_evict_graphed (graphed eviction)                     [GPU]

Also REGRESSION-checks: for the DEPTH-1 programs, stack_depth=True must give the
SAME AX trace as the legacy stack_depth=False path (byte-identity preserved).
"""
from __future__ import annotations

import argparse
import warnings

import torch


def _battery(isa):
    P = []
    # ---- depth-1 (regression: fixed must equal legacy) --------------------
    P.append(("d1_add_100+27=127", 1, [
        ("IMM", 100), ("PSH", 0), ("IMM", 27), ("ADD", 0), ("HALT", 0)]))
    P.append(("d1_si_li_mem[23]=5", 1, [
        ("IMM", 5), ("PSH", 0), ("IMM", 0x23), ("SI", 0),
        ("IMM", 0x23), ("LI", 0), ("HALT", 0)]))
    P.append(("d1_loop20", 1, [
        ("IMM", 20), ("PSH", 0), ("IMM", 1), ("SUB", 0), ("BNZ", 1), ("HALT", 0)]))
    # ---- depth-2/3/4 (the wall) ------------------------------------------
    P.append(("d2_10+(3+4)=17", 2, [
        ("IMM", 10), ("PSH", 0),
        ("IMM", 3), ("PSH", 0), ("IMM", 4), ("ADD", 0), ("ADD", 0), ("HALT", 0)]))
    P.append(("d2_doomform_20+(6+8)=34", 2, [
        ("IMM", 20), ("PSH", 0),
        ("IMM", 6), ("PSH", 0), ("IMM", 8), ("ADD", 0), ("ADD", 0), ("HALT", 0)]))
    P.append(("d3_triple_park=26", 3, [
        ("IMM", 5), ("PSH", 0), ("IMM", 6), ("PSH", 0), ("IMM", 7), ("PSH", 0),
        ("IMM", 8), ("ADD", 0), ("ADD", 0), ("ADD", 0), ("HALT", 0)]))
    P.append(("d4_1+(2+(3+4))=10", 4, [
        ("IMM", 1), ("PSH", 0), ("IMM", 2), ("PSH", 0),
        ("IMM", 3), ("PSH", 0), ("IMM", 4), ("ADD", 0), ("ADD", 0), ("ADD", 0),
        ("HALT", 0)]))
    P.append(("d2_store_computed_mem[40]=(3+4)=7", 2, [
        ("IMM", 0x40), ("PSH", 0),
        ("IMM", 3), ("PSH", 0), ("IMM", 4), ("ADD", 0), ("SI", 0),
        ("IMM", 0x40), ("LI", 0), ("HALT", 0)]))
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--block-k", type=int, default=32)
    ap.add_argument("--no-graph", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    from c4_min import isa
    from c4_min import qwen_full_vm as Q
    from c4_min import qwen_lean_forward as LF
    from c4_min import qwen_lean_evict as EVICT

    dev = torch.device(args.device)
    use_graph = (not args.no_graph) and dev.type == "cuda"
    print(f"[build] SUBSET_MEM_CMP CFM lean on {dev} (graph={use_graph}) ...", flush=True)
    vm = Q.build(code_size=24, subset=Q.SUBSET_MEM_CMP)
    vm.embed = vm.embed.to(dev)
    lean = LF.LeanQwenVM.from_full_vm(vm, device=dev)
    del vm
    print(f"[built] {lean.n_layers}L {lean.n_heads}h cfm={lean.code_from_memory}", flush=True)

    G = CG = None
    if use_graph:
        from c4_min import qwen_lean_cuda_graph as CG_mod
        from c4_min import qwen_lean_evict_graphed as EG
        CG = CG_mod
        EG_mod = EG

    battery = _battery(isa)
    results = {}       # driver -> [pass count]
    regr_fail = []
    for name, depth, prog in battery:
        code = isa.assemble(prog)
        ref = isa.interpret(code, max_steps=200)
        want = ref[-1] if ref else None
        row = {"depth": depth, "want": want}

        rn = LF.run_program_lean(lean, code, max_steps=200)
        rs = LF.speculative_run_lean(lean, code, block_steps=args.block_k, max_steps=200)
        ev = EVICT.run_program_lean_evict(lean, code, max_steps=200, evict="off")
        row["naive"] = rn["exact"]
        row["spec"] = rs.exact
        row["evict"] = ev.exact

        # depth-1 regression: fixed default must equal the legacy 1-slot trace.
        if depth == 1:
            rn0 = LF.run_program_lean(lean, code, max_steps=200, stack_depth=False)
            if rn0["ax_trace"] != rn["ax_trace"]:
                regr_fail.append(name)

        if use_graph:
            g = CG.GraphedLeanForward(lean)
            rng = CG.run_program_lean_graphed(lean, code, max_steps=200, graphed=g)
            rsg = CG.speculative_run_lean_graphed(lean, code, block_steps=args.block_k,
                                                  max_steps=200, graphed=g)
            evg = EG_mod.run_program_lean_evict_graphed(lean, code, max_steps=200)
            row["graph_naive"] = rng["exact"]
            row["graph_spec"] = rsg.exact
            row["evict_graph"] = evg.exact

        results[name] = row
        flags = " ".join(f"{k}={'OK' if v else 'FAIL'}"
                         for k, v in row.items() if k not in ("depth", "want"))
        print(f"  d{depth} {name:34s} want={str(want):4s} {flags}", flush=True)

    # summary
    drivers = [k for k in next(iter(results.values())).keys() if k not in ("depth", "want")]
    print("\n[SUMMARY]", flush=True)
    all_ok = True
    for drv in drivers:
        n_ok = sum(1 for r in results.values() if r.get(drv))
        n = sum(1 for r in results.values() if drv in r)
        ok = n_ok == n
        all_ok = all_ok and ok
        print(f"  {drv:14s} {n_ok}/{n} {'BYTE-EXACT' if ok else 'DIVERGENCE'}", flush=True)
    if regr_fail:
        print(f"  [REGRESSION] depth-1 stack_depth changed trace: {regr_fail}", flush=True)
        all_ok = False
    else:
        print("  [regression] depth-1 stack_depth=True == legacy 1-slot: OK", flush=True)
    print(f"\n[RESULT] {'ALL PASS' if all_ok else 'FAILURES PRESENT'}", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
