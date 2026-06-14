#!/usr/bin/env python3
"""Scan WHERE l16_ent_nested_stack0_saved_bp_byte0_f0 fires across programs
that have ENT frames (function call, var, recursion). For each program, report
every position where the rule's activation > 0, and whether OUTPUT_HI hi-nibble
is being forced to 0xF (its effect). Determines if the rule EVER fires
legitimately (genuine ENT-step STACK0 needing 0xF0) vs only misfires on the
OP_ENT broadcast (LEA / next-step STACK0 rows)."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import torch
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.compiler import compile_c
from tools.probe_groundtruth import build_groundtruth_probe


def main():
    p = build_groundtruth_probe()
    model = p.model
    L20 = next(b["physical"] for b in p.block_layer_map() if b["logical"] == 20)
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic)
    _m, layout = compile_full_vm_dynamic(disk_cache=False)
    dp = layout.dim_positions

    def D(nm):
        base, off = nm, 0
        if "+" in nm:
            base, o = nm.split("+")
            off = int(o)
        return dp[base] + off

    S = 100.0
    COND = [("OP_ENT", 10.0), ("MARK_STACK0", 10.0), ("HAS_SE", 1.0),
            ("MEM_STORE", 0.2), ("ADDR_B0_LO+8", -1.0), ("ADDR_B0_HI+14", -1.0),
            ("IS_BYTE", -1000.0), ("MARK_PC", -1000.0), ("MARK_AX", -1000.0),
            ("MARK_SP", -1000.0), ("MARK_BP", -1000.0), ("MARK_MEM", -1000.0)]

    def act(row):
        a = -80.0 * S
        for nm, w in COND:
            a += w * S * float(row[D(nm)])
        return a

    progs = {
        "var x=990": "int main() { int x; x = 990; return x; }",
        "func add":  "int add(int a,int b){return a+b;} int main(){return add(3,4);}",
        "var nested":"int f(int n){int y; y=n+1; return y;} "
                     "int main(){int x; x=f(5); return x;}",
        "fib rec":   "int fib(int n){if(n<2)return n;return fib(n-1)+fib(n-2);} "
                     "int main(){return fib(4);}",
    }
    OP_ENT = D("OP_ENT")
    MS = D("MARK_STACK0")
    for label, src in progs.items():
        try:
            bc = compile_c(src)[0]
        except Exception as e:
            print(f"\n{label}: compile failed {e}")
            continue
        plen = len(p._build_context(bc))
        try:
            ctx = p._final_context(bc, max_steps=40)
        except Exception as e:
            print(f"\n{label}: decode failed {e}")
            continue
        padded = torch.tensor([ctx], dtype=torch.long, device=p._device)
        xin = model.forward(padded, stop_after_block=L20 - 1)[0]
        fires = []
        for j in range(plen, xin.shape[0]):
            a = act(xin[j])
            if a > 0:
                fires.append((j - plen, a, float(xin[j][OP_ENT]),
                              float(xin[j][MS])))
        print(f"\n=== {label}: {len(fires)} firing position(s) ===")
        for rel, a, ent, ms in fires[:12]:
            print(f"   rel{rel:+4d}: act={a:.0f} OP_ENT={ent:.2f} "
                  f"MARK_STACK0={ms:.2f}")


if __name__ == "__main__":
    main()
