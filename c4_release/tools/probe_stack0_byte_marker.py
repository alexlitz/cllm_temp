#!/usr/bin/env python3
"""What residual marker distinguishes the var LEA STACK0 byte rows (off21-24)
from a genuine ENT SP byte row? Dump H1+0..4, BYTE_INDEX, STACK0_BYTE* on the
var STACK0 byte rows AND on the var ENT SP byte row, so I can find a blocker
that vetoes the sp_byte1_ff / ent rules on STACK0 byte rows but is byte-
identical on the SP row."""
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
            base, o = nm.split("+"); off = int(o)
        return dp[base] + off

    track = (["IS_BYTE", "HAS_SE", "OP_ENT", "MARK_STACK0"]
             + [f"H1+{k}" for k in range(5)]
             + [f"H1.*.-1+{k}" for k in range(5)]
             + [f"BYTE_INDEX_{k}" for k in range(4)]
             + ["STACK0_BYTE0", "STACK0_BYTE1", "MEM_STORE"])

    bc = compile_c("int main() { int x; x = 990; return x; }")[0]
    plen = len(p._build_context(bc))
    ctx = p._final_context(bc, max_steps=12)
    padded = torch.tensor([ctx], dtype=torch.long, device=p._device)
    xin = model.forward(padded, stop_after_block=L20 - 1)[0]

    # var step1 (LEA): STACK0 byte rows off21..24; SP byte rows (step1 SP
    # section). step1 register layout: PC(0-4) AX MARK... we want SP byte0 row.
    # The SP section in a step is off ~5-9. Print step1 off19..28 marker tags.
    rows = [("LEA STACK0 off20(mk)", 1, 20), ("LEA STACK0[0] off21", 1, 21),
            ("LEA STACK0[1] off22", 1, 22), ("LEA STACK0[2] off23", 1, 23),
            ("ENT SP off? scan", 0, None)]
    base = plen + 35  # step1
    for lab, st, off in rows[:4]:
        r = xin[base + off]
        vals = "  ".join(f"{nm}={round(float(r[D(nm)]),2)}"
                         for nm in track if abs(float(r[D(nm)])) > 0.05)
        print(f"{lab}: {vals}")

    # Scan step0 (ENT) rows for the SP byte0 row (H1+2 high, IS_BYTE, BYTE_INDEX_0)
    print("\n--- step0 (ENT) SP byte rows (H1+2>0.5, IS_BYTE) ---")
    base0 = plen
    for off in range(35):
        r = xin[base0 + off]
        if float(r[D("H1+2")]) > 0.5 and float(r[D("IS_BYTE")]) > 0.5:
            vals = "  ".join(f"{nm}={round(float(r[D(nm)]),2)}"
                             for nm in track if abs(float(r[D(nm)])) > 0.05)
            print(f"  off{off}: {vals}")


if __name__ == "__main__":
    main()
