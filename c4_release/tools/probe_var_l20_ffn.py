#!/usr/bin/env python3
"""Find the L20 (block 32) FFN hidden unit(s) that write OUTPUT_HI+0/+15 on
the var LEA STACK0 row, and what input dims gate them. This identifies the
exact FFN rule to guard."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import torch
import torch.nn.functional as F
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.compiler import compile_c
from tools.probe_groundtruth import build_groundtruth_probe
SRC = "int main() { int x; x = 990; return x; }"
DIM0, DIM15 = 85, 100


def dense(w):
    return w.to_dense() if w.layout != torch.strided else w


def main():
    p = build_groundtruth_probe()
    model = p.model
    L20 = next(b["physical"] for b in p.block_layer_map() if b["logical"] == 20)
    bytecode = compile_c(SRC)[0]
    plen = len(p._build_context(bytecode))
    ctx = p._final_context(bytecode, max_steps=12)
    padded = torch.tensor([ctx], dtype=torch.long, device=p._device)
    pred = plen + 35 + 20

    # L20 input residual (after attn, == FFN input since attn delta ~0).
    xin = model.forward(padded, stop_after_block=L20 - 1)[0]  # [S,D]
    x = xin[pred].detach().float()  # [D]  FFN input row

    ffn = model.blocks[L20].ffn
    Wup = dense(ffn.W_up).detach().float()
    Wgate = dense(ffn.W_gate).detach().float()
    Wdown = dense(ffn.W_down).detach().float()
    bup = ffn.b_up.detach().float()
    bgate = ffn.b_gate.detach().float()

    up = Wup @ x + bup
    gate = Wgate @ x + bgate
    hidden = F.silu(up) * gate  # [H]

    # Which hidden units contribute to OUTPUT_HI+0 (dim85) and +15 (dim100)?
    # Wdown is [D, H]; contribution to out dim d = sum_h Wdown[d,h]*hidden[h].
    contrib0 = Wdown[DIM0] * hidden    # [H]
    contrib15 = Wdown[DIM15] * hidden  # [H]
    print("=== L20 FFN units writing OUTPUT_HI+15 (forces hi=F, wrong) ===")
    order = torch.argsort(contrib15.abs(), descending=True)
    for h in order[:8].tolist():
        if abs(float(contrib15[h])) < 1.0:
            break
        print(f"  unit{h:4d}: ->OUT_HI+15 {float(contrib15[h]):+.1f} "
              f"->OUT_HI+0 {float(contrib0[h]):+.1f}  "
              f"hidden={float(hidden[h]):.2f} up={float(up[h]):.2f} "
              f"gate={float(gate[h]):.2f}")

    print("\n=== L20 FFN units writing OUTPUT_HI+0 (suppresses hi=0) ===")
    order0 = torch.argsort(contrib0.abs(), descending=True)
    for h in order0[:8].tolist():
        if abs(float(contrib0[h])) < 1.0:
            break
        print(f"  unit{h:4d}: ->OUT_HI+0 {float(contrib0[h]):+.1f} "
              f"->OUT_HI+15 {float(contrib15[h]):+.1f}  "
              f"hidden={float(hidden[h]):.2f}")

    # For the top OUT_HI+15 unit, what input dims drive its gate/up?
    top = int(order[0])
    print(f"\n=== Top OUT_HI+15 unit {top}: input dims driving up & gate ===")
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic)
    _m, layout = compile_full_vm_dynamic(disk_cache=False)
    dp = layout.dim_positions
    items = sorted(dp.items(), key=lambda kv: kv[1])
    rev = {}
    for i, (name, start) in enumerate(items):
        end = items[i + 1][1] if i + 1 < len(items) else start + 1
        for d in range(start, end):
            rev[d] = f"{name}+{d-start}" if (end - start) > 1 else name
    print(f"  b_up[{top}]={float(bup[top]):.1f}  b_gate[{top}]={float(bgate[top]):.1f}")
    for label, W in (("up", Wup), ("gate", Wgate)):
        # ALL nonzero coefficients (the rule's full condition set), not just
        # the ones active on this row.
        nz = (W[top].abs() > 1e-6).nonzero(as_tuple=True)[0]
        parts = []
        for d in nz.tolist():
            parts.append(f"{rev.get(d,d)}(w={float(W[top,d]):+.1f},x={float(x[d]):.1f})")
        print(f"  {label} nonzero coeffs ({len(parts)}): " + "  ".join(parts[:24]))


if __name__ == "__main__":
    main()
