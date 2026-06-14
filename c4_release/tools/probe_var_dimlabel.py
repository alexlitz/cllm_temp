#!/usr/bin/env python3
"""Authoritatively label residual dims using the model's REAL widened layout
(compile_full_vm_dynamic returns it). Then re-run the leak decomposition with
correct names. Confirms whether dim100/dim85 are OUTPUT_HI+15 / OUTPUT_HI+0
and which block writes the ~1.145e9 leak.
"""
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
SRC = "int main() { int x; x = 990; return x; }"


def main():
    # Build layout once (separate small build, disk cache off to get layout).
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic)
    _m, layout = compile_full_vm_dynamic(disk_cache=False)
    dp = layout.dim_positions  # {name: start}
    # Build reverse map: dim index -> "NAME+k" using widths from sorted starts.
    items = sorted(dp.items(), key=lambda kv: kv[1])
    rev = {}
    for i, (name, start) in enumerate(items):
        end = items[i + 1][1] if i + 1 < len(items) else start + 1
        for d in range(start, end):
            rev[d] = f"{name}+{d-start}" if (end - start) > 1 else name
    print(f"d_model(layout) = {max(dp.values())+1}+  total names={len(dp)}")
    for d in (85, 100):
        print(f"  dim{d} = {rev.get(d,'?')}")

    # Now decompose with the probe model (separate, but same compile => same
    # layout) at the STACK0[0]=240 row.
    p = build_groundtruth_probe()
    plen = len(p._build_context(compile_c(SRC)[0]))
    model = p.model
    nblocks = len(model.blocks)
    bmap = {b["physical"]: b for b in p.block_layer_map()}
    bytecode = compile_c(SRC)[0]
    ctx = p._final_context(bytecode, max_steps=12)
    padded = torch.tensor([ctx], dtype=torch.long, device=p._device)
    resid = model.forward(padded, stop_after_block=nblocks - 1)[0]

    hw = model.head.weight.detach()
    if hw.layout != torch.strided:
        hw = hw.to_dense()
    hw = hw.float().cpu().contiguous()

    pred = plen + 35 + 20
    row = resid[pred].detach().float().cpu().contiguous()
    diff = (hw[240] * row - hw[0] * row).contiguous()
    order = torch.argsort(diff.abs(), descending=True)
    print(f"\n=== Top (logit[240]-logit[0]) contributors at pos{pred} ===")
    for d in order[:12].tolist():
        if abs(float(diff[d])) < 1.0:
            break
        print(f"  dim{d:4d} {rev.get(d,'?'):24s} resid={float(row[d]):.1f} "
              f"hw240={float(hw[240,d]):.1f} hw0={float(hw[0,d]):.1f} "
              f"dcontrib={float(diff[d]):.1f}")

    # Trace the top leak dim per block (which L-block writes the 1.145e9).
    top = [d for d in order[:4].tolist() if abs(float(diff[d])) > 1.0]
    print(f"\n=== Per-block residual at leak dims ===")
    for d in top:
        print(f"  --- dim{d} {rev.get(d,'?')} ---")
        prev = None
        for blk in range(nblocks):
            r = model.forward(padded, stop_after_block=blk)[0]
            v = round(float(r[pred][d]), 1)
            if v != prev:
                lg = bmap.get(blk, {}).get("logical", "?")
                print(f"     blk{blk:2d}(L{lg}): {v}")
                prev = v


if __name__ == "__main__":
    main()
