#!/usr/bin/env python3
"""Decompose block-30 / logical-L19 PureFFN unit contributions to
OUTPUT_LO[15] / OUTPUT_HI[0] at the var_simple_12 step-3 PSH SP byte2 leak row
(pred_row=211). READ-ONLY, spec_k=0, hook-free.

Usage: python tools/probe_var_l19_byte2_ffn.py [id] [block] [pred_row]
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)
import torch
import torch.nn.functional as F
from tools.probe_groundtruth import build_groundtruth_probe
from src.compiler import compile_c
from tests.test_suite_1000 import generate_test_programs


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 262
    blk = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    pred_row = int(sys.argv[3]) if len(sys.argv) > 3 else 211
    tests = generate_test_programs()
    src, exp, desc = tests[idx]
    bc, data = compile_c(src)
    probe = build_groundtruth_probe()
    m = probe.model
    dp = m.dim_positions
    dev = next(m.parameters()).device
    ctx = probe._final_context(bc)
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)

    LO = dp["OUTPUT_LO"]
    HI = dp["OUTPUT_HI_THIS_STEP"]
    print(f"id={idx} {desc} block={blk} pred_row={pred_row}")
    print(f"OUTPUT_LO base={LO} OUTPUT_HI_THIS_STEP base={HI}")

    with torch.no_grad():
        # residual entering block `blk` = output after block blk-1
        x_in = m.forward(padded, stop_after_block=blk - 1)[0].float()  # [S,D]
        x_out = m.forward(padded, stop_after_block=blk)[0].float()
    row_in = x_in[pred_row]
    row_out = x_out[pred_row]
    print(f"\nresidual delta at this row (block {blk} adds):")
    for nm, base in (("OUTPUT_LO", LO), ("OUTPUT_HI", HI)):
        for k in (0, 15):
            d = base + k
            print(f"  {nm}[{k}] (dim {d}): in={float(row_in[d]):+.3f} "
                  f"out={float(row_out[d]):+.3f} "
                  f"delta={float(row_out[d]-row_in[d]):+.3f}")

    ffn = m.blocks[blk].ffn
    def dn(p):
        p = p.data
        if p.layout != torch.strided:
            p = p.to_dense()
        return p.float()
    Wu = dn(ffn.W_up); Wg = dn(ffn.W_gate); Wd = dn(ffn.W_down)
    bu = dn(ffn.b_up); bg = dn(ffn.b_gate); bd = dn(ffn.b_down).cpu()
    devw = Wu.device
    x = row_in.to(devw).float()
    up = (Wu @ x) + bu
    gate = (Wg @ x) + bg
    hidden = (F.silu(up) * gate).float()  # [H]
    H = hidden.shape[0]
    print(f"\nhidden_dim={H}")

    for d, tag in ((LO + 15, "OUTPUT_LO[15]"), (HI + 0, "OUTPUT_HI[0]")):
        import numpy as np
        contrib = (hidden * Wd[d]).float().cpu().numpy()  # per-unit contribution
        tot = float(contrib.sum()) + float(bd[d])
        print(f"\n=== contributions to {tag} (dim {d}); total={tot:+.3f} "
              f"(b_down={float(bd[d]):+.3f}) ===")
        order = np.argsort(-np.abs(contrib))
        hcpu = hidden.float().cpu().numpy()
        Wdcpu = Wd[d].float().cpu().numpy()
        bucpu = bu.float().cpu().numpy()
        bgcpu = bg.float().cpu().numpy()
        for h in order[:12].tolist():
            c = float(contrib[h])
            if abs(c) < 1e-4:
                break
            print(f"  unit {h:5d}: contrib={c:+.4f}  hidden={float(hcpu[h]):+.4f} "
                  f"W_down={float(Wdcpu[h]):+.4f}  W_up_b={float(bucpu[h]):+.3f} "
                  f"W_gate_b={float(bgcpu[h]):+.3f}")


if __name__ == "__main__":
    main()
