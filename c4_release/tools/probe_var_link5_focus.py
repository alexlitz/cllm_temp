#!/usr/bin/env python3
"""Focused diagnostic for var_simple_12 (id 262) links 5+6.

Link A: step-3 PSH SP byte3 = 0x04 leak (want 0x00).
Link B: step-4 IMM PC byte0 = 0x42 (want 0x3a).

Dumps the raw final context around the SP/PC marker rows, the per-block
residual walk on the EXACT prediction rows, and the dominant OUTPUT / LM-head
dims so we can name the writer block/op.

READ-ONLY. spec_k=0, hook-free. Dims via probe.model.dim_positions.

Usage:  python tools/probe_var_link5_focus.py
"""
from __future__ import annotations
import os, sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ.setdefault("PYTHONUNBUFFERED", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import torch  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402
from tests.test_suite_1000 import generate_test_programs  # noqa: E402
from src.compiler import compile_c  # noqa: E402

MARKERS = {257: "REG_PC", 258: "REG_AX", 259: "REG_SP", 260: "REG_BP",
           262: "STEP_END", 263: "HALT"}


def main():
    idx = 262
    tests = generate_test_programs()
    src, exp, desc = tests[idx]
    bc, data = compile_c(src)

    probe = build_groundtruth_probe()
    m = probe.model
    dp = m.dim_positions
    dev = next(m.parameters()).device
    bl = probe.block_layer_map()
    ctx = probe._final_context(bc)
    prompt_len = len(probe._build_context(bc))
    print(f"prompt_len={prompt_len} total_ctx={len(ctx)}", flush=True)

    print("\n=== marker map (pos : token : decode) ===")
    i = prompt_len
    step = 0
    sp_pos = {}; pc_pos = {}
    while i < len(ctx):
        t = ctx[i]
        nm = MARKERS.get(t)
        if nm in ("REG_PC", "REG_AX", "REG_SP", "REG_BP"):
            bs = [ctx[i + 1 + j] & 0xFF for j in range(4)]
            val = sum(bs[j] << (8 * j) for j in range(4))
            print(f"  step{step} pos={i:3d} {nm}: bytes={bs} val={val} (0x{val:x})")
            if nm == "REG_SP":
                sp_pos[step] = i
            if nm == "REG_PC":
                pc_pos[step] = i
            i += 5
        elif nm == "STEP_END":
            print(f"  step{step} pos={i:3d} STEP_END")
            step += 1
            i += 1
        elif nm == "HALT":
            print(f"  pos={i:3d} HALT")
            break
        else:
            i += 1

    padded = torch.tensor([ctx], dtype=torch.long, device=dev)

    rev = {d: n for n, d in dp.items() if isinstance(d, int)}

    def topdims(row, k=8):
        v = row.abs()
        idx = torch.topk(v, k).indices.tolist()
        return [(rev.get(d, f"d{d}"), d, round(float(row[d]), 2)) for d in idx]

    a_byte_pos = sp_pos[3] + 1 + 3
    a_pred = a_byte_pos - 1
    b_byte_pos = pc_pos[4] + 1 + 0
    b_pred = b_byte_pos - 1
    print(f"\nLink A: SP step3 marker@{sp_pos[3]} byte3@{a_byte_pos} "
          f"pred_row={a_pred} ctx[byte]={ctx[a_byte_pos]}")
    print(f"Link B: PC step4 marker@{pc_pos[4]} byte0@{b_byte_pos} "
          f"pred_row={b_pred} ctx[byte]={ctx[b_byte_pos]}")

    rows = {"A_sp_b3": a_pred, "B_pc_b0": b_pred}
    cache = {}
    with torch.no_grad():
        for phys in range(len(m.blocks)):
            r = m.forward(padded, stop_after_block=phys)[0].float()
            cache[phys] = {nm: r[pr].clone() for nm, pr in rows.items()}
        final_logits = m.forward(padded)[0].float()

    for nm, pr in rows.items():
        print(f"\n########## {nm} pred_row={pr} ##########")
        logits = final_logits[pr]
        topk = torch.topk(logits, 8)
        print("LM top8:", [(int(t), round(float(v), 2))
                           for v, t in zip(topk.values, topk.indices)])
        print("per-block top-|dim| (only printed when set changes):")
        prev = None
        for phys in range(len(m.blocks)):
            row = cache[phys][nm]
            td = topdims(row, 6)
            cur = str([(n, round(v, 1)) for n, d, v in td])
            if cur != prev:
                print(f"  blk{phys:2d}/L{bl[phys]['logical']:2d}: {td}")
                prev = cur


if __name__ == "__main__":
    main()
