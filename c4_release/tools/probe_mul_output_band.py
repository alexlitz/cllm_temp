#!/usr/bin/env python3
"""Dump OUTPUT_LO/HI (+MUL_RESULT_HI) band AFTER the L11 wide_mul fires.

Reads the residual at the OUTPUT of the L11 physical block for mul cases,
showing which product one-hot lanes the wide_mul FFN lit and at what
magnitude -- so we can see the artifact-vs-true-lane competition.

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. [C4_MUL_WIDTH2=1] python tools/probe_mul_output_band.py
"""
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import torch

from neural_vm.embedding import Opcode
from tools.probe_groundtruth import build_groundtruth_probe


def _mk(ops):
    bc = []
    for op in ops:
        if isinstance(op, tuple):
            opcode, imm = op
            bc.append(opcode | (imm << 8))
        else:
            bc.append(op)
    return bc


def prog(a, b):
    return _mk([(Opcode.IMM, a), Opcode.PSH, (Opcode.IMM, b), Opcode.MUL, Opcode.EXIT])


def band(row, dp, name, width=16, thr=0.005):
    base = dp.get(name)
    if base is None:
        return None
    cells = [round(float(row[base + i].item()), 4) for i in range(width)]
    return [(i, v) for i, v in enumerate(cells) if abs(v) > thr]


def main():
    w2 = os.environ.get("C4_MUL_WIDTH2") == "1"
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    ax_base = dp["MARK_AX"]
    se_base = dp["MARK_SE_ONLY"]

    l11_phys = None
    for phys, blk in enumerate(model.blocks):
        if getattr(blk, "_logical_layer", phys) == 11:
            l11_phys = phys
            break
    print(f"width2={w2}  L11 physical block = {l11_phys}")

    CASES = [(6, 7), (100, 5)]
    for a, b in CASES:
        bc = prog(a, b)
        ctx = probe._final_context(bc, max_steps=20)
        S = len(ctx)
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        opmul_base = dp["OP_MUL"]
        with torch.no_grad():
            r_in = model.forward(toks, stop_after_block=l11_phys - 1)[0]
        # MUL row = AX row with OP_MUL hot in the L11-input residual.
        cand = [r for r in range(S)
                if r_in[r, ax_base].abs().item() > 0.5
                and r_in[r, opmul_base].item() > 0.5]
        ax_row = cand[-1] if cand else None
        with torch.no_grad():
            r_out = model.forward(toks, stop_after_block=l11_phys)[0]
        exp = a * b
        print(f"=== A={a} B={b}  expect product={exp}=0x{exp:04x} ax_row={ax_row} ===")
        print(f"    OUTPUT_LO     ={band(r_out[ax_row],dp,'OUTPUT_LO')}")
        print(f"    OUTPUT_HI     ={band(r_out[ax_row],dp,'OUTPUT_HI')}")
        if w2:
            print(f"    MUL_RESULT_HI_LO={band(r_out[ax_row],dp,'MUL_RESULT_HI_LO')}")
            print(f"    MUL_RESULT_HI_HI={band(r_out[ax_row],dp,'MUL_RESULT_HI_HI')}")
        print()


if __name__ == "__main__":
    main()
