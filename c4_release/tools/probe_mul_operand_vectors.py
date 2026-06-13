#!/usr/bin/env python3
"""Dump FULL operand vectors at the MARK_AX MUL row as JSON.

Captures, for several (A,B) MUL cases, the 16-cell ALU_LO, ALU_HI,
AX_CARRY_LO, AX_CARRY_HI vectors plus MARK_AX and OP_MUL at the MUL row,
read from the residual at the INPUT of the L11 block (the row the
wide_mul 5-way AND consumes). Output is JSON to stdout for an offline
weight tuner (tools/tune_mul_width2.py).

The MUL row is the LAST MARK_AX row at/before the final MARK_SE_ONLY row
AND with OP_MUL hot (distinguishes it from the IMM-staging AX rows).

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. C4_MUL_WIDTH2=1 \
    python tools/probe_mul_operand_vectors.py > /tmp/mul_w2_vectors.json
"""
import json
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


def vec(row, dp, name, width=16):
    base = dp.get(name)
    if base is None:
        return None
    return [round(float(row[base + i].item()), 4) for i in range(width)]


CASES = [(6, 7), (100, 5), (12, 12), (15, 17), (3, 9), (16, 16), (200, 2),
         (50, 5), (9, 9), (255, 1), (1, 255), (17, 15)]


def main():
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    ax_base = dp["MARK_AX"]
    se_base = dp["MARK_SE_ONLY"]
    opmul_base = dp["OP_MUL"]

    l11_phys = None
    for phys, blk in enumerate(model.blocks):
        if getattr(blk, "_logical_layer", phys) == 11:
            l11_phys = phys
            break

    out = {"l11_phys": l11_phys, "cases": []}
    for a, b in CASES:
        bc = prog(a, b)
        ctx = probe._final_context(bc, max_steps=20)
        S = len(ctx)
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        with torch.no_grad():
            r_in = model.forward(toks, stop_after_block=l11_phys - 1)[0]
        # MUL row: AX row with OP_MUL hot (input residual carries OP_MUL).
        cand = [r for r in range(S)
                if r_in[r, ax_base].abs().item() > 0.5
                and r_in[r, opmul_base].item() > 0.5]
        mul_row = cand[-1] if cand else None
        if mul_row is None:
            # fallback: last AX row before last SE row
            ax_rows = [r for r in range(S) if r_in[r, ax_base].abs().item() > 0.5]
            se_rows = [r for r in range(S) if r_in[r, se_base].abs().item() > 0.5]
            se_row = se_rows[-1] if se_rows else None
            mul_row = max((r for r in ax_rows if (se_row is None or r < se_row)),
                          default=(ax_rows[-1] if ax_rows else None))
        out["cases"].append({
            "a": a, "b": b, "product": a * b, "mul_row": mul_row, "S": S,
            "MARK_AX": round(float(r_in[mul_row, ax_base].item()), 4),
            "OP_MUL": round(float(r_in[mul_row, opmul_base].item()), 4),
            "ALU_LO": vec(r_in[mul_row], dp, "ALU_LO"),
            "ALU_HI": vec(r_in[mul_row], dp, "ALU_HI"),
            "AX_CARRY_LO": vec(r_in[mul_row], dp, "AX_CARRY_LO"),
            "AX_CARRY_HI": vec(r_in[mul_row], dp, "AX_CARRY_HI"),
        })
    print(json.dumps(out))


if __name__ == "__main__":
    main()
