#!/usr/bin/env python3
"""Dump the L8 ADD/SUB MARK_AX operand vectors to JSON for offline tuning.

Reads the residual at the OP compute row (last MARK_AX) just BEFORE the
AddSub5StageBlock (physical block 9 output = block 10 input), capturing the
full 16-cell ALU_LO/HI (operand A) and AX_CARRY_LO/HI (operand B) bands plus
OP_ADD/OP_SUB/MARK_AX. The declarative ADD/SUB wrap reads exactly these.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/probe_addsub_operand_vectors.py \
        > /tmp/addsub_vectors.json
"""
import os
import sys
import json

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import torch

from tools.probe_groundtruth import build_groundtruth_probe
from tests.test_suite_1000 import generate_test_programs
from src.compiler import compile_c

# Block index whose OUTPUT is the AddSub wrap's input. The AddSub5StageBlock
# sits at the first L8 post-op-expansion block; its input is the L8 main
# PureFFN output. We find that index dynamically.


def _find_addsub_input_block(probe):
    blmap = probe.block_layer_map()
    for i, r in enumerate(blmap):
        if "AddSub5StageBlock" in r["ffn"]:
            return i - 1  # block whose output feeds the AddSub wrap
    raise RuntimeError("AddSub5StageBlock not found in block map")


def band(row, dp, name):
    base = dp[name]
    return [round(float(row[base + i].item()), 4) for i in range(16)]


def main():
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    in_blk = _find_addsub_input_block(probe)
    progs = generate_test_programs()
    out = {"input_block": in_blk, "cases": []}
    # Representative sample (override with ids on argv). 12 ADD + 12 SUB is
    # enough to tune the 3-way AND weights against the real operand artifacts.
    extra = [int(a) for a in sys.argv[1:] if a.isdigit()]
    sample = extra or (list(range(0, 12)) + list(range(50, 62)))
    for i in sample:
        src, want, *_ = progs[i]
        bc = compile_c(src)[0]
        ctx = probe._final_context(bc, max_steps=20)
        got = probe._decode_exit_code(ctx)
        S = len(ctx)
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        with torch.no_grad():
            emb = model.embed(toks)[0]
            ax_rows = [r for r in range(S)
                       if emb[r, dp["MARK_AX"]].abs().item() > 0.5]
            op_row = ax_rows[-1]
            resid = model.forward(toks, stop_after_block=in_blk)[0]
        row = resid[op_row]
        out["cases"].append({
            "id": i,
            "src": src,
            "want": want,
            "got": got,
            "op": "ADD" if i < 50 else "SUB",
            "ALU_LO": band(row, dp, "ALU_LO"),
            "ALU_HI": band(row, dp, "ALU_HI"),
            "AX_CARRY_LO": band(row, dp, "AX_CARRY_LO"),
            "AX_CARRY_HI": band(row, dp, "AX_CARRY_HI"),
            "OP_ADD": round(float(row[dp["OP_ADD"]].item()), 3),
            "OP_SUB": round(float(row[dp["OP_SUB"]].item()), 3),
            "MARK_AX": round(float(row[dp["MARK_AX"]].item()), 3),
        })
    print(json.dumps(out))


if __name__ == "__main__":
    main()
