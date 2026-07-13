#!/usr/bin/env python3
"""Probe the ADD carry side-signal + OUTPUT after the AddSub block.

Answers CBC-Phase-1 gate (d): on a CLEAN operand (C4_CLEAN_OPERAND=1) does the
inter-byte carry compute at the golden magnitude (~2.0) instead of the dirty-
operand inflation (~23)? Run OFF and ON separately and compare.

Usage:
    C4_CLEAN_OPERAND=1 python tools/probe_clean_operand_carry.py
    python tools/probe_clean_operand_carry.py   # OFF
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
            bc.append(op[0] | (op[1] << 8))
        else:
            bc.append(op)
    return bc


PROGRAMS = {
    # 654 + 114 = 768 (0x300): byte-0 carries (0x8e + 0x72 = 0x100), the
    # Phase-1 add_0 regressing id.
    "add_654_114": (_mk([(Opcode.IMM, 654), Opcode.PSH, (Opcode.IMM, 114),
                         Opcode.ADD, Opcode.EXIT]), 768),
    # 200 + 100 = 300 (0x12C): byte-0 carries.
    "add_200_100": (_mk([(Opcode.IMM, 200), Opcode.PSH, (Opcode.IMM, 100),
                         Opcode.ADD, Opcode.EXIT]), 300),
}


def band(row, dp, name, w=16, thr=0.3):
    base = dp[name]
    return [(i, round(float(row[base + i].item()), 2))
            for i in range(w) if abs(row[base + i].item()) > thr]


def main():
    on = os.environ.get("C4_CLEAN_OPERAND", "0") != "0"
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    blmap = probe.block_layer_map()
    addsub_blk = next(i for i, r in enumerate(blmap)
                      if "AddSub5StageBlock" in r["ffn"])
    print(f"=== C4_CLEAN_OPERAND {'ON' if on else 'OFF'}  addsub_block={addsub_blk} ===")
    for pname, (bc, want) in PROGRAMS.items():
        ctx = probe._final_context(bc, max_steps=20)
        got = probe._decode_exit_code(ctx)
        S = len(ctx)
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        with torch.no_grad():
            emb = model.embed(toks)[0]
        ax_rows = [r for r in range(S) if emb[r, dp["MARK_AX"]].abs().item() > 0.5]
        op_row = ax_rows[-1]
        with torch.no_grad():
            resid = model.forward(toks, stop_after_block=addsub_blk)[0]
        row = resid[op_row]
        # CARRY bands: names may be CARRY+0..3 or AX_CARRY; try known dims.
        carry = {}
        for cname in ("CARRY", "AX_CARRY_LO", "AX_CARRY_HI"):
            if cname in dp:
                carry[cname] = band(row, dp, cname)
        print(f"  {pname}: want={want} got={got} {'PASS' if got==want else 'FAIL'}")
        print(f"    OUTPUT_LO={band(row,dp,'OUTPUT_LO')}  OUTPUT_HI={band(row,dp,'OUTPUT_HI')}")
        for cname, cb in carry.items():
            print(f"    {cname}={cb}")
        # explicit CARRY+1 (the inter-byte carry Phase-1 measured at 23 vs 2.0)
        if "CARRY" in dp:
            c1 = float(row[dp["CARRY"] + 1].item())
            print(f"    CARRY+1 (inter-byte) = {round(c1,3)}")


if __name__ == "__main__":
    main()
