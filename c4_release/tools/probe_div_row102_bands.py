#!/usr/bin/env python3
"""At the picked STACK0_BYTE1 frame row (102 for 1162/37), dump ALL one-hot
nibble bands at block 13 (BEFORE the divmod compute at block 14) to find
which band carries the dividend high byte 0x04 EARLY enough for the divide.

Usage: CUDA_VISIBLE_DEVICES=0 python tools/probe_div_row102_bands.py
"""
import os, sys
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import torch
from neural_vm.embedding import Opcode
from tools.probe_groundtruth import build_groundtruth_probe


def _mk(ops):
    bc=[]
    for op in ops:
        if isinstance(op, tuple):
            o,i=op; bc.append(o|(i<<8))
        else: bc.append(op)
    return bc


PROG = _mk([(Opcode.IMM,1162),Opcode.PSH,(Opcode.IMM,37),Opcode.DIV,Opcode.EXIT])


def main():
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    ctx = probe._final_context(PROG, max_steps=6)
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)
    STACK0_BYTE1 = dp["STACK0_BYTE1"]; MARK_AX = dp["MARK_AX"]
    OP_DIV = dp["OP_DIV"]; OP_MOD = dp["OP_MOD"]

    for blk in (11, 12, 13, 14):
        resid = model.forward(padded, stop_after_block=blk)[0]
        div_rows = [r for r in range(resid.shape[0])
                    if float(resid[r,MARK_AX].item())>0.5
                    and (float(resid[r,OP_DIV].item())>0.5 or float(resid[r,OP_MOD].item())>0.5)]
        divrow = div_rows[-1]
        stack1_rows = [r for r in range(divrow+1) if float(resid[r,STACK0_BYTE1].item())>0.5]
        picked = stack1_rows[-1]
        pr = resid[picked]
        print(f"\n=== block {blk}: picked STACK0_BYTE1 row {picked} (looking for a band == 0x04) ===")
        # scan all named 16-wide nibble bands for a hot cell == 4 (and partner == 0)
        for name, base in sorted(dp.items(), key=lambda kv: kv[1]):
            if not isinstance(base, int): continue
            if base + 16 > pr.shape[0]: continue
            hot = [(i, round(float(pr[base+i].item()),2)) for i in range(16)
                   if float(pr[base+i].item())>0.5]
            if hot and any(i==4 for i,_ in hot):
                print(f"  {name:28s} base={base}: {hot}")


if __name__ == "__main__":
    main()
