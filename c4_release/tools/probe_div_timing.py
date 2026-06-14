#!/usr/bin/env python3
"""Find the block where STACK0_BYTE_VAL_1 first appears at the picked row,
vs the block where the divmod post_op computes OUTPUT, to settle the timing.

Usage: CUDA_VISIBLE_DEVICES=0 python tools/probe_div_timing.py
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
    bc = []
    for op in ops:
        if isinstance(op, tuple):
            o, i = op; bc.append(o | (i << 8))
        else: bc.append(op)
    return bc


PROG = _mk([(Opcode.IMM, 1162), Opcode.PSH, (Opcode.IMM, 37), Opcode.DIV, Opcode.EXIT])


def onehot(row, base, w=16):
    return [(i, round(float(row[base+i].item()),2)) for i in range(w) if float(row[base+i].item())>0.5]


def main():
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    nblocks = len(model.blocks)
    S1LO = dp["STACK0_BYTE_VAL_1_LO"]; OLO = dp["OUTPUT_LO"]; OHI = dp["OUTPUT_HI"]
    MARK_AX = dp["MARK_AX"]; OP_DIV = dp["OP_DIV"]; OP_MOD = dp["OP_MOD"]
    STACK0_BYTE1 = dp["STACK0_BYTE1"]
    ctx = probe._final_context(PROG, max_steps=6)
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)

    # Print block_layer_map for the L10..L15 region.
    blmap = probe.block_layer_map()
    print("# block -> logical layer (ffn type) for blocks 9..20:")
    for r in blmap:
        if 9 <= r["physical"] <= 20:
            print(f"  blk{r['physical']:2d} L{r['logical']:2d} exp={r['is_post_op_expansion']!s:5} ffn={r['ffn']}")

    # find div compute row and picked stack1 row at a late block
    resid_late = model.forward(padded, stop_after_block=20)[0]
    div_rows = [r for r in range(resid_late.shape[0])
                if float(resid_late[r,MARK_AX].item())>0.5
                and (float(resid_late[r,OP_DIV].item())>0.5 or float(resid_late[r,OP_MOD].item())>0.5)]
    divrow = div_rows[-1]
    stack1_rows = [r for r in range(divrow+1) if float(resid_late[r,STACK0_BYTE1].item())>0.5]
    picked = stack1_rows[-1]
    print(f"\n# divrow={divrow} picked_stack1_row={picked}")
    print(f"\n{'blk':>3} {'S0VAL1@picked':>20} {'OUTPUT@divrow(lo,hi)':>22}")
    for blk in range(9, min(28, nblocks)):
        r = model.forward(padded, stop_after_block=blk)[0]
        s1 = onehot(r[picked], S1LO)
        olo = onehot(r[divrow], OLO); ohi = onehot(r[divrow], OHI)
        print(f"{blk:>3} {str(s1):>20} lo={olo} hi={ohi}")


if __name__ == "__main__":
    main()
