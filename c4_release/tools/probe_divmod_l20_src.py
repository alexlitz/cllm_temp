#!/usr/bin/env python3
"""At the L20 corruptor block, inspect what ALU_LO/HI and AX_CARRY hold at the
divmod row (to see what the l16_stack0_marker_from_alu materializer reads),
and compare passing vs failing cases.
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
        else:
            bc.append(op)
    return bc


def oh_val(row, base, w=16):
    vals = [float(row[base + i].item()) for i in range(w)]
    m = max(vals)
    return vals.index(m) if m > 0.5 else None


def main():
    args = sys.argv[1:] or ["1162/37", "843/31", "2009/43", "176/4"]
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    nblocks = len(model.blocks)
    OLO = dp["OUTPUT_LO"]; OHI = dp["OUTPUT_HI"]
    MARK_AX = dp["MARK_AX"]; OP_DIV = dp["OP_DIV"]; OP_MOD = dp["OP_MOD"]
    ALU_LO = dp["ALU_LO"]; ALU_HI = dp["ALU_HI"]
    AXC_LO = dp["AX_CARRY_LO"]; AXC_HI = dp["AX_CARRY_HI"]
    # probe blocks just before L20 (33) and at L20 (34)
    for arg in args:
        a, b = arg.split("/"); a = int(a); b = int(b)
        PROG = _mk([(Opcode.IMM, a), Opcode.PSH, (Opcode.IMM, b), Opcode.DIV, Opcode.EXIT])
        ctx = probe._final_context(PROG, max_steps=6)
        padded = torch.tensor([ctx], dtype=torch.long, device=dev)
        rl = model.forward(padded, stop_after_block=nblocks - 1)[0]
        drs = [r for r in range(rl.shape[0])
               if float(rl[r, MARK_AX].item()) > 0.5
               and (float(rl[r, OP_DIV].item()) > 0.5 or float(rl[r, OP_MOD].item()) > 0.5)]
        row = drs[-1]
        print(f"\n=== {a}/{b} (q={a//b}=0x{a//b:X}) divisor=0x{b:X} row={row} ===")
        for blk in [28, 33, 34, 41]:
            r = model.forward(padded, stop_after_block=blk)[0][row]
            def rawband(base):
                vals = [round(float(r[base + i].item()), 1) for i in range(16)]
                hot = [(i, v) for i, v in enumerate(vals) if abs(v) > 0.4]
                return hot
            print(f"  blk{blk}: ALU=({oh_val(r,ALU_LO)},{oh_val(r,ALU_HI)}) "
                  f"AXC=({oh_val(r,AXC_LO)},{oh_val(r,AXC_HI)}) "
                  f"OUT=({oh_val(r,OLO)},{oh_val(r,OHI)})")
            print(f"        ALU_LO hot={rawband(ALU_LO)} ALU_HI hot={rawband(ALU_HI)}")


if __name__ == "__main__":
    main()
