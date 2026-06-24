#!/usr/bin/env python3
"""FAST divmod operand-source probe (4 forwards/program, not 56).

Reads the residual-stream operand-source bands the FlattenedDivMod BDToGEConverter
consumes (ALU_LO/HI = byte0 source, SE_ALU_LO/HI = byte0 recovery, AX_CARRY = byte0
of operand B + divisor, STACK0_BYTE_VAL_1 = byte1 source) at the DIV/MOD compute
row, plus the final emitted exit. Tells byte-0-drop vs byte-1-drop vs downstream
OUTPUT leak apart without the slow per-block scan.

  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 CUDA_VISIBLE_DEVICES="" \
    python tools/probe_divmod_fast.py 364/14 1132/33 268/17 428/16 div=True
Pass op=mod after the operands to build MOD bytecode for those.
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
    if base is None:
        return None
    vals = [float(row[base + i].item()) for i in range(w)]
    m = max(vals)
    if m <= 0.5:
        return None
    return vals.index(m)


def raw_max(row, base, w=16):
    if base is None:
        return None
    vals = [float(row[base + i].item()) for i in range(w)]
    return round(max(vals), 1)


def main():
    args = sys.argv[1:]
    op_mod = any(a == "op=mod" for a in args)
    pairs = [a for a in args if "/" in a]
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    nblocks = len(model.blocks)
    OLO = dp["OUTPUT_LO"]; OHI = dp["OUTPUT_HI"]
    MARK_AX = dp["MARK_AX"]; OP_DIV = dp["OP_DIV"]; OP_MOD = dp["OP_MOD"]
    ALU_LO = dp.get("ALU_LO"); ALU_HI = dp.get("ALU_HI")
    AXC_LO = dp.get("AX_CARRY_LO"); AXC_HI = dp.get("AX_CARRY_HI")
    SE_LO = dp.get("SE_ALU_LO"); SE_HI = dp.get("SE_ALU_HI")
    SBV1_LO = dp.get("STACK0_BYTE_VAL_1_LO"); SBV1_HI = dp.get("STACK0_BYTE_VAL_1_HI")
    op = Opcode.MOD if op_mod else Opcode.DIV
    print(f"# nblocks={nblocks} op={'MOD' if op_mod else 'DIV'} "
          f"NO_STACK0_EMIT={os.environ.get('C4_NO_STACK0_EMIT')} "
          f"DIV_MULTIBYTE={os.environ.get('C4_DIV_MULTIBYTE','1')} "
          f"BYTE0_SE={os.environ.get('C4_DIVMOD_BYTE0_SE_RECOVER','1')}")
    for arg in pairs:
        a, b = arg.split("/"); a = int(a); b = int(b)
        res = (a % b) if op_mod else (a // b)
        PROG = _mk([(Opcode.IMM, a), Opcode.PSH, (Opcode.IMM, b), op, Opcode.EXIT])
        ctx = probe._final_context(PROG, max_steps=6)
        padded = torch.tensor([ctx], dtype=torch.long, device=dev)
        # last block residual: find divmod row
        rl = model.forward(padded, stop_after_block=nblocks - 1)[0]
        drs = [r for r in range(rl.shape[0])
               if float(rl[r, MARK_AX].item()) > 0.5
               and (float(rl[r, OP_DIV].item()) > 0.5 or float(rl[r, OP_MOD].item()) > 0.5)]
        print(f"\n=== {a}/{b} ({'mod' if op_mod else 'div'}) lo=0x{a&0xFF:02X}({a&0xFF}) "
              f"hi=0x{(a>>8)&0xFF:02X}({(a>>8)&0xFF}) expect={res} ===")
        if not drs:
            print("  NO DIV/MOD compute row found at final block")
        else:
            row = drs[-1]
            # scan blocks 28..end, print whenever OUTPUT one-hot changes
            prev = None
            for blk in range(28, nblocks):
                rr = model.forward(padded, stop_after_block=blk)[0][row]
                cur = (oh_val(rr, OLO), oh_val(rr, OHI))
                if cur != prev or blk == nblocks - 1:
                    lay = getattr(model.blocks[blk], '_logical_layer', blk)
                    print(f"  blk{blk:2d} L{lay}: "
                          f"OUT=({cur[0]},{cur[1]})|raw({raw_max(rr,OLO)},{raw_max(rr,OHI)}) "
                          f"SBV1=({oh_val(rr,SBV1_LO)},{oh_val(rr,SBV1_HI)}) "
                          f"AXC=({oh_val(rr,AXC_LO)},{oh_val(rr,AXC_HI)})")
                    prev = cur
        _, code = probe.emitted_result(PROG, max_steps=6)
        print(f"  --> neural exit = {code} (expect {res})")


if __name__ == "__main__":
    main()
