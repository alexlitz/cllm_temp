#!/usr/bin/env python3
"""At block 13 (before divmod compute), find ANY (row, band) carrying the
dividend high byte 0x04 for 1162/37, restricted to clean operand-ish bands.

Specifically check: STACK0_BYTE_VAL_1, AX_FULL, ADDR_B1, CLEAN_EMBED across
all rows, plus which rows have OP_DIV / MARK_AX / STACK0_BYTE1.

Usage: CUDA_VISIBLE_DEVICES=0 python tools/probe_div_hibyte_anyrow.py
"""
import os, sys
os.environ.setdefault("C4_TEST_SPEC_K","0"); os.environ.setdefault("C4_SMOKE_SPEC_K","0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import torch
from neural_vm.embedding import Opcode
from tools.probe_groundtruth import build_groundtruth_probe


def _mk(ops):
    bc=[]
    for op in ops:
        if isinstance(op,tuple): o,i=op; bc.append(o|(i<<8))
        else: bc.append(op)
    return bc

PROG=_mk([(Opcode.IMM,1162),Opcode.PSH,(Opcode.IMM,37),Opcode.DIV,Opcode.EXIT])

CHECK_BANDS = ["STACK0_BYTE_VAL_1_LO","STACK0_BYTE_VAL_1_HI","AX_FULL_LO","AX_FULL_HI",
               "ADDR_B1_LO","ADDR_B1_HI","CLEAN_EMBED_LO","CLEAN_EMBED_HI",
               "MEM_VAL_B1","OUTPUT_LO","OUTPUT_HI"]

def hot(row, base, w=16):
    return [(i,round(float(row[base+i].item()),2)) for i in range(w) if float(row[base+i].item())>0.5]

def main():
    probe=build_groundtruth_probe(); model=probe.model; dp=model.dim_positions
    dev=next(model.parameters()).device
    ctx=probe._final_context(PROG,max_steps=6)
    padded=torch.tensor([ctx],dtype=torch.long,device=dev)
    MARK_AX=dp["MARK_AX"]; STACK0_BYTE1=dp["STACK0_BYTE1"]; BI1=dp["BYTE_INDEX_1"]
    for blk in (13, 14, 15):
        resid=model.forward(padded,stop_after_block=blk)[0]
        print(f"\n=== block {blk}: rows where a CHECK band == 0x04 (nibble 4) ===")
        for r in range(resid.shape[0]):
            row=resid[r]
            hits=[]
            for name in CHECK_BANDS:
                base=dp.get(name)
                if base is None or base+16>row.shape[0]: continue
                h=hot(row,base)
                if any(i==4 for i,_ in h):
                    hits.append((name,h))
            if hits:
                ax=round(float(row[MARK_AX].item()),2); s1=round(float(row[STACK0_BYTE1].item()),2)
                bi1=round(float(row[BI1].item()),2)
                print(f"  row{r:3d} MARK_AX={ax} STACK0_BYTE1={s1} BI1={bi1}: " +
                      " ".join(f"{n}={h}" for n,h in hits))


if __name__=="__main__":
    main()
