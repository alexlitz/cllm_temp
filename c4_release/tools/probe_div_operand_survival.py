#!/usr/bin/env python3
"""At the DIV compute row (141 for 1162/37), trace the operands across blocks
14..20 to see if the dividend byte0 (ALU_LO/HI), divisor (AX_CARRY_LO/HI),
and the high byte (STACK0_BYTE_VAL_1) are ALL clean+present at a candidate
later block (L11=15/16, L12=16, L13=17) — i.e. could the divmod safely run
there and read STACK0_BYTE_VAL_1.

Usage: CUDA_VISIBLE_DEVICES=0 python tools/probe_div_operand_survival.py
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

PROGS={
  "div_1162_37":(_mk([(Opcode.IMM,1162),Opcode.PSH,(Opcode.IMM,37),Opcode.DIV,Opcode.EXIT]),31),
  "div_84_2":(_mk([(Opcode.IMM,84),Opcode.PSH,(Opcode.IMM,2),Opcode.DIV,Opcode.EXIT]),42),
}

def oh(row,base,w=16):
    return [i for i in range(w) if float(row[base+i].item())>0.5]

def main():
    probe=build_groundtruth_probe(); model=probe.model; dp=model.dim_positions
    dev=next(model.parameters()).device
    ALU_LO=dp["ALU_LO"]; ALU_HI=dp["ALU_HI"]; AXC_LO=dp["AX_CARRY_LO"]; AXC_HI=dp["AX_CARRY_HI"]
    S1LO=dp["STACK0_BYTE_VAL_1_LO"]; S1HI=dp["STACK0_BYTE_VAL_1_HI"]
    MARK_AX=dp["MARK_AX"]; OP_DIV=dp["OP_DIV"]; OP_MOD=dp["OP_MOD"]; STACK0_BYTE1=dp["STACK0_BYTE1"]
    for pn,(prog,exp) in PROGS.items():
        ctx=probe._final_context(prog,max_steps=6)
        padded=torch.tensor([ctx],dtype=torch.long,device=dev)
        resid=model.forward(padded,stop_after_block=20)[0]
        drs=[r for r in range(resid.shape[0]) if float(resid[r,MARK_AX].item())>0.5 and (float(resid[r,OP_DIV].item())>0.5 or float(resid[r,OP_MOD].item())>0.5)]
        divrow=drs[-1]
        s1rows=[r for r in range(divrow+1) if float(resid[r,STACK0_BYTE1].item())>0.5]
        picked=s1rows[-1]
        print(f"\n=== {pn} exp={exp} divrow={divrow} picked_stack1={picked} ===")
        print(f"{'blk':>3} {'ALU(b0 lo,hi)':>14} {'AXC(div lo,hi)':>15} {'S1@divrow':>11} {'S1@picked':>11}")
        for blk in (14,15,16,17,18,19,20):
            r=model.forward(padded,stop_after_block=blk)[0]
            alu=(oh(r[divrow],ALU_LO),oh(r[divrow],ALU_HI))
            axc=(oh(r[divrow],AXC_LO),oh(r[divrow],AXC_HI))
            s1d=(oh(r[divrow],S1LO),oh(r[divrow],S1HI))
            s1p=(oh(r[picked],S1LO),oh(r[picked],S1HI))
            print(f"{blk:>3} {str(alu):>14} {str(axc):>15} {str(s1d):>11} {str(s1p):>11}")

if __name__=="__main__":
    main()
