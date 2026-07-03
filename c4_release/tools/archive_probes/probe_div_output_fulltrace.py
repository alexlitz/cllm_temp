#!/usr/bin/env python3
"""Trace OUTPUT_LO/HI at the DIV compute row across ALL blocks, to find which
block actually WRITES the (final) quotient. Settles whether the L15 divide
overwrites an earlier L10 partial write.

Usage: CUDA_VISIBLE_DEVICES=0 [C4_DIV_MULTIBYTE=1] python tools/probe_div_output_fulltrace.py
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

def oh(row,base,w=16):
    return [i for i in range(w) if float(row[base+i].item())>0.5]

def main():
    probe=build_groundtruth_probe(); model=probe.model; dp=model.dim_positions
    dev=next(model.parameters()).device; nblocks=len(model.blocks)
    OLO=dp["OUTPUT_LO"]; OHI=dp["OUTPUT_HI"]; MARK_AX=dp["MARK_AX"]
    OP_DIV=dp["OP_DIV"]; OP_MOD=dp["OP_MOD"]
    print(f"# C4_DIV_MULTIBYTE={os.environ.get('C4_DIV_MULTIBYTE','0')} nblocks={nblocks}")
    ctx=probe._final_context(PROG,max_steps=6)
    padded=torch.tensor([ctx],dtype=torch.long,device=dev)
    rl=model.forward(padded,stop_after_block=nblocks-1)[0]
    drs=[r for r in range(rl.shape[0]) if float(rl[r,MARK_AX].item())>0.5 and (float(rl[r,OP_DIV].item())>0.5 or float(rl[r,OP_MOD].item())>0.5)]
    row=drs[-1]
    print(f"# DIV compute row={row}")
    prev=None
    for blk in range(nblocks):
        r=model.forward(padded,stop_after_block=blk)[0]
        out=(oh(r[row],OLO),oh(r[row],OHI))
        if out!=prev:
            print(f"  blk{blk:2d} L{getattr(model.blocks[blk],'_logical_layer',blk)}: OUTPUT lo={out[0]} hi={out[1]}")
            prev=out
    _,code=probe.emitted_result(PROG,max_steps=6)
    print(f"# final neural exit = {code} (expect 31)")

if __name__=="__main__":
    main()
