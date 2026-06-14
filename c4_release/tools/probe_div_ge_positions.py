#!/usr/bin/env python3
"""Run the FlattenedDivMod's BDToGEConverter on the actual divmod-input
residual and print GE positions 0..3 NIB_A at the DIV compute row, to see
whether the high byte (position 2/3) lands. Flag C4_DIV_MULTIBYTE controls it.

Usage: CUDA_VISIBLE_DEVICES=0 C4_DIV_MULTIBYTE=1 python tools/probe_div_ge_positions.py
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

def main():
    probe=build_groundtruth_probe(); model=probe.model; dp=model.dim_positions
    dev=next(model.parameters()).device
    MARK_AX=dp["MARK_AX"]; OP_DIV=dp["OP_DIV"]; OP_MOD=dp["OP_MOD"]
    print(f"# C4_DIV_MULTIBYTE={os.environ.get('C4_DIV_MULTIBYTE','0')}")
    # Find the FlattenedDivMod module + which block holds it.
    divmod_blk=None; divmod_mod=None
    for phys, blk in enumerate(model.blocks):
        for p in getattr(blk,"post_ops",[]):
            if "DivMod" in type(p).__name__:
                divmod_blk=phys; divmod_mod=p
        if "DivMod" in type(getattr(blk,"ffn",None)).__name__:
            divmod_blk=phys; divmod_mod=blk.ffn
    print(f"# divmod module at block {divmod_blk} type={type(divmod_mod).__name__ if divmod_mod else None}")

    ctx=probe._final_context(PROG,max_steps=6)
    padded=torch.tensor([ctx],dtype=torch.long,device=dev)
    # Residual INPUT to the divmod block = output of block (divmod_blk-1).
    if divmod_blk is None:
        print("!! no divmod module found in post_ops/ffn (expanded into own block?)")
        # Find by scanning for the converter — fall back to block before OUTPUT write.
        return
    x_in = model.forward(padded, stop_after_block=divmod_blk-1)  # [1,S,D]
    # Locate the converter on the module.
    conv=None
    for m in divmod_mod.modules():
        if type(m).__name__=="BDToGEConverter": conv=m; break
    if conv is None:
        print("!! no BDToGEConverter in divmod module"); return
    with torch.no_grad():
        x_ge = conv(x_in)  # [1,S,8,160]
    resid=x_in[0]
    div_rows=[r for r in range(resid.shape[0]) if float(resid[r,MARK_AX].item())>0.5 and (float(resid[r,OP_DIV].item())>0.5 or float(resid[r,OP_MOD].item())>0.5)]
    row=div_rows[-1]
    NIB_A=conv.ge.NIB_A
    print(f"# DIV compute row={row}")
    for pos in range(4):
        print(f"  GE pos{pos} NIB_A = {float(x_ge[0,row,pos,NIB_A].item()):.3f}")
    # dividend value = sum nib_i * 16^i
    val=sum(float(x_ge[0,row,p,NIB_A].item())*(16**p) for p in range(8))
    print(f"  => reconstructed dividend ~= {val:.1f}  (expect 1162)")

if __name__=="__main__":
    main()
