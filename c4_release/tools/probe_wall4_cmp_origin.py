#!/usr/bin/env python3
"""Find which block first writes CMP at the binop AX row, and what OP_* / CMP_GROUP
look like there. spec_k=0, cached."""
import os, sys
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from neural_vm.embedding import Opcode
from tools.probe_groundtruth import build_groundtruth_probe

def _mk(ops):
    bc=[]
    for op in ops:
        if isinstance(op, tuple):
            opc, imm = op; bc.append(opc | (imm<<8))
        else: bc.append(op)
    return bc

PROGS = {
  "eq_true":(_mk([(Opcode.IMM,5),Opcode.PSH,(Opcode.IMM,5),Opcode.EQ,Opcode.EXIT]),1),
  "lt_true":(_mk([(Opcode.IMM,10),Opcode.PSH,(Opcode.IMM,20),Opcode.LT,Opcode.EXIT]),1),
}

def band(probe,bc,blk,pos,dp,base,w):
    b=dp.get(base)
    if b is None: return None
    dn={f"{base}+{i}":b+i for i in range(w)}
    v=probe.residual_at(bc,block_idx=blk,position=pos,dim_names=dn)
    return [v[f"{base}+{i}"] for i in range(w)]

def hot(c,t=0.3):
    return [] if c is None else [(i,round(v,2)) for i,v in enumerate(c) if abs(v)>t]

def main(sel):
    probe=build_groundtruth_probe(); model=probe.model; dp=model.dim_positions
    dev=next(model.parameters()).device
    se_b=dp["MARK_SE_ONLY"]; ax_b=dp["MARK_AX"]
    for pn in sel:
        bc,exp=PROGS[pn]
        ctx=probe._final_context(bc,max_steps=20); S=len(ctx)
        toks=torch.tensor([ctx],dtype=torch.long,device=dev)
        with torch.no_grad(): emb=model.embed(toks)[0]
        se_rows=[r for r in range(S) if emb[r,se_b].abs().item()>0.5]
        ax_rows=[r for r in range(S) if emb[r,ax_b].abs().item()>0.5]
        se_row=se_rows[-1]; ax_row=max((r for r in ax_rows if r<se_row),default=ax_rows[-1])
        print(f"\n=== {pn} exp={exp} ax_row={ax_row} se_row={se_row} ===")
        for blk in range(0,12):
            cmp=band(probe,bc,blk,ax_row,dp,"CMP",4)
            grp=band(probe,bc,blk,ax_row,dp,"CMP_GROUP",1)
            ops={}
            for o in ("OP_EQ","OP_LT","OP_LE","OP_GT","OP_GE","OP_NE"):
                vv=band(probe,bc,blk,ax_row,dp,o,1)
                if vv and abs(vv[0])>0.3: ops[o]=round(vv[0],2)
            print(f"  blk{blk:2d} AXrow CMP={hot(cmp)} CMP_GROUP={hot(grp)} OPS={ops}")

if __name__=="__main__":
    main(sys.argv[1:] or list(PROGS.keys()))
