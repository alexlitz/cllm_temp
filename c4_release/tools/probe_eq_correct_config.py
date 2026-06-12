#!/usr/bin/env python3
"""Re-probe the EQ comparison mechanism in the EXACT production config
(disk_cache=False, efficient, DEFAULT_N_HEADS/FFN_HIDDEN, max_seq_len=4096).
Earlier probes used a different (default-arg) config with a different block/dim
structure. Dump operands + CMP + OUTPUT_LO at the AX row across all blocks for
eq_true/eq_false/lt_true so the EQ engine can be re-calibrated. spec_k=0.
"""
import os, sys
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(_ROOT))
import torch
from neural_vm.embedding import Opcode
from neural_vm.unified_compiler.full_vm_compiler_dynamic import compile_full_vm_dynamic

def _mk(ops):
    bc=[]
    for op in ops:
        if isinstance(op,tuple): opc,imm=op; bc.append(opc|(imm<<8))
        else: bc.append(op)
    return bc
PROGS={
 "eq_true":_mk([(Opcode.IMM,5),Opcode.PSH,(Opcode.IMM,5),Opcode.EQ,Opcode.EXIT]),
 "eq_false":_mk([(Opcode.IMM,5),Opcode.PSH,(Opcode.IMM,7),Opcode.EQ,Opcode.EXIT]),
 "lt_true":_mk([(Opcode.IMM,10),Opcode.PSH,(Opcode.IMM,20),Opcode.LT,Opcode.EXIT]),
}
def hot(c,t=0.3):
    return [] if c is None else [(i,round(v,2)) for i,v in enumerate(c) if abs(v)>t]

def main(sel):
    from neural_vm.vm_step import DEFAULT_N_HEADS, DEFAULT_FFN_HIDDEN
    print("compiling production config...",flush=True)
    model,_=compile_full_vm_dynamic(strict=False,disk_cache=False,alu_mode="efficient",
        n_heads=DEFAULT_N_HEADS,ffn_hidden=DEFAULT_FFN_HIDDEN,max_seq_len=4096)
    if torch.cuda.is_available(): model=model.cuda()
    model.eval(); dp=model.dim_positions; dev=next(model.parameters()).device
    from neural_vm.run_vm import AutoregressiveVMRunner
    from neural_vm.batched_pure_neural import BatchedPureNeuralRunner
    from tools.probe_groundtruth import GroundTruthProbe
    mr=AutoregressiveVMRunner(pure_neural=True,trust_neural_alu=True,spec_k=0,cache_model=False)
    mr.model=model; mr._func_call_handlers={}; mr._syscall_handlers={}
    runner=BatchedPureNeuralRunner(model_runner=mr); probe=GroundTruthProbe(runner)
    ax_b=dp["MARK_AX"]; se_b=dp["MARK_SE_ONLY"]
    def band(bc,blk,pos,base,w):
        b=dp.get(base)
        if b is None: return None
        dn={f"{base}+{i}":b+i for i in range(w)}
        v=probe.residual_at(bc,block_idx=blk,position=pos,dim_names=dn)
        return [v[f"{base}+{i}"] for i in range(w)]
    for pn in sel:
        bc=PROGS[pn]
        r=runner.run_batch([bc],max_steps=20,spec_k=0,bucket_by_predicted_length=False)
        ctx=probe._final_context(bc,max_steps=20); S=len(ctx)
        toks=torch.tensor([ctx],dtype=torch.long,device=dev)
        with torch.no_grad(): emb=model.embed(toks)[0]
        se_rows=[r2 for r2 in range(S) if emb[r2,se_b].abs().item()>0.5]
        ax_rows=[r2 for r2 in range(S) if emb[r2,ax_b].abs().item()>0.5]
        se_row=se_rows[-1] if se_rows else S-1
        ax_row=max((r2 for r2 in ax_rows if r2<se_row),default=(ax_rows[-1] if ax_rows else S-1))
        print(f"\n=== {pn} got={r[0][1]} ax_row={ax_row} nblocks={len(model.blocks)} ===")
        # operands at AX row, find which blocks have them
        for blk in (8,9,10,11,12):
            al=band(bc,blk,ax_row,"ALU_LO",16); ah=band(bc,blk,ax_row,"ALU_HI",16)
            cl=band(bc,blk,ax_row,"AX_CARRY_LO",16); ch=band(bc,blk,ax_row,"AX_CARRY_HI",16)
            opeq=band(bc,blk,ax_row,"OP_EQ",1); oplt=band(bc,blk,ax_row,"OP_LT",1)
            ops=f"OP_EQ={round(opeq[0],1) if opeq else None} OP_LT={round(oplt[0],1) if oplt else None}"
            print(f"  blk{blk:2d} ALU_LO={hot(al)} ALU_HI={hot(ah)} AXC_LO={hot(cl)} AXC_HI={hot(ch)} {ops}")
        # CMP + OUTPUT trajectory
        prev=None
        for blk in range(8,len(model.blocks)):
            cmp=band(bc,blk,ax_row,"CMP",8)
            olo=band(bc,blk,ax_row,"OUTPUT_LO",16)
            am=max(range(16),key=lambda i:olo[i]) if olo else None
            sig=(tuple(round(x,1) for x in (cmp or [])),am)
            if sig!=prev:
                print(f"  blk{blk:2d} CMP={hot(cmp)} OLO_am={am} OLO={hot(olo)[:3]}")
                prev=sig
if __name__=="__main__":
    main(sys.argv[1:] or ["eq_true","eq_false","lt_true"])
