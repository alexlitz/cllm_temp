"""Flag-OFF (baked) regression check: draft+verify byte-exact on corpus programs.
Confirms cfm changes (code_off=0 path) are byte-identical to before."""
import warnings; warnings.filterwarnings('ignore')
import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
os.environ.pop('C4_PF_CFM', None)   # FLAG OFF (baked)
import torch
from c4_min import isa
from c4_min.pf_speculative import draft_pf_program, verify_blocks
from c4_min.compact_alloc import build_compact_sparse_streaming

progs = {
  'add>255': [isa.Instr(isa.IMM,300),isa.Instr(isa.PSH),isa.Instr(isa.IMM,44),isa.Instr(isa.ADD),isa.Instr(isa.HALT)],
  'bnz-loop':[isa.Instr(isa.IMM,3),isa.Instr(isa.PSH),isa.Instr(isa.IMM,1),isa.Instr(isa.SUB),isa.Instr(isa.BNZ,1),isa.Instr(isa.HALT)],
  'mul':     [isa.Instr(isa.IMM,12),isa.Instr(isa.PSH),isa.Instr(isa.IMM,11),isa.Instr(isa.MUL),isa.Instr(isa.HALT)],
  'div':     [isa.Instr(isa.IMM,100),isa.Instr(isa.PSH),isa.Instr(isa.IMM,7),isa.Instr(isa.DIV),isa.Instr(isa.HALT)],
}
expect={'add>255':344,'bnz-loop':0,'mul':132,'div':14}
sparse,L,_=build_compact_sparse_streaming(code_size=64,recurrent_divmod=True,compute_mode='dense_kernel')
sparse=sparse.to('cuda:0')
allok=True
for n,code in progs.items():
    draft=draft_pf_program(code,max_steps=50,mask=0xFFFFFFFF)
    assert draft.code_off==0, f'FLAG OFF must have code_off=0, got {draft.code_off}'
    st={}
    vr=verify_blocks(sparse,L,code,draft,block_steps=50,device='cuda:0',evict=False,mask=0xFFFFFFFF,stats=st,fast=True)
    ok=vr.all_matched and draft.final_ax_masked==expect[n]
    allok&=ok
    print(f'{n:10s} code_off={draft.code_off} final_ax={draft.final_ax_masked} matched={vr.all_matched} {"OK" if ok else "FAIL"}', flush=True)
print('FLAG-OFF (baked) ALL OK' if allok else 'FLAG-OFF SOME FAIL', flush=True)
sys.exit(0 if allok else 1)
