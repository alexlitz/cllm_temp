"""cfm fast-path on several diverse programs (loops, jmp, mul/div, mem, >255 lits)."""
import warnings; warnings.filterwarnings('ignore')
import os, sys, time
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
os.environ['C4_PF_CFM']='1'
os.environ.setdefault('C4_DRAFT_CMP32','1')
import torch
from c4_min import isa
from c4_min.pf_speculative import draft_pf_program, verify_blocks
from c4_min.compact_alloc import build_compact_sparse_streaming

progs = {
  'jmp':     [isa.Instr(isa.JMP,3),isa.Instr(isa.IMM,99),isa.Instr(isa.HALT),isa.Instr(isa.IMM,7),isa.Instr(isa.HALT)],
  'bnz-loop':[isa.Instr(isa.IMM,3),isa.Instr(isa.PSH),isa.Instr(isa.IMM,1),isa.Instr(isa.SUB),isa.Instr(isa.BNZ,1),isa.Instr(isa.HALT)],
  'mul':     [isa.Instr(isa.IMM,12),isa.Instr(isa.PSH),isa.Instr(isa.IMM,11),isa.Instr(isa.MUL),isa.Instr(isa.HALT)],
  'div':     [isa.Instr(isa.IMM,100),isa.Instr(isa.PSH),isa.Instr(isa.IMM,7),isa.Instr(isa.DIV),isa.Instr(isa.HALT)],
  'bigimm':  [isa.Instr(isa.IMM,70000),isa.Instr(isa.PSH),isa.Instr(isa.IMM,5000),isa.Instr(isa.ADD),isa.Instr(isa.HALT)],
}
# expected final AX (32-bit): jmp->7, bnz-loop 3-1-1-1->0, mul 12*11=132, div 100/7=14, bigimm 75000
expect = {'jmp':7,'bnz-loop':0,'mul':132,'div':14,'bigimm':75000}

sparse,L,stats = build_compact_sparse_streaming(code_size=64, recurrent_divmod=True, compute_mode='dense_kernel')
dev='cuda:0' if torch.cuda.is_available() else 'cpu'
if dev!='cpu': sparse=sparse.to(dev)
print(f'built dim={sparse.embed.shape[1]} blocks={len(sparse.blocks)}', flush=True)

allok=True
for n,code in progs.items():
    draft = draft_pf_program(code, max_steps=50, mask=0xFFFFFFFF)
    st={}
    vr = verify_blocks(sparse, L, code, draft, block_steps=50, device=dev, evict=False, mask=0xFFFFFFFF, stats=st, fast=True)
    ok = vr.all_matched and draft.final_ax_masked==expect[n]
    allok &= ok
    print(f'{n:10s} steps={draft.step_count} final_ax={draft.final_ax_masked} expect={expect[n]} matched={vr.all_matched}  {"OK" if ok else "FAIL "+str(vr.first_mismatch)}', flush=True)
print('ALL OK' if allok else 'SOME FAIL', flush=True)
sys.exit(0 if allok else 1)
