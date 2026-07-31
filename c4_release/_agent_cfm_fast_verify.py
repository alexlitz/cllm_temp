"""cfm fast-path (draft + verify_blocks) correctness on a small program.
Additive test; golden 069cc32f unchanged."""
import warnings; warnings.filterwarnings('ignore')
import os, sys, time
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
os.environ['C4_PF_CFM']='1'
os.environ.setdefault('C4_DRAFT_CMP32','1')
import torch
from c4_min import isa
from c4_min.pf_speculative import draft_pf_program, verify_blocks
from c4_min.compact_alloc import build_compact_sparse_streaming

# program: IMM 300; PSH; IMM 44; ADD; HALT   (proves cfm + 20-bit IMM via fast path)
code = [isa.Instr(isa.IMM,300),isa.Instr(isa.PSH),isa.Instr(isa.IMM,44),
        isa.Instr(isa.ADD),isa.Instr(isa.HALT)]

draft = draft_pf_program(code, max_steps=50, mask=0xFFFFFFFF)
print(f'draft steps={draft.step_count} halted={draft.halted} code_off={draft.code_off} final_ax={draft.final_ax_masked}', flush=True)
for i,f in enumerate(draft.frames):
    print(f'  step {i}: op={f["op"]:5s} pc={f["pc"]} ax={f["ax"]}', flush=True)

t0=time.time()
sparse,L,stats = build_compact_sparse_streaming(code_size=max(len(code)+2,64), recurrent_divmod=True, compute_mode='dense_kernel')
print(f'built dim={sparse.embed.shape[1]} blocks={len(sparse.blocks)} in {time.time()-t0:.1f}s', flush=True)
dev='cuda:0' if torch.cuda.is_available() else 'cpu'
if dev!='cpu': sparse=sparse.to(dev)

stats={}
vr = verify_blocks(sparse, L, code, draft, block_steps=50, device=dev, evict=False, mask=0xFFFFFFFF, stats=stats, fast=True)
print(f'VERIFY matched={vr.all_matched} accepted={vr.accepted_steps}/{vr.total_steps} decoded_final_ax={vr.decoded_final_ax}', flush=True)
if vr.first_mismatch:
    print('FIRST MISMATCH:', vr.first_mismatch, flush=True)
print('CFM FAST-PATH OK' if vr.all_matched and draft.final_ax_masked==344 else 'FAIL', flush=True)
sys.exit(0 if vr.all_matched else 1)
