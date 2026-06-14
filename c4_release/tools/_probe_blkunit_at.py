"""Find FFN units of block BLK driving output dims at a SPECIFIC row, + gates.
Usage: _probe_blkunit_at.py <id> <blk> <emit_row> [dims=84,100]"""
import os,sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES","1")
os.environ["C4_SMOKE_SPEC_K"]="0"; os.environ["C4_TEST_SPEC_K"]="0"
_H=os.path.dirname(os.path.abspath(__file__));_P=os.path.dirname(_H)
if _P not in sys.path: sys.path.insert(0,_P)
import contextlib,io,torch,torch.nn.functional as F
from tools.probe_groundtruth import build_groundtruth_probe
from neural_vm.batched_pure_neural import Token
from tests.test_suite_1000 import generate_test_programs
from src.compiler import compile_c
from neural_vm.unified_compiler.full_vm_compiler_dynamic import compile_full_vm_dynamic
def _d(t):
    try:
        if t.layout!=torch.strided: return t.to_dense()
    except: pass
    return t
pid=int(sys.argv[1]);BLK=int(sys.argv[2]);emit_row=int(sys.argv[3])
DIMS=[int(x) for x in sys.argv[4].split(",")] if len(sys.argv)>4 else [84,100]
ms=int(sys.argv[5]) if len(sys.argv)>5 else 12
src,exp,desc=generate_test_programs()[pid];bc=compile_c(src)[0]
with contextlib.redirect_stderr(io.StringIO()):
    p=build_groundtruth_probe()
    _,layout=compile_full_vm_dynamic(disk_cache=True, alu_mode='efficient')
inv={}
for k,v in layout.dim_positions.items(): inv.setdefault(int(v),[]).append(k)
model=p.model;dev=p._device
ctx=p._final_context(bc,max_steps=ms)
padded=torch.tensor([ctx],dtype=torch.long,device=dev)
x_in=_d(model.forward(padded,stop_after_block=BLK-1))[0,emit_row]
ffn=model.blocks[BLK].ffn
Wup=_d(ffn.W_up);bup=_d(ffn.b_up);Wg=_d(ffn.W_gate);bg=_d(ffn.b_gate);Wd=_d(ffn.W_down);bd=_d(ffn.b_down)
up=Wup@x_in+bup; gate=Wg@x_in+bg
h=F.silu(up)*gate
print(f"id{pid} blk{BLK} row{emit_row} dims{DIMS}")
for d in DIMS:
    contrib=Wd[d]*h
    top=torch.topk(contrib.abs(),4)
    print(f" dim {d} ({inv.get(d,[])}):")
    for v,u in zip(top.values.tolist(),top.indices.tolist()):
        u=int(u)
        if abs(float(contrib[u]))<1e-3: continue
        wu=Wup[u];nz=(wu.abs()>1e-6).nonzero().flatten().tolist()
        wg=Wg[u];nzg=(wg.abs()>1e-6).nonzero().flatten().tolist()
        wd=Wd[:,u];nzd=(wd.abs()>1e-6).nonzero().flatten().tolist()
        print(f"   unit {u}: down={float(Wd[d,u]):+.3f} h={float(h[u]):+.3e} contrib={float(contrib[u]):+.3e} b_up={float(bup[u]):+.1f}")
        print(f"     reads(up): "+", ".join(f"{dd}{inv.get(dd,[''])[0] if inv.get(dd) else ''}={round(float(wu[dd]),1)}[x={round(float(x_in[dd]),2)}]" for dd in nz[:30]))
        if nzg: print(f"     reads(gate): "+", ".join(f"{dd}{inv.get(dd,[''])[0] if inv.get(dd) else ''}={round(float(wg[dd]),1)}[x={round(float(x_in[dd]),2)}]" for dd in nzg[:10])+f" b_gate={float(bg[u]):+.1f}")
        print(f"     writes: "+", ".join(f"{dd}{inv.get(dd,[''])[0] if inv.get(dd) else ''}={round(float(wd[dd]),2)}" for dd in nzd[:10]))
