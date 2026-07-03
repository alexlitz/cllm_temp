import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES","0")
os.environ["C4_SMOKE_SPEC_K"]="0"; os.environ["C4_TEST_SPEC_K"]="0"
os.environ["C4_SKIP_DIM_INTEGRITY"]="1"; os.environ["C4_SKIP_GATE_CHECK"]="1"
import warnings; warnings.filterwarnings("ignore")
import sys
_HERE=os.path.dirname(os.path.abspath(__file__))
_ROOT=os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0,_ROOT)
import torch
from src.compiler import compile_c
from neural_vm.batched_pure_neural import Token
from tools.probe_groundtruth import build_groundtruth_probe
SRC=os.environ.get("PROBE_SRC","int main() { return 827 - 26; }")
RS=int(os.environ.get("PROBE_STEP","3")); BLK=int(os.environ.get("PROBE_BLK","16"))
def td(w): return (w.to_dense() if w.layout!=torch.strided else w).detach().cpu().float()
def top(row,base,n=16,k=3):
    v=[(i,float(row[base+i])) for i in range(n)]; v=[(i,x) for i,x in v if abs(x)>0.3]
    v.sort(key=lambda z:-abs(z[1])); return " ".join(f"[{i}]={x:.1f}" for i,x in v[:k]) or "."
bc,_=compile_c(SRC); p=build_groundtruth_probe()
from neural_vm.unified_compiler.full_vm_compiler_dynamic import compile_full_vm_dynamic
_m,_l=compile_full_vm_dynamic(disk_cache=True); dp=dict(_l.dim_positions)
STEP=int(Token.STEP_TOKENS); pl=len(p._build_context(bc))
ctx=p._final_context(bc,max_steps=RS+2); padded=torch.tensor([ctx],device=p._device)
sb_lo,sb_hi=dp["STACK0_BYTE_VAL_1_LO"],dp["STACK0_BYTE_VAL_1_HI"]
x=td(p.model.forward(padded,stop_after_block=BLK)[0])
lo=pl+(RS-1)*STEP; hi=min(pl+(RS+1)*STEP,len(ctx))
print(f"=== STACK0_BYTE_VAL_1 across rows blk{BLK} {SRC!r} ===")
for pos in range(lo,hi):
    sl=top(x[pos],sb_lo); sh=top(x[pos],sb_hi)
    if sl!="." or sh!=".":
        ins=(pos-pl)%STEP; stp=(pos-pl)//STEP; tok=ctx[pos] if pos<len(ctx) else -1
        print(f"  pos{pos} s{stp} in={ins:2d} tok={tok:3d}: SBV1_LO {sl} | SBV1_HI {sh}")
