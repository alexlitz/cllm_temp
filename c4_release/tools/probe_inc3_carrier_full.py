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
RS=int(os.environ.get("PROBE_STEP","3")); BLK=int(os.environ.get("PROBE_BLK","17"))
def td(w): return (w.to_dense() if w.layout!=torch.strided else w).detach().cpu().float()
def top(row,base,n=16,k=2):
    v=[(i,float(row[base+i])) for i in range(n)]; v=[(i,x) for i,x in v if abs(x)>0.4]
    v.sort(key=lambda z:-abs(z[1])); return ",".join(f"{i}={x:.0f}" for i,x in v[:k]) or "."
bc,_=compile_c(SRC); p=build_groundtruth_probe()
from neural_vm.unified_compiler.full_vm_compiler_dynamic import compile_full_vm_dynamic
_m,_l=compile_full_vm_dynamic(disk_cache=True); dp=dict(_l.dim_positions)
STEP=int(Token.STEP_TOKENS); pl=len(p._build_context(bc))
ctx=p._final_context(bc,max_steps=RS+2); padded=torch.tensor([ctx],device=p._device)
x=td(p.model.forward(padded,stop_after_block=BLK)[0])
sb_lo=dp["STACK0_BYTE_VAL_1_LO"]
flagn=["MARK_AX","STACK0_BYTE1","TEMP","BYTE_INDEX_0","BYTE_INDEX_1","OP_SUB","OP_ADD"]
print(f"=== carrier+flags blk{BLK} {SRC!r} STEP={STEP} (looking for SBV1_LO) ===")
for stp in range(max(0,RS-1),RS+1):
  for j in range(STEP):
    pos=pl+stp*STEP+j
    if pos>=len(ctx): break
    row=x[pos]; sl=top(row,sb_lo)
    fl=[]
    for f in flagn:
        d=dp.get(f);
        if d is None: continue
        for off in (0,8,9):
            v=float(row[d+off])
            if abs(v)>0.4: fl.append(f"{f}+{off}={v:.0f}")
    if sl!="." or "STACK0_BYTE1+0" in str(fl):
        print(f"  s{stp} in={j:2d} tok={ctx[pos]:3d}: SBV1_LO[{sl}] {fl}")
