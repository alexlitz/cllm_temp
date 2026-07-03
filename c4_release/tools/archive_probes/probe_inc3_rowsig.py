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
RS=int(os.environ.get("PROBE_STEP","3")); BLK=int(os.environ.get("PROBE_BLK","13"))
def td(w): return (w.to_dense() if w.layout!=torch.strided else w).detach().cpu().float()
bc,_=compile_c(SRC); p=build_groundtruth_probe()
from neural_vm.unified_compiler.full_vm_compiler_dynamic import compile_full_vm_dynamic
_m,_l=compile_full_vm_dynamic(disk_cache=True); dp=dict(_l.dim_positions)
STEP=int(Token.STEP_TOKENS); pl=len(p._build_context(bc))
ctx=p._final_context(bc,max_steps=RS+2); padded=torch.tensor([ctx],device=p._device)
x=td(p.model.forward(padded,stop_after_block=BLK)[0])
base=pl+RS*STEP
# print discriminating flags at in=4..9
flags=["MARK_AX","IS_BYTE","BYTE_INDEX_0","BYTE_INDEX_1","BYTE_INDEX_2","BYTE_INDEX_3","H1","ADDR_B0_LO","ADDR_B1_HI","OP_SUB","OP_ADD","STACK0_BYTE_VAL_1_LO"]
print(f"=== rowsig blk{BLK} {SRC!r} STEP={STEP} ===")
for j in range(4,10):
    pos=base+j; row=x[pos]
    on=[]
    for f in flags:
        d=dp.get(f)
        if d is None: continue
        # for banded, check base and +5/+8
        for off in (0,5,8):
            v=float(row[d+off])
            if abs(v)>0.4: on.append(f"{f}+{off}={v:.1f}")
    tok=ctx[pos] if pos<len(ctx) else -1
    print(f"  in={j} tok={tok:3d}: {on}")
