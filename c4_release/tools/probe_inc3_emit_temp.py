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
bc,_=compile_c(SRC); p=build_groundtruth_probe()
from neural_vm.unified_compiler.full_vm_compiler_dynamic import compile_full_vm_dynamic
_m,_l=compile_full_vm_dynamic(disk_cache=True); dp=dict(_l.dim_positions)
STEP=int(Token.STEP_TOKENS); pl=len(p._build_context(bc))
ctx=p._final_context(bc,max_steps=RS+2); padded=torch.tensor([ctx],device=p._device)
# Read at block BLK-1 (input to head-4 relay region, ~L12 block 17 reads after block 16)
x=td(p.model.forward(padded,stop_after_block=BLK)[0])
base=pl+RS*STEP
print(f"=== TEMP/BYTE_INDEX at emit rows blk{BLK} {SRC!r} STEP={STEP} ===")
for j in range(4,9):
    pos=base+j; row=x[pos]
    t8=float(row[dp["TEMP"]+8]); t9=float(row[dp["TEMP"]+9])
    bi0=float(row[dp["BYTE_INDEX_0"]]); ma=float(row[dp["MARK_AX"]])
    isb=float(row[dp["IS_BYTE"]])
    print(f"  in={j} tok={ctx[pos]:3d}: MARK_AX={ma:.1f} IS_BYTE={isb:.1f} BYTE_INDEX_0={bi0:.1f} TEMP+8={t8:.1f} TEMP+9={t9:.1f}")
