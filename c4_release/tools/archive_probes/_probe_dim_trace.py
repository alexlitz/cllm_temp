"""Trace dims 84/100 (and 69/85) across blocks at a specific row."""
import os,sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES","1")
os.environ["C4_SMOKE_SPEC_K"]="0"; os.environ["C4_TEST_SPEC_K"]="0"
_H=os.path.dirname(os.path.abspath(__file__));_P=os.path.dirname(_H)
if _P not in sys.path: sys.path.insert(0,_P)
import contextlib,io,torch
from tools.probe_groundtruth import build_groundtruth_probe
from neural_vm.batched_pure_neural import Token
from tests.test_suite_1000 import generate_test_programs
from src.compiler import compile_c
def _d(t):
    try:
        if t.layout!=torch.strided: return t.to_dense()
    except: pass
    return t
pid=int(sys.argv[1]);row=int(sys.argv[2]);ms=int(sys.argv[3]) if len(sys.argv)>3 else 12
DIMS=[69,84,85,100,6,7,17]
src,exp,desc=generate_test_programs()[pid];bc=compile_c(src)[0]
with contextlib.redirect_stderr(io.StringIO()):
    p=build_groundtruth_probe()
model=p.model;dev=p._device
ctx=p._final_context(bc,max_steps=ms)
padded=torch.tensor([ctx],dtype=torch.long,device=dev)
print(f"id{pid} row{row} (ctxtok={ctx[row]}) dims {DIMS}")
prev=None
for blk in range(len(model.blocks)):
    r=_d(model.forward(padded,stop_after_block=blk))[0,row]
    vals=[round(float(r[d]),1) for d in DIMS]
    if prev is None or any(abs(vals[i]-prev[i])>1.0 for i in range(len(DIMS))):
        print(f"  blk {blk:2d}: "+"  ".join(f"{d}={v}" for d,v in zip(DIMS,vals)))
    prev=vals
