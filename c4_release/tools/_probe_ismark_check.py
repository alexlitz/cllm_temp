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
IS_MARK=7; MARK_SE=5; MARK_SE_ONLY=10; IS_BYTE=6
BLK_IN=30
with contextlib.redirect_stderr(io.StringIO()):
    probe=build_groundtruth_probe()
model=probe.model;dev=probe._device;SE=int(Token.STEP_END);REG_SP=int(Token.REG_SP)
for pid in (550,575,250,700,950):
    try:
        src,exp,desc=generate_test_programs()[pid];bc=compile_c(src)[0]
        ctx=probe._final_context(bc,max_steps=12);pl=len(probe._build_context(bc))
        spur=[i-1 for i in range(pl,len(ctx)) if ctx[i]==255 and ctx[i-1]==SE]
        spb0=[i+1 for i in range(pl,len(ctx)-1) if ctx[i]==REG_SP]
        padded=torch.tensor([ctx],dtype=torch.long,device=dev)
        resid=_d(model.forward(padded,stop_after_block=BLK_IN))[0]
        print(f"id{pid} {desc}: spur_rows={spur[:4]}")
        for r in spur[:3]:
            v=resid[r];print(f"  SPUR r{r}: IS_MARK={float(v[IS_MARK]):.3f} MARK_SE={float(v[MARK_SE]):.3f} MSE_ONLY={float(v[MARK_SE_ONLY]):.3f} IS_BYTE={float(v[IS_BYTE]):.3f}")
        sp0mx=0.0
        for r in spb0:
            v=resid[r];sp0mx=max(sp0mx,abs(float(v[IS_MARK])))
        print(f"  GENUINE SP-byte0 rows: max|IS_MARK|={sp0mx:.4f} (n={len(spb0)})")
    except Exception as e:
        print(f"id{pid}: ERR {e}")
