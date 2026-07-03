"""Dump every token of a given step with marker names + decoded values."""
import os,sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES","1")
os.environ["C4_SMOKE_SPEC_K"]="0"; os.environ["C4_TEST_SPEC_K"]="0"
_H=os.path.dirname(os.path.abspath(__file__));_P=os.path.dirname(_H)
if _P not in sys.path: sys.path.insert(0,_P)
import contextlib,io
from tools.probe_groundtruth import build_groundtruth_probe
from neural_vm.batched_pure_neural import Token
from tests.test_suite_1000 import generate_test_programs
from src.compiler import compile_c
NAMES={int(v):k for k,v in vars(Token).items() if isinstance(v,int)}
pid=int(sys.argv[1]);want=int(sys.argv[2]);ms=int(sys.argv[3]) if len(sys.argv)>3 else 12
src,exp,desc=generate_test_programs()[pid];bc=compile_c(src)[0]
with contextlib.redirect_stderr(io.StringIO()):
    p=build_groundtruth_probe()
ctx=p._final_context(bc,max_steps=ms);pl=len(p._build_context(bc))
SE=int(Token.STEP_END)
steps=[];cur=[]
for i in range(pl,len(ctx)):
    cur.append((i,ctx[i]))
    if ctx[i]==SE: steps.append(cur);cur=[]
if cur: steps.append(cur)
toks=steps[want]
print(f"id{pid} step {want}: {len(toks)} tokens")
for i,t in toks:
    print(f"  pos {i}: tok {t:3d} {NAMES.get(t,'')}")
