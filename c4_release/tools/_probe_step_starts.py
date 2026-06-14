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
pid=int(sys.argv[1]) if len(sys.argv)>1 else 550
ms=int(sys.argv[2]) if len(sys.argv)>2 else 12
src,exp,desc=generate_test_programs()[pid];bc=compile_c(src)[0]
with contextlib.redirect_stderr(io.StringIO()):
    p=build_groundtruth_probe()
ctx=p._final_context(bc,max_steps=ms);pl=len(p._build_context(bc))
SE=int(Token.STEP_END)
# split into steps
steps=[];cur=[]
for i in range(pl,len(ctx)):
    t=ctx[i];cur.append((i,t))
    if t==SE: steps.append(cur);cur=[]
if cur: steps.append(cur)
print(f"id{pid} {desc} prompt_len={pl} nsteps={len(steps)}")
for s,toks in enumerate(steps):
    n=len(toks)
    flag=" <==37!" if n==37 else (" <==EXTRA" if n>35 else "")
    lead=[(i,t,NAMES.get(t,str(t))) for i,t in toks[:4]]
    print(f"  step {s:2d}: ntok={n}{flag} lead={[(i,t) for i,t,_ in lead]}")
