import os,sys
_H=os.path.dirname(os.path.abspath(__file__)); _P=os.path.dirname(_H); sys.path.insert(0,_P)
os.environ["C4_SMOKE_SPEC_K"]="0"; os.environ["C4_TEST_SPEC_K"]="0"
import contextlib,io
with contextlib.redirect_stderr(io.StringIO()):
    from tools.probe_groundtruth import build_groundtruth_probe
    from neural_vm.batched_pure_neural import Token
    from tests.test_suite_1000 import generate_test_programs
    from src.compiler import compile_c
    probe=build_groundtruth_probe()
STEP=int(Token.STEP_TOKENS)
pid=int(sys.argv[1])
src,exp,desc=generate_test_programs()[pid]; bc=compile_c(src)[0]
ctx=probe._final_context(bc,max_steps=12); pl=len(probe._build_context(bc))
def dec(toks,reg):
    for i in range(len(toks)):
        if toks[i]==reg and i+4<len(toks):
            return sum((toks[i+1+j]&0xFF)<<(8*j) for j in range(4))
    return None
out=[]
for s in range(9):
    st=ctx[pl+s*STEP:pl+(s+1)*STEP]
    out.append(dec(st,int(Token.REG_PC)))
print(f"RESULT id{pid} u={probe.model.blocks[46].ffn.W_down.shape[1]} fixed35_PC={out}", flush=True)
