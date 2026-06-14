import os,sys
_H=os.path.dirname(os.path.abspath(__file__)); _P=os.path.dirname(_H); sys.path.insert(0,_P)
os.environ["C4_SMOKE_SPEC_K"]="0"; os.environ["C4_TEST_SPEC_K"]="0"
import contextlib,io
_buf=io.StringIO()
with contextlib.redirect_stderr(_buf):
    from tools.probe_groundtruth import build_groundtruth_probe
    from neural_vm.batched_pure_neural import Token
    from tests.test_suite_1000 import generate_test_programs
    from src.compiler import compile_c
    probe=build_groundtruth_probe()
SE=int(Token.STEP_END)
pid=int(sys.argv[1])
src,exp,desc=generate_test_programs()[pid]; bc=compile_c(src)[0]
ctx=probe._final_context(bc,max_steps=12); pl=len(probe._build_context(bc))
counts=[]; cur=0; lead=[]; prev=SE
for t in ctx[pl:]:
    if t==SE: counts.append(cur+1); cur=0
    else: cur+=1
    if prev==SE and t!=SE and t<256: lead.append(t)
    prev=t
if cur: counts.append(cur)
sys.stderr.write("")
print(f"RESULT id{pid} block46units={probe.model.blocks[46].ffn.W_down.shape[1]} counts={counts} spurious_lead={lead}", flush=True)
