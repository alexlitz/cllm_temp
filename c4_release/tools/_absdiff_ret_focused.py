import os, sys
sys.path.insert(0, os.path.abspath("."))
os.environ['C4_SMOKE_SPEC_K']='0'; os.environ['C4_TEST_SPEC_K']='0'
os.environ['C4_SKIP_DIM_INTEGRITY']='1'; os.environ['C4_SKIP_GATE_CHECK']='1'
import warnings; warnings.filterwarnings('ignore')
import torch
from tests.test_suite_1000 import generate_test_programs
from src.compiler import compile_c
from tools.interp_oracle_gate import build_gate_context, build_code_prompt, oracle_tape_and_steps
from neural_vm.verification.faithful_interpreter import STEP_TOKENS
ctx = build_gate_context(verbose=False)
progs = generate_test_programs()
def dec(v): return [(v>>(8*j))&0xFF for j in range(4)]
# absdiff sample + ALL func_mul (incl the passing 603/605/608)
IDS = [1046,1053,1059,1067,1047,1050] + list(range(600,610))
tag = os.environ.get('C4_ABSDIFF_RET_BYTE1','0')
print(f'=== focused (fix={"ON" if tag=="1" else "OFF"}): return-step AX vs exp ===', flush=True)
for pid in IDS:
    src,exp,desc=progs[pid]; bc,data=compile_c(src)
    ot=oracle_tape_and_steps(bc,data,max_steps=44)
    lev_s=[i for i,op in enumerate(ot.opcodes) if op==8]
    adj = lev_s[0]+1 if lev_s else len(ot.steps)-1
    prompt=build_code_prompt(bc,data); prefix=len(prompt); full=prompt+ot.draft_tokens
    logits=ctx.fwd.forward(full); fa=logits.argmax(dim=-1).tolist()
    def pred(t): return int(fa[prefix+t-1])
    base=adj*STEP_TOKENS; sl=[pred(base+k) for k in range(STEP_TOKENS)]
    ax=sum((sl[6+j]&0xFF)<<(8*j) for j in range(4))
    o_pc,o_ax=ot.steps[adj]
    cluster=desc.split(':')[0]
    print(f'  id{pid} {cluster:16s} exp={exp:5d}: return_AX={ax:5d} oracle_AX={o_ax:5d} {"OK" if ax==o_ax else "DIFF"}', flush=True)
