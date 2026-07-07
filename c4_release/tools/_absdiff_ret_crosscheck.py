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

# Sample across clusters. Cross-check: for each program, dump the per-step (PC,AX)
# flat decode. Run this script twice (fix OFF / ON) and diff the JSON to find any
# step that changed -> a candidate regression.
SAMPLE = list(range(550,560)) + list(range(600,610)) + [1046,1053,1059,1067] \
       + list(range(1,6)) + list(range(250,256)) + list(range(300,306))
ctx = build_gate_context(verbose=False)
progs = generate_test_programs()
import json
out = {}
for pid in SAMPLE:
    if pid >= len(progs): continue
    src,exp,desc = progs[pid]
    try:
        bc,data = compile_c(src)
        ot = oracle_tape_and_steps(bc,data,max_steps=44)
    except Exception:
        continue
    prompt = build_code_prompt(bc,data); prefix=len(prompt); full=prompt+ot.draft_tokens
    logits = ctx.fwd.forward(full); fa = logits.argmax(dim=-1).tolist()
    def pred(t): return int(fa[prefix+t-1])
    steps=[]
    for s in range(len(ot.steps)):
        base=s*STEP_TOKENS; sl=[pred(base+k) for k in range(STEP_TOKENS)]
        pc=sum((sl[1+j]&0xFF)<<(8*j) for j in range(4))
        ax=sum((sl[6+j]&0xFF)<<(8*j) for j in range(4))
        steps.append((pc,ax))
    out[pid]=steps
tag = os.environ.get('C4_ABSDIFF_RET_BYTE1','0')
fn = f'/tmp/absdiff_crosscheck_{tag}.json'
json.dump(out, open(fn,'w'))
print(f'wrote {fn} ({len(out)} programs)')
