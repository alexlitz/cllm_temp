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
NAMES={int(v):k for k,v in vars(Token).items() if isinstance(v,int)}
pid=int(sys.argv[1]) if len(sys.argv)>1 else 550
ms=int(sys.argv[2]) if len(sys.argv)>2 else 12
# positions to inspect: pass as comma list of EMIT rows (logits row = pos-1 of the emitted token)
emit_positions=[int(x) for x in sys.argv[3].split(",")] if len(sys.argv)>3 else None
src,exp,desc=generate_test_programs()[pid];bc=compile_c(src)[0]
with contextlib.redirect_stderr(io.StringIO()):
    p=build_groundtruth_probe()
ctx=p._final_context(bc,max_steps=ms);pl=len(p._build_context(bc))
SE=int(Token.STEP_END);REG_PC=int(Token.REG_PC)
model=p.model;dev=p._device
padded=torch.tensor([ctx],dtype=torch.long,device=dev)
logits=model.forward(padded)[0]
# find step-2 / step-6 starts: first token after a STEP_END that begins a 37-tok step
# just inspect emit rows where the emitted token != REG_PC right after a STEP_END
if emit_positions is None:
    emit_positions=[]
    for i in range(pl,len(ctx)):
        if ctx[i-1]==SE and ctx[i]!=REG_PC:
            emit_positions.append(i-1)  # logits row
            emit_positions.append(i)    # next logits row too
for er in emit_positions:
    row=logits[er]
    tk=torch.topk(row,6)
    pairs=[(int(t),NAMES.get(int(t),str(int(t))),round(float(v),1)) for v,t in zip(tk.values.tolist(),tk.indices.tolist())]
    print(f"emit row {er} (predicts ctx[{er+1}]={ctx[er+1]}={NAMES.get(ctx[er+1],ctx[er+1])}): logit[REG_PC=257]={float(row[REG_PC]):.1f}")
    print(f"    top6: {pairs}")
