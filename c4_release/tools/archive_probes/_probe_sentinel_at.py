"""Localize the sentinel emitter for token 255 at a SPECIFIC emit row.
Usage: _probe_sentinel_at.py <id> <emit_row> [token]"""
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
pid=int(sys.argv[1]);emit_row=int(sys.argv[2]);TOK=int(sys.argv[3]) if len(sys.argv)>3 else 255
ms=int(sys.argv[4]) if len(sys.argv)>4 else 12
src,exp,desc=generate_test_programs()[pid];bc=compile_c(src)[0]
with contextlib.redirect_stderr(io.StringIO()):
    p=build_groundtruth_probe()
model=p.model;dev=p._device
ctx=p._final_context(bc,max_steps=ms)
padded=torch.tensor([ctx],dtype=torch.long,device=dev)
W=_d(model.head.weight);bh=model.head.bias
b=_d(bh)[TOK] if bh is not None else 0.0
wT=W[TOK]
print(f"id{pid} emit_row={emit_row} token={TOK} ctxlen={len(ctx)} blocks={len(model.blocks)}")
prev=0.0
for blk in range(len(model.blocks)):
    r=_d(model.forward(padded,stop_after_block=blk))[0,emit_row]
    l=float((r*wT).sum())+float(b)
    d=l-prev
    if abs(d)>1e3:
        print(f"  block {blk}: logit={l:.4e} delta={d:+.4e}")
    prev=l
# diff at biggest jump
deltas=[]
prev=0.0
for blk in range(len(model.blocks)):
    r=_d(model.forward(padded,stop_after_block=blk))[0,emit_row]
    l=float((r*wT).sum())+float(b);deltas.append((blk,l-prev));prev=l
jb=max(deltas,key=lambda x:x[1])[0]
print(f"  >>> injecting block {jb}")
rb=_d(model.forward(padded,stop_after_block=jb-1))[0,emit_row]
ra=_d(model.forward(padded,stop_after_block=jb))[0,emit_row]
diff=(ra-rb)*wT
top=torch.topk(diff.abs(),10)
for v,dd in zip(top.values.tolist(),top.indices.tolist()):
    dd=int(dd);print(f"    dim {dd}: before={float(rb[dd]):+.3e} after={float(ra[dd]):+.3e} w={float(wT[dd]):+.2f} dcontrib={float(diff[dd]):+.3e}")
