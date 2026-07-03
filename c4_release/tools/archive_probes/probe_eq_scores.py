import os, sys
os.environ.setdefault('C4_TEST_SPEC_K','0'); os.environ.setdefault('C4_SMOKE_SPEC_K','0')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from neural_vm.embedding import Opcode
from tools.probe_groundtruth import build_groundtruth_probe, Token
def _mk(ops): return [o[0]|(o[1]<<8) if isinstance(o,tuple) else o for o in ops]
# weights MUST match the engine
WM,WAH,WXH,WAL,WXL = 0.3,0.4,2.0,0.2,2.0
P=[('49==49',49,49,1),('41==41',41,41,1),('42==42',42,42,1),('21==21',21,21,1),
   ('16==9',16,9,0),('28==12',28,12,0),('50==11',50,11,0),('37==3',37,3,0),('10==20',10,20,0),
   ('30==45',30,45,0),('7==45',7,45,0)]
p=build_groundtruth_probe(); m=p.model; dp=m.dim_positions; dev=next(m.parameters()).device
REG_AX=int(Token.REG_AX)
for nm,a,b,exp in P:
    bc=_mk([(Opcode.IMM,a),Opcode.PSH,(Opcode.IMM,b),Opcode.EQ,Opcode.EXIT])
    ctx=p._final_context(bc,max_steps=20); S=len(ctx)
    ax=max(i for i in range(S) if ctx[i]==REG_AX and i+4<S)
    toks=torch.tensor([ctx],dtype=torch.long,device=dev)
    with torch.no_grad(): r=m.forward(toks,stop_after_block=11)[0]
    def g(base): 
        bb=dp.get(base); return [float(r[ax,bb+k].item()) for k in range(16)]
    ah,xh,al,xl=g('ALU_HI'),g('AX_CARRY_HI'),g('ALU_LO'),g('AX_CARRY_LO')
    best=-99; bh=bl=None
    for h in range(16):
        for l in range(16):
            s=WM+WAH*ah[h]+WXH*xh[h]+WAL*al[l]+WXL*xl[l]
            if s>best: best,bh,bl=s,h,l
    print(f'{nm:9} exp={exp} best_score={best:.3f} (h={bh},l={bl})')
