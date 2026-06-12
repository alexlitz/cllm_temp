import os, sys
os.environ.setdefault('C4_TEST_SPEC_K','0'); os.environ.setdefault('C4_SMOKE_SPEC_K','0')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from neural_vm.embedding import Opcode
from tools.probe_groundtruth import build_groundtruth_probe, Token
def _mk(ops): return [o[0]|(o[1]<<8) if isinstance(o,tuple) else o for o in ops]
# stress eq_false with small/close operands + a few eq_true
P=[('7!=8',7,8,0),('0!=1',0,1,0),('3!=4',3,4,0),('8!=9',8,9,0),
   ('7==7',7,7,1),('1==1',1,1,1),('100==100',100,100,1),('255==255',255,255,1),
   ('10!=20',10,20,0),('42==42',42,42,1)]
p=build_groundtruth_probe(); m=p.model; runner=p.runner
bad=0
for nm,a,b,exp in P:
    bc=_mk([(Opcode.IMM,a),Opcode.PSH,(Opcode.IMM,b),Opcode.EQ,Opcode.EXIT])
    res=runner.run_batch([bc],max_steps=20,spec_k=0,bucket_by_predicted_length=False)
    got=res[0][1]; ok=(got==exp)
    if not ok: bad+=1
    print(f'  {nm:12} exp={exp} got={got} {"OK" if ok else "*** BAD"}')
print('BAD:',bad)
