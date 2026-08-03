#!/usr/bin/env python3
"""Does the all-C VM (build_lib_model_streaming + run_pure_forward_cached, mask=32)
decode a 32-bit IMM/MUL/32-bit-address SI/LI correctly?  This is Path B (what
run_c4_min.py drives).  If yes, the 4th wall is CFM-lean-specific (IMM 8-bit)."""
import warnings, torch, argparse
warnings.filterwarnings("ignore")
from c4_min import isa
from c4_min.lib_neural import build_lib_model_streaming
from c4_min.nibble_pure_forward_cached import run_pure_forward_cached
ap=argparse.ArgumentParser(); ap.add_argument("--device",default="cuda:0"); ap.add_argument("--mode",default="dense_kernel"); a=ap.parse_args()
dev=torch.device(a.device)
print("[build] all-C VM streaming addr32 mask32 ...", flush=True)
import time; t0=time.time()
model,L,_=build_lib_model_streaming(code_size=64, recurrent_divmod=True, addr32=True, compute_mode=a.mode)
if a.device!="cpu": model=model.to(a.device)
print(f"[built] {time.time()-t0:.1f}s", flush=True)
tests=[
 ("IMM1000*1000", [("IMM",1000),("PSH",0),("IMM",1000),("MUL",0),("HALT",0)], 1_000_000),
 ("70000/256",    [("IMM",70000),("PSH",0),("IMM",256),("DIV",0),("HALT",0)], 273),
 ("store@0x10040",[("IMM",200),("PSH",0),("IMM",0x10040),("SI",0),("IMM",0x10040),("LI",0),("HALT",0)],200),
]
for name,prog,want in tests:
    code=isa.assemble(prog)
    stats={}
    tr=run_pure_forward_cached(model,L,code,max_steps=32,mask=0xFFFFFFFF,evict=True,stats=stats)
    got=tr[-1] if tr else None
    print(f"  {'OK ' if got==want else 'FAIL'} {name:16s} want={want} got={got}  (steps={len(tr)})")
