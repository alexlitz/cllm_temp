#!/usr/bin/env python3
import warnings, torch
warnings.filterwarnings("ignore")
from c4_min import isa
from c4_min.lib_neural import build_lib_model_streaming
from c4_min.nibble_pure_forward_cached import run_pure_forward_cached
model,L,_=build_lib_model_streaming(code_size=64, recurrent_divmod=True, addr32=True, compute_mode="dense_kernel")
model=model.to("cuda:0")
# C4 SI: PSH address, IMM value, SI => mem[addr]=value. Then IMM addr, LI.
for addr,val in [(0x40,200),(0x1040,77),(0x10040,123),(0x10044,55),(0x200000,99)]:
    prog=[("IMM",addr),("PSH",0),("IMM",val),("SI",0),("IMM",addr),("LI",0),("HALT",0)]
    code=isa.assemble(prog)
    tr=run_pure_forward_cached(model,L,code,max_steps=32,mask=0xFFFFFFFF,evict=True)
    print(f"  {'OK ' if tr[-1]==val else 'FAIL'} store {val}@0x{addr:x} -> LI got={tr[-1]}")
# and a DISCRIMINATION test: two addrs sharing low byte (0x10040 vs 0x40) must NOT alias.
print("--- aliasing discrimination (0x40 vs 0x10040, same low byte) ---")
prog=[("IMM",0x40),("PSH",0),("IMM",11),("SI",0),
      ("IMM",0x10040),("PSH",0),("IMM",99),("SI",0),
      ("IMM",0x40),("LI",0),("HALT",0)]
code=isa.assemble(prog)
tr=run_pure_forward_cached(model,L,code,max_steps=32,mask=0xFFFFFFFF,evict=True, verbose=True)
print(f"  load 0x40 (stored 11, NOT 99): got={tr[-1]} {'OK-no-alias' if tr[-1]==11 else 'ALIASED->wall#1'}")
