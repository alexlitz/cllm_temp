#!/usr/bin/env python3
import warnings, torch, argparse
warnings.filterwarnings("ignore")
from c4_min import isa
from c4_min.lib_neural import build_lib_model_streaming
from c4_min.nibble_pure_forward_cached import run_pure_forward_cached
dev=torch.device("cuda:0")
model,L,_=build_lib_model_streaming(code_size=64, recurrent_divmod=True, addr32=True, compute_mode="dense_kernel")
model=model.to("cuda:0")
# small addr well below 0x100: does it work at all?
for addr in [0x20, 0x40, 0x100, 0x1040, 0x10040]:
    prog=[("IMM",200),("PSH",0),("IMM",addr),("SI",0),("IMM",addr),("LI",0),("HALT",0)]
    code=isa.assemble(prog)
    tr=run_pure_forward_cached(model,L,code,max_steps=32,mask=0xFFFFFFFF,evict=True, verbose=(addr==0x10040))
    print(f"  addr=0x{addr:x}: LI got={tr[-1]}  trace={tr}")
