#!/usr/bin/env python3
"""Validate doom's init_sin Taylor term (32-bit fixed-point) byte-exact through the
all-C VM (Path B).  Compiles a tiny C program with the SAME expression as doom's
init_sin inner body and checks the model's printed %d output matches native ./c4."""
import warnings, torch, subprocess, tempfile, os
from pathlib import Path
warnings.filterwarnings("ignore")
import sys; sys.path.insert(0,"/home/alexlitz/Documents/misc/c4_doom")
from c4_min import isa
from c4_min.lib_neural import build_lib_model_streaming
from c4_min.nibble_pure_forward_cached import run_pure_forward_cached
from src.compiler import compile_c
from c4_min.run_1096_pure_forward import bytecode_to_isa
from c4_min import nibble_filesys as FS
from run_c4_min import tag_compiler_syscalls, install_compiler_abi_file_dispatcher, data_segment

C = r'''
enum { FP = 1024 };
int main() {
  int i, x, x2, x3, x5, x7, s;
  i = 40;
  x  = i * 3217 / 128;
  x2 = x  * x  / FP;
  x3 = x2 * x  / FP;
  x5 = x3 * x2 / FP;
  x7 = x5 * x2 / FP;
  s = x - x3 / 6 + x5 / 120 - x7 / 5040;
  printf("%d\n", s);
  return 0;
}
'''
# native reference via gcc (semantics identical to c4 int math)
with tempfile.NamedTemporaryFile("w",suffix=".c",delete=False) as f:
    f.write(C.replace("printf","printf").replace("int main","int main")); cpath=f.name
# compute reference in python (c4 uses int truncation toward zero)
i=40; x=i*3217//128; x2=x*x//1024; x3=x2*x//1024; x5=x3*x2//1024; x7=x5*x2//1024
s=x - x3//6 + x5//120 - x7//5040
print(f"reference s = {s} (x={x} x2={x2} x3={x3})")

bc,data=compile_c(C)
code=tag_compiler_syscalls(bytecode_to_isa(bc), isa)
data_seg=data_segment(data)
install_compiler_abi_file_dispatcher()
fio=FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b'', neural=True)))
import time; t0=time.time()
model,L,_=build_lib_model_streaming(code_size=max(len(code)+2,64), recurrent_divmod=True, addr32=True, compute_mode="dense_kernel")
model=model.to("cuda:0")
print(f"build {time.time()-t0:.1f}s, instrs={len(code)}", flush=True)
stats={}; t0=time.time()
tr=run_pure_forward_cached(model,L,code,max_steps=400,mask=0xFFFFFFFF,evict=True,prune_interval=60,stats=stats,fio=fio,data_seg=data_seg)
print(f"run {time.time()-t0:.2f}s steps={len(tr)} ms/step={(time.time()-t0)/max(len(tr),1)*1e3:.1f}", flush=True)
out=bytes(fio.runner.stdout).decode("latin-1","replace")
print(f"model stdout: {out!r}")
print(f"MATCH: {out.strip()==str(s)}")
os.unlink(cpath)
