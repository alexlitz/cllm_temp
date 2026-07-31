#!/usr/bin/env python3
"""Prove the EXACT doom first-output printf emits byte-exact ESC bytes through Path B.
Minimal C program with doom's literal main() first printf: printf("%c[2J%c[H", ESC, ESC).
The 7 emitted bytes (1b 5b 32 4a 1b 5b 48) must byte-match the ./c4 reference prefix."""
import warnings, torch
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
enum { ESC = 27 };
int main() {
  printf("%c[2J%c[H", ESC, ESC);
  return 0;
}
'''
bc,data=compile_c(C)
code=tag_compiler_syscalls(bytecode_to_isa(bc), isa)
data_seg=data_segment(data)
install_compiler_abi_file_dispatcher()
fio=FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b'', neural=True)))
import time; t0=time.time()
model,L,_=build_lib_model_streaming(code_size=max(len(code)+2,64), recurrent_divmod=True, addr32=True, compute_mode="dense_kernel")
model=model.to("cuda:0")
print(f"build {time.time()-t0:.1f}s instrs={len(code)}", flush=True)
stats={}; t0=time.time()
tr=run_pure_forward_cached(model,L,code,max_steps=200,mask=0xFFFFFFFF,evict=True,prune_interval=60,stats=stats,fio=fio,data_seg=data_seg)
run_s=time.time()-t0
out=bytes(fio.runner.stdout)
print(f"steps={len(tr)} run={run_s:.2f}s ms/step={run_s/max(len(tr),1)*1e3:.1f}", flush=True)
print(f"model stdout hex = {out.hex()}")
ref=Path("/tmp/doom_ref.bin").read_bytes()
print(f"ref  prefix  hex = {ref[:len(out)].hex()}")
match=0
for i in range(min(len(out),len(ref))):
    if out[i]==ref[i]: match+=1
    else: break
print(f"byte_exact_prefix vs ./c4 = {match} bytes ({'ESC[2JESC[H OK' if out[:7]==bytes([27,91,50,74,27,91,72]) else 'MISMATCH'})")
