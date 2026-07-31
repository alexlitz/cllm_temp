#!/usr/bin/env python3
"""Measure Path B (all-C VM) build time + ms/step + memory, and confirm it emits
doom's first ESC bytes.  Build ONCE, run the doom code with a modest step budget,
time it, and check the stdout prefix vs the native ./c4 reference."""
import warnings, torch, time, argparse
from pathlib import Path
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0,"/home/alexlitz/Documents/misc/c4_doom")
from c4_min import isa
from c4_min.lib_neural import build_lib_model_streaming
from c4_min.nibble_pure_forward_cached import run_pure_forward_cached
from src.compiler import compile_c
from c4_min.run_1096_pure_forward import bytecode_to_isa
from c4_min import nibble_filesys as FS
# compiler-ABI bridge from run_c4_min
from run_c4_min import tag_compiler_syscalls, install_compiler_abi_file_dispatcher, data_segment

ap=argparse.ArgumentParser()
ap.add_argument("--device",default="cuda:0")
ap.add_argument("--mode",default="dense_kernel")
ap.add_argument("--max-steps",type=int,default=200)
ap.add_argument("--prune-interval",type=int,default=60)
a=ap.parse_args()

src=Path("/home/alexlitz/Documents/misc/c4_doom/doom.c").read_text()
bc,data=compile_c(src)
code=tag_compiler_syscalls(bytecode_to_isa(bc), isa)
data_seg=data_segment(data)
print(f"doom instrs={len(code)} data={len(data or [])}", flush=True)

install_compiler_abi_file_dispatcher()
fio=FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b'q', neural=True)))

t0=time.time()
model,L,_=build_lib_model_streaming(code_size=max(len(code)+2,64), recurrent_divmod=True, addr32=True, compute_mode=a.mode)
if a.device!="cpu": model=model.to(a.device)
build_s=time.time()-t0
print(f"build_s={build_s:.1f}", flush=True)
if a.device.startswith("cuda"):
    torch.cuda.synchronize(); print(f"vram_after_build={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

stats={}
t0=time.time()
trace=run_pure_forward_cached(model,L,code,max_steps=a.max_steps,mask=0xFFFFFFFF,
    evict=True, prune_interval=a.prune_interval, stats=stats, fio=fio, data_seg=data_seg)
if a.device.startswith("cuda"): torch.cuda.synchronize()
run_s=time.time()-t0
stdout=bytes(fio.runner.stdout)
print(f"steps={len(trace)} run_s={run_s:.2f} ms/step={run_s/max(len(trace),1)*1e3:.1f}", flush=True)
print(f"max_seq_len={stats.get('max_seq_len')} max_cache_size={stats.get('max_cache_size')}", flush=True)
print(f"stdout_bytes={len(stdout)} tool_calls={len(fio.calls)}", flush=True)
if a.device.startswith("cuda"): print(f"vram_peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)
print(f"stdout_hex={stdout[:20].hex()}", flush=True)
# compare to native ref prefix
ref=Path("/tmp/doom_ref.bin").read_bytes()
match=0
for i in range(min(len(stdout),len(ref))):
    if stdout[i]==ref[i]: match+=1
    else: break
print(f"byte_exact_prefix={match} / ref {len(ref)}", flush=True)
