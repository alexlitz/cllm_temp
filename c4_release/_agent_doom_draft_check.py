"""Confirm the doom draft reaches the first printf (byte-exact ESC) — the target."""
import warnings; warnings.filterwarnings('ignore')
import os, sys, time
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
os.environ['C4_PF_CFM']='1'
os.environ.setdefault('C4_DRAFT_CMP32','1')
os.environ.setdefault('C4_MEM_ADDR_BITS','18')
sys.path.insert(0,'/home/alexlitz/Documents/misc/c4_doom')
import torch
from pathlib import Path
from c4_min import isa
from c4_min import nibble_filesys as FS
from c4_min.pf_speculative import draft_pf_program, _draft_cmp32
from src.compiler import compile_c
from c4_min.run_1096_pure_forward import bytecode_to_isa
from run_c4_min import tag_compiler_syscalls, install_compiler_abi_file_dispatcher, data_segment

assert _draft_cmp32()
src = Path('/home/alexlitz/Documents/misc/c4_doom/doom.c').read_text()
bc, data = compile_c(src)
code = tag_compiler_syscalls(bytecode_to_isa(bc), isa)
data_seg = data_segment(data)
ref = Path('/tmp/doom_ref.bin').read_bytes()
print(f'[draft] instrs={len(code)} ref first7={ref[:7].hex()}', flush=True)

install_compiler_abi_file_dispatcher()
fio = FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b"q", neural=True)))
t0=time.time()
draft = draft_pf_program(code, max_steps=30200, mask=0xFFFFFFFF, data_seg=data_seg, fio=fio)
draft_out = bytes(fio.runner.stdout)
print(f'[draft] steps={draft.step_count} halted={draft.halted} n_prtf={len(draft.prtf_steps)} '
      f'first_prtf_step={draft.prtf_steps[:1]} code_off={draft.code_off} '
      f'stdout={draft_out[:16].hex()} wall={time.time()-t0:.1f}s', flush=True)
n=min(len(draft_out),len(ref)); m=0
for i in range(n):
    if draft_out[i]==ref[i]: m+=1
    else: break
print(f'[draft] byte-exact prefix vs ./c4 = {m} bytes ({"ESC OK" if draft_out[:7]==ref[:7] else "MISMATCH"})', flush=True)
# tokens length at first printf
if draft.prtf_steps:
    fp = draft.prtf_steps[0]
    print(f'[draft] first printf @ step {fp}; win_start={draft.win_starts[fp]} total_tokens_to_there~{draft.win_starts[fp]}', flush=True)
