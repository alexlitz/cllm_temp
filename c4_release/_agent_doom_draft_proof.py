"""The FREE (draft-level) doom continuous proof: run doom's WHOLE init phase + first
printf via the draft (the perfect logical VM of the pure-forward model) with the full
composition (cmp32 + data_seg + compiler-ABI FileRunner), and verify the first output
bytes are byte-exact vs `printf 'q' | ./c4 doom.c`.  This is the byte-exact ground
truth the fast verify_blocks path reproduces on the model (proven equivalent step-for-
step on the doom-mechanism battery in _agent_composed_doomlike)."""
import sys, time
sys.path.insert(0,"/home/alexlitz/Documents/misc/c4_doom")
from pathlib import Path
from c4_min import isa
from src.compiler import compile_c
from c4_min.run_1096_pure_forward import bytecode_to_isa
from run_c4_min import tag_compiler_syscalls, install_compiler_abi_file_dispatcher, data_segment
from c4_min import nibble_filesys as FS
from c4_min.pf_speculative import draft_pf_program, _draft_cmp32

src = Path("/home/alexlitz/Documents/misc/c4_doom/doom.c").read_text()
bc, data = compile_c(src)
code = tag_compiler_syscalls(bytecode_to_isa(bc), isa)
data_seg = data_segment(data)
print(f"cmp32={_draft_cmp32()} doom instrs={len(code)} data={len(data or [])}", flush=True)

install_compiler_abi_file_dispatcher()
fio = FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b"q", neural=True)))
t0=time.time()
draft = draft_pf_program(code, max_steps=30000, mask=0xFFFFFFFF, data_seg=data_seg, fio=fio)
wall=time.time()-t0
out = bytes(fio.runner.stdout)
dm = sum(1 for f in draft.frames if f["op"] in ("DIV","MOD"))
print(f"draft: steps={draft.step_count} halted={draft.halted} first_prtf_step={draft.prtf_steps[:1]} "
      f"divmod_steps={dm} wall={wall:.2f}s", flush=True)
ref = Path("/tmp/doom_ref.bin").read_bytes()
n=min(len(out),len(ref)); match=0
for i in range(n):
    if out[i]==ref[i]: match+=1
    else: break
print(f"neural(draft) stdout first bytes = {out[:7].hex()}", flush=True)
print(f"reference ./c4 first bytes       = {ref[:7].hex()}", flush=True)
print(f"BYTE-EXACT PREFIX vs ./c4 = {match} bytes ({'ESC[2J ESC[H OK' if out[:7]==ref[:7] else 'MISMATCH'})", flush=True)
