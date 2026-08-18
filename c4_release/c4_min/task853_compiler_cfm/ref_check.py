"""Reference (no-GPU) check: run minic_mod.c as a program via the c4_min draft
(== native c4 semantics), capture the emitted bytecode text, then EXECUTE that
emitted bytecode on the same reference VM and confirm the module computes the
expected result.  This is the 'byte-exact vs native ./c4' correctness oracle
that the neural run must then reproduce byte-for-byte."""
import os, sys
os.environ['C4_DRAFT_READ_TO_MEM'] = '1'
os.environ['C4_DRAFT_CMP32'] = '1'
os.environ['C4_SHIFT32'] = '1'
WT = sys.argv[1]
sys.path.insert(0, WT)
sys.path.insert(0, '/home/alexlitz/Documents/misc/c4_doom')
from pathlib import Path
from c4_min import isa
from c4_min import nibble_filesys as FS
from c4_min.pf_speculative import draft_pf_program
from src.compiler import compile_c
from c4_min.run_1096_pure_forward import bytecode_to_isa
from run_c4_min import tag_compiler_syscalls, install_compiler_abi_file_dispatcher, data_segment

MINIC = sys.argv[2]
MODULE = sys.argv[3]

src = Path(MINIC).read_text()
bc, data = compile_c(src)
code = tag_compiler_syscalls(bytecode_to_isa(bc), isa)
data_seg = data_segment(data)
inp = Path(MODULE).read_bytes()
install_compiler_abi_file_dispatcher()
fio = FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}),
                                          stdin=FS.InputKVStream(inp, neural=True)))
d = draft_pf_program(code, max_steps=200000, mask=0xFFFFFFFF, data_seg=data_seg, fio=fio)
out = bytes(fio.runner.stdout).decode('latin1')
print(f"[ref] minic_mod instrs={len(code)} draft_steps={d.step_count} halted={d.halted}")
print("[ref] EMITTED BYTECODE:")
print(out)

# parse emitted "op imm\n" pairs into an Instr list
emitted = []
for ln in out.strip().split('\n'):
    ln = ln.strip()
    if not ln:
        continue
    op, imm = ln.split()
    emitted.append((int(op), int(imm)))
print(f"[ref] emitted {len(emitted)} instructions")
