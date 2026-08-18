"""Full reference oracle: (1) run minic_mod.c as a program (via the c4_min draft ==
native c4 semantics) on the MODULE, capturing the emitted bytecode; (2) prepend a
thunk (JSR main; PSH; EXIT) and EXECUTE the emitted bytecode on the SAME draft VM
with the doom stack base; report the value main() returns.  This is the
'byte-exact vs native ./c4' correctness oracle for the compiled module."""
import os, sys
os.environ['C4_DRAFT_READ_TO_MEM'] = '1'
os.environ['C4_DRAFT_CMP32'] = '1'
os.environ['C4_SHIFT32'] = '1'
os.environ.setdefault('C4_MEM_ADDR_BITS', '18')
WT = sys.argv[1]
sys.path.insert(0, WT); sys.path.insert(0, '/home/alexlitz/Documents/misc/c4_doom')
from pathlib import Path
import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
import c4_min.nibble_pure_forward_cached as _PFCa
_PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = 0x10000
from c4_min import isa
from c4_min import nibble_filesys as FS
from c4_min.pf_speculative import draft_pf_program
from src.compiler import compile_c
from c4_min.run_1096_pure_forward import bytecode_to_isa
from run_c4_min import tag_compiler_syscalls, install_compiler_abi_file_dispatcher, data_segment

MINIC = sys.argv[2]
MODULE = sys.argv[3]
MAIN_IDX = int(sys.argv[4])   # instruction index of main()'s ENT in the emitted stream

# --- pass 1: run minic_mod on the module, get emitted bytecode text ---
mc = Path(MINIC).read_text()
bc, data = compile_c(mc)
code = tag_compiler_syscalls(bytecode_to_isa(bc), isa)
data_seg = data_segment(data)
inp = Path(MODULE).read_bytes()
install_compiler_abi_file_dispatcher()
fio = FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}),
                                          stdin=FS.InputKVStream(inp, neural=True)))
d = draft_pf_program(code, max_steps=300000, mask=0xFFFFFFFF, data_seg=data_seg, fio=fio)
out = bytes(fio.runner.stdout).decode('latin1')
pairs = []
for ln in out.strip().split('\n'):
    ln = ln.strip()
    if ln:
        o, m = ln.split()
        pairs.append((int(o), int(m)))
print(f"[exec] minic emitted {len(pairs)} instrs (draft steps {d.step_count}, halted {d.halted})")

# --- pass 2: [JSR main; PSH; EXIT] thunk + body; retarget branch imms by +THUNK ---
IMM, JSR, PSH, EXIT = isa.IMM, isa.JSR, isa.PSH, isa.HALT
THUNK = 3
branch_ops = {JSR, isa.JMP, isa.BZ, isa.BNZ}
body = [(o, (m + THUNK) if o in branch_ops else m) for (o, m) in pairs]
prog_pairs = [(JSR, MAIN_IDX + THUNK), (PSH, 0), (EXIT, 0)] + body
prog = tag_compiler_syscalls([isa.Instr(o, m & 0xFFFFFFFF) for (o, m) in prog_pairs], isa)

install_compiler_abi_file_dispatcher()
fio2 = FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}),
                                           stdin=FS.InputKVStream(b"", neural=True)))
d2 = draft_pf_program(prog, max_steps=300000, mask=0xFFFFFFFF, data_seg=[], fio=fio2)
last_ax = d2.frames[-1]['ax'] if getattr(d2, 'frames', None) else None
def sgn(v):
    return v - (1 << 32) if v is not None and (v & (1 << 31)) else v
print(f"[exec] executed emitted bytecode: steps={d2.step_count} halted={d2.halted} "
      f"main()->AX={last_ax} (signed {sgn(last_ax)})")
