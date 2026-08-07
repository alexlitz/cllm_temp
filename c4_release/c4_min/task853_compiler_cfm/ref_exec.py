"""Full reference oracle: (1) run minic_mod.c as a program (via the c4_min draft ==
native c4 semantics) on the module, capturing the emitted bytecode; (2) prepend a
call thunk (JSR main; then HALT/EXIT the returned AX) and EXECUTE the emitted
bytecode on the SAME draft VM; report the value main() returns.  This is the
'byte-exact vs native ./c4' correctness oracle."""
import os, sys
os.environ['C4_DRAFT_READ_TO_MEM'] = '1'
os.environ['C4_DRAFT_CMP32'] = '1'
os.environ['C4_SHIFT32'] = '1'
WT = sys.argv[1]
sys.path.insert(0, WT); sys.path.insert(0, '/home/alexlitz/Documents/misc/c4_doom')
from pathlib import Path
from c4_min import isa
from c4_min import nibble_filesys as FS
from c4_min.pf_speculative import draft_pf_program
from src.compiler import compile_c
from c4_min.run_1096_pure_forward import bytecode_to_isa
from run_c4_min import tag_compiler_syscalls, install_compiler_abi_file_dispatcher, data_segment

MINIC = sys.argv[2]
MODULE = sys.argv[3]
MAIN_IDX = int(sys.argv[4]) if len(sys.argv) > 4 else None  # code index of main()

# --- pass 1: run minic_mod on the module, get emitted bytecode text ---
mc = Path(MINIC).read_text()
bc, data = compile_c(mc)
code = tag_compiler_syscalls(bytecode_to_isa(bc), isa)
data_seg = data_segment(data)
inp = Path(MODULE).read_bytes()
install_compiler_abi_file_dispatcher()
fio = FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}),
                                          stdin=FS.InputKVStream(inp, neural=True)))
d = draft_pf_program(code, max_steps=200000, mask=0xFFFFFFFF, data_seg=data_seg, fio=fio)
out = bytes(fio.runner.stdout).decode('latin1')
pairs = []
for ln in out.strip().split('\n'):
    ln = ln.strip()
    if ln:
        o, m = ln.split()
        pairs.append((int(o), int(m)))
print(f"[exec] minic emitted {len(pairs)} instrs (draft steps {d.step_count}, halted {d.halted})")

# --- pass 2: build an executable program = [JSR main; PSH; EXIT] thunk + body ---
# The emitted body starts at index 0.  We prepend a 2-instr thunk, so every JSR
# target index in the body shifts by +THUNK.  Re-target JSR/JMP/BZ/BNZ imms.
IMM, JSR, PSH, EXIT, ENT, LEV, ADJ, JMP, BZ, BNZ = (
    isa.IMM, isa.JSR, isa.PSH, isa.HALT, isa.ENT, isa.LEV, isa.ADJ,
    isa.JMP, isa.BZ, isa.BNZ)
THUNK = 3  # JSR main ; PSH ; EXIT
branch_ops = {JSR, JMP, BZ, BNZ}
body = []
for (o, m) in pairs:
    if o in branch_ops:
        m = m + THUNK
    body.append((o, m))

if MAIN_IDX is None:
    # heuristic: main() is the LAST ENT in the stream's function starts.  We accept
    # it as an explicit arg for correctness; fall back to 0.
    MAIN_IDX = 0
prog_pairs = [(JSR, MAIN_IDX + THUNK), (PSH, 0), (EXIT, 0)] + body
prog = [isa.Instr(o, m & 0xFFFFFFFF) for (o, m) in prog_pairs]
prog = tag_compiler_syscalls(prog, isa)

install_compiler_abi_file_dispatcher()
fio2 = FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}),
                                           stdin=FS.InputKVStream(b"", neural=True)))
d2 = draft_pf_program(prog, max_steps=200000, mask=0xFFFFFFFF, data_seg=[], fio=fio2)
# the EXIT frame's AX is the value main returned; read the last frame's ax
last_ax = d2.frames[-1]['ax'] if hasattr(d2, 'frames') and d2.frames else None
# signed interpretation
def sgn(v):
    return v - (1 << 32) if v & (1 << 31) else v
print(f"[exec] executed emitted bytecode: steps={d2.step_count} halted={d2.halted} "
      f"main()->AX={last_ax} (signed {sgn(last_ax) if last_ax is not None else None})")
