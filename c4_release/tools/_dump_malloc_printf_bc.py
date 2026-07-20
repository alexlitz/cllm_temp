"""Dump the retargeted malloc_printf bytecode + literal-density stats."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from c4_min import isa
from c4_min import libprog_corpus as LC

entry = [e for e in LC.CORPUS if e.name == "malloc_printf"][0]
raw, data = LC.compile_entry(entry)
instrs = LC.retarget_to_neural_abi(raw)
print("n instrs:", len(instrs))
big = 0
for pc, ins in enumerate(instrs):
    flag = ""
    if abs(ins.imm) >= 256 or (ins.imm & 0xFFFFFFFF) >= 256:
        big += 1
        flag = "  <== BIG"
    if ins.op in (isa.LEA, isa.ENT, isa.ADJ, isa.LEV, isa.JSR):
        flag += "  [FRAME]"
    print(f"{pc:3d}  {isa.NAMES.get(ins.op, ins.op):5s} {ins.imm:>12d}  (0x{ins.imm & 0xFFFFFFFF:x}){flag}")
print(f"\nBIG literals (>=256): {big} / {len(instrs)}")
print("data seg bytes:", len(data))
