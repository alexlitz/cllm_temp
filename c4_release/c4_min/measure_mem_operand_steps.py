"""measure_mem_operand_steps.py — the STEP-COUNT reduction of the memory-operand MAC.

Measures, on the SAME CPU model.forward, the number of VM steps (= model.forwards)
for a length-N dot product done two ways:

  1. INTERPRETED (the path being replaced): each element is
        IMM a; LI; PSH; IMM b; LI; MUL; PSH; IMM acc; LI; ADD
     — 10 model.forwards per MAC (the LEA/LI loads dominate; here IMM+LI is the
     2-step load, twice per element, plus the multiply, the push, and the
     accumulator load+add).  Plus the final store.  ~10N forwards.

  2. MEMORY-OPERAND MAC (C4_MEM_OPERAND): each element is ONE ``MAC [a],[b]``
     — 1 model.forward per element, the two CAM reads + the multiply-accumulate
     folded into that one instruction's forward.  N forwards.

Both are run to byte-exactness and the actual forward counts reported.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["C4_MEM_OPERAND"] = "1"

from c4_min import isa
from c4_min import nibble_mem_operand as MO
from c4_min import nibble_pure_forward_complete as PFC


def interpreted_dot_program(n, a_base=0x40, b_base=0x60):
    """The interpreted dot product acc=Σ a[i]*b[i] in the base ISA (each memory
    read is IMM addr; LI — the 2-step load the memory-operand mode replaces).

    Per element:  IMM a; LI; PSH; IMM b; LI; MUL; PSH; IMM acc; LI; ADD  (acc in mem)
    But keeping the accumulator in AX avoids the acc round-trip to memory; the
    canonical c4 codegen keeps it on the stack.  We use the AX-accumulator form:
        acc starts 0 in AX
        per element:  PSH(acc); IMM a; LI; PSH; IMM b; LI; MUL; ADD
                      -> push acc, load a, push a, load b, mul (b*a -> a*b),
                         add acc  => 8 steps/element
    which is the LEANEST interpreted MAC (no acc memory round-trip).  Even so it is
    8 model.forwards/element vs the memory-operand MAC's 1.
    """
    prog = []
    for i in range(n):
        prog += [
            isa.Instr(isa.PSH, 0),                 # push running acc (AX)
            isa.Instr(isa.IMM, a_base + 4 * i),    # AX = &a[i]
            isa.Instr(isa.LI, 0),                  # AX = a[i]        (the LOAD)
            isa.Instr(isa.PSH, 0),                 # push a[i]
            isa.Instr(isa.IMM, b_base + 4 * i),    # AX = &b[i]
            isa.Instr(isa.LI, 0),                  # AX = b[i]        (the LOAD)
            isa.Instr(isa.MUL, 0),                 # AX = a[i]*b[i]
            isa.Instr(isa.ADD, 0),                 # AX = acc + a[i]*b[i]
        ]
    prog.append(isa.Instr(isa.HALT, 0))
    return prog


def mac_dot_program(n, a_base=0x40, b_base=0x60):
    prog, mac_b = [], {}
    for i in range(n):
        mac_b[len(prog)] = b_base + 4 * i
        prog.append(isa.Instr(MO.MAC, a_base + 4 * i))
    prog.append(isa.Instr(isa.HALT, 0))
    return prog, mac_b


def main():
    N = 4
    avec = [3, 5, 2, 7][:N]
    bvec = [4, 1, 6, 2][:N]
    seed = {}
    for i in range(N):
        seed[0x40 + 4 * i] = avec[i]
        seed[0x60 + 4 * i] = bvec[i]
    import numpy as np
    expect = int(np.dot(avec, bvec)) & 0xFF

    # ---- interpreted path (complete VM) ----
    iprog = interpreted_dot_program(N)
    cmodel, cL = PFC.build_pure_forward_complete_model(code_size=len(iprog))
    itrace = PFC.run_pure_forward_complete(cmodel, cL, iprog, max_steps=256, seed_mem=seed)
    interp_steps = len(itrace)              # one trace entry per executed instruction
    interp_final = itrace[-1]

    # ---- memory-operand MAC path ----
    mprog, mac_b = mac_dot_program(N)
    mmodel, mL = MO.build_mem_operand_model(code_size=len(mprog))
    mtrace = MO.run_mem_operand(mmodel, mL, mprog, mac_b, max_steps=64, seed_mem=seed)
    mac_steps = len(mtrace)
    mac_final = mtrace[-1]

    print(f"dot product length N = {N}   (expect Σ a[i]b[i] & 0xFF = {expect})")
    print()
    print(f"  INTERPRETED (IMM;LI load per operand + MUL + ADD):")
    print(f"    {interp_steps} VM steps (model.forwards)  -> final AX {interp_final} "
          f"{'OK' if interp_final == expect else 'MISMATCH'}")
    print(f"    = {interp_steps / N:.1f} forwards / MAC element")
    print()
    print(f"  MEMORY-OPERAND MAC (one fused MAC per element):")
    print(f"    {mac_steps} VM steps (model.forwards)  -> final AX {mac_final} "
          f"{'OK' if mac_final == expect else 'MISMATCH'}")
    print(f"    = {mac_steps / N:.1f} forwards / MAC element")
    print()
    # exclude the trailing HALT step from the per-element rate for a fair 'work' count.
    interp_work = interp_steps - 1
    mac_work = mac_steps - 1
    print(f"  reduction: {interp_work} -> {mac_work} forwards "
          f"({interp_work / max(1, mac_work):.1f}x fewer) for the N={N} MACs")
    print(f"  per-MAC:  {interp_work / N:.0f} interpreted forwards -> "
          f"{mac_work / N:.0f} memory-operand forward")


if __name__ == "__main__":
    main()
