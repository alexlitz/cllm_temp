#!/usr/bin/env python3
"""CPU check: does the family-#2 (and #3) ADDR_B0_HI+14 gate break the SI/LI
memory roundtrip?  Decodes the AX result of the SI/LI smoke program
(0x200<-42; LI -> 42) via the faithful per-step decode (interp_oracle_gate),
flag-controlled by the shell env C4_OUTPUT_SELFREINFORCE_DECOUPLE.  Also runs
the three sweep targets to confirm the gate doesn't introduce a NEW flat
divergence.  GPU-free.
"""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from neural_vm.embedding import Opcode
from interp_oracle_gate import build_gate_context, classify_program


def asm(ops):
    bc = []
    for op in ops:
        if isinstance(op, tuple):
            opcode, imm = op
            bc.append(int(opcode) | (imm << 8))
        else:
            bc.append(int(op))
    return bc


def main():
    flag = os.environ.get("C4_OUTPUT_SELFREINFORCE_DECOUPLE", "0")
    print(f"C4_OUTPUT_SELFREINFORCE_DECOUPLE={flag}")
    ctx = build_gate_context()

    progs = {
        "si_li_roundtrip(42)": asm([
            (Opcode.IMM, 0x200), Opcode.PSH, (Opcode.IMM, 42), Opcode.SI,
            (Opcode.IMM, 0x200), Opcode.LI, Opcode.EXIT]),
        "si_li_zero(0)": asm([
            (Opcode.IMM, 0x300), Opcode.PSH, (Opcode.IMM, 0), Opcode.SI,
            (Opcode.IMM, 0x300), Opcode.LI, Opcode.EXIT]),
        "shr_8bit(0x100>>8=1)": asm([
            (Opcode.IMM, 0x100), Opcode.PSH, (Opcode.IMM, 8), Opcode.SHR,
            Opcode.EXIT]),
        "cmp_and_branch(42)": asm([
            (Opcode.IMM, 5), Opcode.PSH, (Opcode.IMM, 5), Opcode.EQ,
            (Opcode.BZ, 6), (Opcode.IMM, 42), Opcode.EXIT]),
    }
    for name, bc in progs.items():
        r = classify_program(ctx, name, bc, b"", max_steps=30, attribute=False)
        print(f"  {r.classification:12} {name}: {r.note or ''} (steps={r.n_steps})")


if __name__ == "__main__":
    main()
