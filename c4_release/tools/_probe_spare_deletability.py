#!/usr/bin/env python3
"""Deletability decode test: does idx104 (mul 23*65 = 1495) decode CORRECTLY
with the MUL SE-recover OFF once C4_ALU_CLEAR_SPARE_OPERAND fixes the root?

Runs the full spec_k=0 decode and prints the decoded exit code. Correct = 1495
(byte-1 present); dropped byte-1 = 215 (= 1495 & 0xFF)."""
import os
import sys
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_NO_STACK0_EMIT", "1")
os.environ.setdefault("C4_OPERAND_FROM_MEMSP", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")
from neural_vm.embedding import Opcode  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402


def _mk(ops):
    bc = []
    for op in ops:
        if isinstance(op, tuple):
            oc, imm = op
            bc.append(oc | (imm << 8))
        else:
            bc.append(op)
    return bc


CASES = {
    "mul_23x65 (idx104, expect 1495)": (
        _mk([(Opcode.IMM, 23), Opcode.PSH, (Opcode.IMM, 65), Opcode.MUL, Opcode.EXIT]), 1495),
    "mul_5x60  (expect 300)": (
        _mk([(Opcode.IMM, 5), Opcode.PSH, (Opcode.IMM, 60), Opcode.MUL, Opcode.EXIT]), 300),
    "gt_57_29  (expect 1)": (
        _mk([(Opcode.IMM, 57), Opcode.PSH, (Opcode.IMM, 29), Opcode.GT, Opcode.EXIT]), 1),
    "eq_7_45   (expect 0)": (
        _mk([(Opcode.IMM, 7), Opcode.PSH, (Opcode.IMM, 45), Opcode.EQ, Opcode.EXIT]), 0),
}


def main():
    probe = build_groundtruth_probe()
    print("C4_ALU_CLEAR_SPARE_OPERAND=%s  C4_MUL_L11_SE_RECOVER=%s  C4_CMP_BYTE0_SE_RECOVER=%s"
          % (os.environ.get("C4_ALU_CLEAR_SPARE_OPERAND", "0"),
             os.environ.get("C4_MUL_L11_SE_RECOVER", "1"),
             os.environ.get("C4_CMP_BYTE0_SE_RECOVER", "1")))
    for name, (bc, expect) in CASES.items():
        got = probe.emitted_result(bc, max_steps=20)[1]
        ok = "OK " if got == expect else "XX "
        print(f"  {ok} {name:36s} got={got} expect={expect}", flush=True)


if __name__ == "__main__":
    main()
