#!/usr/bin/env python3
"""Quick test of binary operations."""
import sys
sys.path.insert(0, '.')

from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode

def test_single_step(bytecode, desc, ax_init=0, bp_init=65536, sp_init=65536):
    """Test a single step operation."""
    draft = DraftVM(bytecode)
    draft.ax = ax_init
    draft.bp = bp_init
    draft.sp = sp_init
    draft.step()
    return draft.ax

# Test AND: AX = AX & imm
tests = [
    ([Opcode.AND | (5 << 8)], "AND 5 (AX=0)", 0, 0 & 5),
    ([Opcode.AND | (255 << 8)], "AND 255 (AX=15)", 15, 15 & 255),
    ([Opcode.OR | (5 << 8)], "OR 5 (AX=0)", 0, 0 | 5),
    ([Opcode.OR | (240 << 8)], "OR 240 (AX=15)", 15, 15 | 240),
    ([Opcode.XOR | (255 << 8)], "XOR 255 (AX=0)", 0, 0 ^ 255),
    ([Opcode.XOR | (240 << 8)], "XOR 240 (AX=15)", 15, 15 ^ 240),
    ([Opcode.MUL | (3 << 8)], "MUL 3 (AX=4)", 4, 4 * 3),
]

print("Testing DraftVM binary ops:")
for bytecode, desc, ax_init, expected in tests:
    result = test_single_step(bytecode, desc, ax_init=ax_init)
    status = "PASS" if result == expected else f"FAIL (got {result})"
    print(f"  {desc}: expected {expected}, {status}")

# Now test with neural VM runner (if it works)
print("\nTesting NeuralVM binary ops:")
try:
    runner = NeuralVMRunner()

    for bytecode, desc, ax_init, expected in tests:
        # Create a program that sets AX, then does the op, then exits
        prog = [Opcode.IMM | (ax_init << 8)] + bytecode + [Opcode.EXIT]
        result = runner.run(prog, max_steps=3)
        status = "PASS" if result == expected else f"FAIL (got {result})"
        print(f"  {desc}: expected {expected}, {status}")
except Exception as e:
    print(f"  Error: {e}")
