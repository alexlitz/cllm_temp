#!/usr/bin/env python3
"""Test if 'successful' programs are actually executing correctly."""
import sys
sys.path.insert(0, '.')

from src.compiler import compile_c
from neural_vm.batch_runner_v2 import UltraBatchRunner
from neural_vm.speculative import DraftVM

print("=" * 70)
print("TESTING ACTUAL EXECUTION PATH")
print("=" * 70)

# Simple program that should execute JSR
code = 'int main() { return 42; }'
bytecode, _ = compile_c(code)

print(f"\nProgram: {code}")
print(f"Bytecode ({len(bytecode)} instructions):")
for i, instr in enumerate(bytecode):
    op = instr & 0xFF
    imm = instr >> 8
    opnames = {
        0: "LEA", 1: "IMM", 2: "JMP", 3: "JSR", 4: "BZ", 5: "BNZ",
        6: "ENT", 7: "ADJ", 8: "LEV", 38: "EXIT"
    }
    opname = opnames.get(op, f"OP{op}")
    print(f"  [{i}] {opname:4s} {imm}")

print("\n" + "=" * 70)
print("DRAFTVM EXECUTION TRACE")
print("=" * 70)

vm = DraftVM(bytecode)
print(f"\nStep 0: idx={vm.idx}, pc={vm.pc}, ax={vm.ax}")

for step in range(6):
    if vm.halted:
        print(f"\nProgram halted at step {step}")
        break

    instr = bytecode[vm.idx]
    op = instr & 0xFF
    imm = instr >> 8
    opnames = {
        0: "LEA", 1: "IMM", 2: "JMP", 3: "JSR", 4: "BZ", 5: "BNZ",
        6: "ENT", 7: "ADJ", 8: "LEV", 38: "EXIT"
    }
    opname = opnames.get(op, f"OP{op}")

    vm.step()
    print(f"Step {step+1}: Executed {opname:4s} {imm:3d} → idx={vm.idx}, pc={vm.pc}, ax={vm.ax}")

print(f"\nFinal: ax={vm.ax} (exit code)")

print("\n" + "=" * 70)
print("BATCH RUNNER EXECUTION (without strict)")
print("=" * 70)

runner = UltraBatchRunner(batch_size=1, strict=False)
result = runner.run_batch([bytecode], max_steps=100)

print(f"\nBatch runner result: {result[0]}")

if result[0] == 42:
    print("RESULT IS CORRECT (42)")
    print("But transformer can't execute JSR correctly!")
    print("So where is the correct result coming from?")
else:
    print(f"Result is wrong (expected 42, got {result[0]})")
