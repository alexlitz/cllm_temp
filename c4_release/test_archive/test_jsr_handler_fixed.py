#!/usr/bin/env python3
"""Test JSR with handler properly enabled."""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

# Properly patch the runner
class TestRunner(AutoregressiveVMRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Re-enable JSR handler AFTER parent init
        self._func_call_handlers[Opcode.JSR] = self._handler_jsr
        print("JSR handler enabled for testing")

TARGET_PC = 26
bytecode = [
    Opcode.JSR | (TARGET_PC << 8),  # Instruction 0: JSR 26
    Opcode.EXIT,                     # Instruction 1: EXIT (shouldn't execute)
    Opcode.NOP,                      # Instruction 2: padding
    Opcode.IMM | (42 << 8),         # Instruction 3 at PC=26: IMM 42
    Opcode.EXIT,                     # Instruction 4: EXIT 42
]

print("Testing JSR with handler enabled...")
runner = TestRunner()
result = runner.run(bytecode, b"", [], "")

print(f"\nResult: {result}")
print(f"Expected: ('', 42)")

if result == ('', 42):
    print("\n✓ JSR works with handler")
    print("  → Issue is with neural path, not fundamental JSR logic")
else:
    exit_code = result[1] if isinstance(result, tuple) else result
    print(f"\n✗ JSR still fails even with handler! Exit code: {exit_code}")
    print("  → There may be a deeper issue with JSR execution")
