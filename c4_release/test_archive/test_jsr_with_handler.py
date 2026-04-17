#!/usr/bin/env python3
"""Test if JSR works with handler enabled."""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

# Temporarily patch the runner to enable JSR handler
class TestRunner(AutoregressiveVMRunner):
    def _init_handlers(self):
        super()._init_handlers()
        # Re-enable JSR handler
        self._func_call_handlers[Opcode.JSR] = self._handler_jsr
        print("JSR handler enabled for testing")

TARGET_PC = 26
bytecode = [
    Opcode.JSR | (TARGET_PC << 8),
    Opcode.EXIT,
    Opcode.NOP,
    Opcode.IMM | (42 << 8),
    Opcode.EXIT,
]

runner = TestRunner()
result = runner.run(bytecode, b"", [], "")

print(f"Result with handler: {result}")
print(f"Expected: ('', 42)")

if result == ('', 42):
    print("\n✓ JSR works with handler enabled")
    print("  → Neural path is broken, handler is needed")
else:
    print(f"\n✗ Even handler doesn't work! Exit code: {result[1] if isinstance(result, tuple) else result}")
