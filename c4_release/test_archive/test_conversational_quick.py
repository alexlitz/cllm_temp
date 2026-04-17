#!/usr/bin/env python3
"""Quick conversational I/O test after PC fix."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Opcode, Token
import inspect

# Simple PRTF program
sig = inspect.signature(AutoregressiveVMRunner.__init__)
if 'conversational_io' in sig.parameters:
    runner = AutoregressiveVMRunner(conversational_io=True)
else:
    print("ERROR: conversational_io parameter not found")
    sys.exit(1)

# Track tokens
generated_tokens = []
original_gen = runner.model.generate_next

def track_gen(ctx):
    tok = original_gen(ctx)
    generated_tokens.append(tok)
    return tok

runner.model.generate_next = track_gen

# Program: PRTF "Hi", IMM 0, EXIT
bytecode = [
    Opcode.PRTF | (0x10000 << 8),
    Opcode.IMM | (0 << 8),
    Opcode.EXIT | (0 << 8)
]
data = b'Hi\x00'

print("Program: PRTF 'Hi', IMM 0, EXIT")
print(f"Expected: Output 'Hi', exit code 0")

print("\nRunning...")
output, exit_code = runner.run(bytecode, data, [], max_steps=10)

print(f"\nResults:")
print(f"  Output: {repr(output)}")
print(f"  Exit code: {exit_code}")
print(f"  Generated {len(generated_tokens)} tokens")

# Check for special tokens
thinking_end_count = sum(1 for t in generated_tokens if t == Token.THINKING_END)
step_end_count = sum(1 for t in generated_tokens if t == Token.STEP_END)
io_emit_byte_count = sum(1 for t in generated_tokens if t == Token.IO_STATE_EMIT_BYTE)

print(f"\nToken Counts:")
print(f"  THINKING_END: {thinking_end_count}")
print(f"  STEP_END: {step_end_count}")
print(f"  IO_STATE_EMIT_BYTE: {io_emit_byte_count}")

if exit_code == 0 and output == b'Hi':
    print("\n✅ PERFECT! Conversational I/O works end-to-end!")
elif thinking_end_count > 0:
    print("\n⚠️  THINKING_END generated but wrong output/exit code")
    print(f"    Expected output: b'Hi', got: {repr(output)}")
    print(f"    Expected exit: 0, got: {exit_code}")
elif output == b'Hi':
    print("\n⚠️  Output correct but no THINKING_END")
else:
    print("\n❌ Conversational I/O not working")
    print(f"    Missing THINKING_END and wrong output")
