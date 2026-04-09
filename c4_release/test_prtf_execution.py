#!/usr/bin/env python3
"""Diagnose PRTF execution after PC fix."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Opcode, Token
import inspect

# Simple PRTF program: just one PRTF call then EXIT
# PRTF should print "Hi" and then EXIT with code 0

sig = inspect.signature(AutoregressiveVMRunner.__init__)
if 'conversational_io' in sig.parameters:
    runner = AutoregressiveVMRunner(conversational_io=True)
else:
    runner = AutoregressiveVMRunner()

# Track generated tokens
generated_tokens = []
original_gen = runner.model.generate_next

def track_gen(ctx):
    tok = original_gen(ctx)
    generated_tokens.append(tok)
    return tok

runner.model.generate_next = track_gen

# Program: PRTF <string address>, IMM 0, EXIT
# String at 0x10000 = "Hi\0"
bytecode = [
    Opcode.PRTF | (0x10000 << 8),  # Print "Hi"
    Opcode.IMM | (0 << 8),          # AX = 0
    Opcode.EXIT | (0 << 8)          # Exit with code 0
]
data = b'Hi\x00'

print("Program:")
print(f"  PRTF 0x10000  (print 'Hi')")
print(f"  IMM 0")
print(f"  EXIT")
print(f"Expected: Output 'Hi', exit code 0")

print("\nRunning...")
output, exit_code = runner.run(bytecode, data, [], max_steps=10)

print(f"\nResults:")
print(f"  Output: {repr(output)}")
print(f"  Exit code: {exit_code} (0x{exit_code:08x})")
print(f"  Generated {len(generated_tokens)} tokens")

# Check for special tokens
thinking_end_count = sum(1 for t in generated_tokens if t == Token.THINKING_END)
step_end_count = sum(1 for t in generated_tokens if t == Token.STEP_END)
print(f"  THINKING_END count: {thinking_end_count}")
print(f"  STEP_END count: {step_end_count}")

if exit_code == 0 and output == b'Hi':
    print("\n✅ PERFECT! Conversational I/O is working!")
elif thinking_end_count > 0:
    print("\n⚠️  THINKING_END generated but wrong exit code/output")
else:
    print("\n❌ THINKING_END not generated - L10 FFN issue")
