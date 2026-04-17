#!/usr/bin/env python3
"""Check which instruction is being executed in each step."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token, Opcode
import inspect

# Simple program: IMM 42, EXIT
# bytecode[0] = IMM 42 (opcode 1, imm 42)
# bytecode[1] = EXIT (opcode 34)
bytecode = [1 | (42 << 8), 34 | (0 << 8)]

print("Bytecode:")
print(f"  [0] = 0x{bytecode[0]:08x} → opcode={bytecode[0]&0xFF} (IMM), imm={bytecode[0]>>8}")
print(f"  [1] = 0x{bytecode[1]:08x} → opcode={bytecode[1]&0xFF} (EXIT), imm={bytecode[1]>>8}")

sig = inspect.signature(AutoregressiveVMRunner.__init__)
if 'conversational_io' in sig.parameters:
    runner = AutoregressiveVMRunner(conversational_io=False)
else:
    runner = AutoregressiveVMRunner()

# Hook to track what the model sees
model = runner.model
original_gen = model.generate_next
step_info = []

def track_gen(ctx):
    tok = original_gen(ctx)

    # Check if this is the start of a new step (REG_PC token)
    if tok == Token.REG_PC:
        step_num = len(step_info)
        print(f"\n=== Step {step_num} starting ===")

        # Find the CODE section in context to see what opcodes are there
        try:
            code_start_idx = ctx.index(Token.CODE_START)
            code_end_idx = ctx.index(Token.CODE_END)
            opcodes_in_context = []
            for i in range(code_start_idx + 1, code_end_idx, 6):  # Every 6 tokens (opcode + 5 bytes)
                if i < len(ctx):
                    opcodes_in_context.append(ctx[i])
            print(f"  Opcodes in CODE section: {opcodes_in_context}")
        except:
            pass

    return tok

model.generate_next = track_gen

print("\nRunning...")
output, exit_code = runner.run(bytecode, b'', [], max_steps=3)

print(f"\n\nFinal exit code: {exit_code}")
