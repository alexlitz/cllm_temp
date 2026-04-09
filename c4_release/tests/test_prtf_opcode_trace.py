"""Trace opcode setting during PRTF execution."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token

c_code = '''
int main() {
    printf("Hi");
    return 0;
}
'''

code, data = compile_c(c_code)
print(f"Compiled: {len(code)} instructions")

# Find PRTF instruction
prtf_idx = None
for i, instr in enumerate(code):
    op = instr & 0xFF
    if op == 33:  # PRTF
        prtf_idx = i
        print(f"PRTF at instruction {i}")
        break

runner = AutoregressiveVMRunner(conversational_io=True)

# Track opcode changes
opcodes_set = []
original_set_opcode = runner.model.set_active_opcode

def track_opcode(opcode_value):
    opcodes_set.append(opcode_value)
    if opcode_value == 33:
        print(f"[TRACE] set_active_opcode(33) called! (PRTF)")
    original_set_opcode(opcode_value)

runner.model.set_active_opcode = track_opcode

# Track token generation
tokens_generated = []
original_generate = runner.model.generate_next

def track_tokens(context):
    token = original_generate(context)
    tokens_generated.append(token)
    if token == Token.THINKING_END:
        print(f"[TRACE] THINKING_END generated at token {len(tokens_generated)}!")
    return token

runner.model.generate_next = track_tokens

print(f"\nRunning for {prtf_idx + 2} steps (to execute PRTF)...")
output, exit_code = runner.run(code, data, [], max_steps=prtf_idx + 2)

print(f"\nExecution complete:")
print(f"  Total tokens: {len(tokens_generated)}")
print(f"  Opcodes set: {len(opcodes_set)}")
print(f"  PRTF opcode (33) set: {33 in opcodes_set}")
print(f"  THINKING_END generated: {Token.THINKING_END in tokens_generated}")

if prtf_idx is not None:
    print(f"\nOpcodes set (first {prtf_idx + 2}):")
    for i, op in enumerate(opcodes_set[:prtf_idx + 2]):
        marker = " <-- PRTF" if op == 33 else ""
        print(f"  Step {i}: opcode {op}{marker}")
