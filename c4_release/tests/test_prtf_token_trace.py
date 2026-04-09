"""Trace all tokens generated during PRTF execution."""

import sys
import os
import torch
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

print("Compiling...")
code, data = compile_c(c_code)

print("\nCreating runner...")
runner = AutoregressiveVMRunner(conversational_io=True)
if torch.cuda.is_available():
    runner.model = runner.model.cuda()

# Track all tokens and opcode changes
tokens_generated = []
opcodes_seen = []

original_generate = runner.model.generate_next
original_set_opcode = runner.model.set_active_opcode

def track_tokens(context):
    token = original_generate(context)
    tokens_generated.append(token)
    return token

def track_opcode(opcode_value):
    opcodes_seen.append(opcode_value)
    print(f"[OPCODE] set_active_opcode({opcode_value}) at token {len(tokens_generated)}")
    original_set_opcode(opcode_value)

runner.model.generate_next = track_tokens
runner.model.set_active_opcode = track_opcode

print("\nRunning for 7 steps...")
try:
    output, exit_code = runner.run(code, data, [], max_steps=7)
    print(f"\nExit code: {exit_code}")
    print(f"Total tokens generated: {len(tokens_generated)}")

    # Find STEP_END tokens
    step_end_positions = [i for i, t in enumerate(tokens_generated) if t == Token.STEP_END]
    print(f"\nSTEP_END tokens at positions: {step_end_positions}")

    # Find THINKING_END tokens
    thinking_end_positions = [i for i, t in enumerate(tokens_generated) if t == Token.THINKING_END]
    print(f"THINKING_END tokens at positions: {thinking_end_positions}")

    if thinking_end_positions:
        print(f"\n✅ THINKING_END was generated!")
    else:
        print(f"\n❌ THINKING_END was never generated")

    print(f"\nOpcodes set: {len(opcodes_seen)}")
    for i, op in enumerate(opcodes_seen[:10]):
        print(f"  {i}: {op} (0x{op:02x})" if op is not None else f"  {i}: None")

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
