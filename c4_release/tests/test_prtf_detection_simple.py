"""Simple test to check if PRTF triggers THINKING_END."""

import sys
import os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token

c_code = '''
int main() {
    printf("Hello\\n");
    return 0;
}
'''

print("Compiling...")
code, data = compile_c(c_code)
print(f"✓ Compiled: {len(code)} instructions, {len(data)} data bytes")

# Disassemble to find PRTF
print("\nLooking for PRTF instruction:")
for i, instr in enumerate(code):
    op = instr & 0xFF
    if op == 33:  # PRTF
        print(f"  Instruction {i}: PRTF (0x21)")
        print(f"  PC value: 0x{i*6:04x}")

print("\nCreating runner with conversational_io=True...")
runner = AutoregressiveVMRunner(conversational_io=True)
if torch.cuda.is_available():
    runner.model = runner.model.cuda()

print("✓ Runner created")

# Track tokens to see if THINKING_END appears
generated_tokens = []
original_generate = runner.model.generate_next

def track_tokens(context, **kwargs):
    token = original_generate(context, **kwargs)
    generated_tokens.append(token)
    if token == Token.THINKING_END:
        step_num = len(context) // 35
        print(f"\n🎯 THINKING_END detected at step {step_num}!")
    elif token == Token.THINKING_START:
        step_num = len(context) // 35
        print(f"   THINKING_START at step {step_num}")
    return token

runner.model.generate_next = track_tokens

print("\nRunning VM (max 15 steps)...")
try:
    output_str, exit_code = runner.run(code, data, [], max_steps=15)
    print(f"\n✓ Execution complete")
    print(f"Exit code: {exit_code}")
    print(f"Output: {repr(output_str)}")

    # Check if THINKING_END was ever generated
    if Token.THINKING_END in generated_tokens:
        print(f"\n✅ SUCCESS: THINKING_END was generated!")
        thinking_end_count = generated_tokens.count(Token.THINKING_END)
        print(f"   Count: {thinking_end_count}")
    else:
        print(f"\n❌ FAILURE: THINKING_END was never generated")
        print(f"   Total tokens generated: {len(generated_tokens)}")

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
