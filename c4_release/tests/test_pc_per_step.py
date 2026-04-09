"""Track PC value at the start of each VM step."""

import sys
import os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token

c_code = '''
int main() {
    printf("H\\n");
    return 0;
}
'''

print("Compiling...")
code, data = compile_c(c_code)

print("\n" + "="*60)
print("TEST: conversational_io=True")
print("="*60)
runner = AutoregressiveVMRunner(conversational_io=True)
if torch.cuda.is_available():
    runner.model = runner.model.cuda()

# Track generated tokens
step_num = 0
token_count = 0
pc_values = []

original_generate = runner.model.generate_next
def track_generate(context):
    global step_num, token_count
    token = original_generate(context)

    # Check if this is the start of a new step (REG_PC at position 0, 35, 70, ...)
    if token == Token.REG_PC and len(context) % 35 == 0:
        step_num = len(context) // 35
        print(f"\nStep {step_num} starting...")

        # Next 4 tokens will be PC bytes
        pc_bytes = []
        for _ in range(4):
            context_copy = list(context) + [token] + pc_bytes
            byte_tok = original_generate(context_copy)
            pc_bytes.append(byte_tok & 0xFF)

        pc_val = sum(pc_bytes[k] << (k*8) for k in range(4))
        pc_values.append(pc_val)
        print(f"  PC = 0x{pc_val:04x} = {pc_val} decimal")
        print(f"  Instruction index = {pc_val // 6}")
        if pc_val % 6 != 0:
            print(f"  ⚠️ WARNING: PC is not at instruction boundary! (not multiple of 6)")

    token_count += 1
    return token

runner.model.generate_next = track_generate

print("\nRunning VM (max 6 steps)...")
output_str, exit_code = runner.run(code, data, [], max_steps=6)

print(f"\n✓ Execution complete, exit code: {exit_code}")
print(f"\nPC progression: {[f'0x{pc:04x}' for pc in pc_values]}")

# Check if any are misaligned
misaligned = [pc for pc in pc_values if pc % 6 != 0]
if misaligned:
    print(f"\n❌ MISALIGNED PCs found: {[f'0x{pc:04x}' for pc in misaligned]}")
else:
    print(f"\n✅ All PCs are properly aligned to instruction boundaries")
