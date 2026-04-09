"""Test PC progression with simple return 42 program."""

import sys
import os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token

c_code = '''
int main() {
    return 42;
}
'''

print("Compiling...")
code, data = compile_c(c_code)

runner = AutoregressiveVMRunner(conversational_io=False)
if torch.cuda.is_available():
    runner.model = runner.model.cuda()

# Capture all tokens
all_tokens = []
original_generate = runner.model.generate_next
def capture_tokens(context):
    token = original_generate(context)
    all_tokens.append(token)
    return token

runner.model.generate_next = capture_tokens

print("\nRunning VM...")
output_str, exit_code = runner.run(code, data, [], max_steps=10)

print(f"Exit code: {exit_code}\n")

# Extract PC from each step
print("PC values at start of each step:")
for step in range(min(10, len(all_tokens) // 35)):
    start_idx = step * 35
    if start_idx + 5 <= len(all_tokens):
        if all_tokens[start_idx] == Token.REG_PC:
            pc_bytes = [all_tokens[start_idx + 1 + k] & 0xFF for k in range(4)]
            pc_val = sum(pc_bytes[k] << (k*8) for k in range(4))
            print(f"  Step {step}: PC = 0x{pc_val:04x} ({pc_val} dec), instr {pc_val//6}", end="")
            if pc_val % 6 != 0:
                print(f" ⚠️ MISALIGNED!")
            else:
                print()
