"""Simple test to show PC at each step."""

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
print("TEST 1: conversational_io=False (baseline)")
print("="*60)
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
output_str, exit_code = runner.run(code, data, [], max_steps=6)

print(f"Exit code: {exit_code}\n")

# Extract PC from each step
print("PC values at start of each step:")
for step in range(min(6, len(all_tokens) // 35)):
    start_idx = step * 35
    if start_idx + 5 <= len(all_tokens):
        # Check if first token is REG_PC
        if all_tokens[start_idx] == Token.REG_PC:
            pc_bytes = [all_tokens[start_idx + 1 + k] & 0xFF for k in range(4)]
            pc_val = sum(pc_bytes[k] << (k*8) for k in range(4))
            print(f"  Step {step}: PC = 0x{pc_val:04x} ({pc_val} dec), instr {pc_val//6}", end="")
            if pc_val % 6 != 0:
                print(f" ⚠️ MISALIGNED!")
            else:
                print()

# Now test with conversational_io=True
print("\n" + "="*60)
print("TEST 2: conversational_io=True")
print("="*60)
runner2 = AutoregressiveVMRunner(conversational_io=True)
if torch.cuda.is_available():
    runner2.model = runner2.model.cuda()

all_tokens2 = []
def capture_tokens2(context):
    token = runner2.model.generate_next.__wrapped__(context)
    all_tokens2.append(token)
    return token

original2 = runner2.model.generate_next
runner2.model.generate_next = capture_tokens2
runner2.model.generate_next.__wrapped__ = original2

print("\nRunning VM...")
output_str2, exit_code2 = runner2.run(code, data, [], max_steps=6)

print(f"Exit code: {exit_code2}\n")

print("PC values at start of each step:")
for step in range(min(6, len(all_tokens2) // 35)):
    start_idx = step * 35
    if start_idx + 5 <= len(all_tokens2):
        if all_tokens2[start_idx] == Token.REG_PC:
            pc_bytes = [all_tokens2[start_idx + 1 + k] & 0xFF for k in range(4)]
            pc_val = sum(pc_bytes[k] << (k*8) for k in range(4))
            print(f"  Step {step}: PC = 0x{pc_val:04x} ({pc_val} dec), instr {pc_val//6}", end="")
            if pc_val % 6 != 0:
                print(f" ⚠️ MISALIGNED!")
            else:
                print()
