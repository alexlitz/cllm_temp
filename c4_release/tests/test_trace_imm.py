"""Trace IMM instruction execution."""

import sys
import os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token

c_code = 'int main() { return 42; }'
code, data = compile_c(c_code)

print("Compiled instructions:")
for i, instr in enumerate(code):
    op = instr & 0xFF
    imm = instr >> 8
    print(f"  {i}: op={op} imm={imm} (0x{imm:08x})")

runner = AutoregressiveVMRunner(conversational_io=False)
if torch.cuda.is_available():
    runner.model = runner.model.cuda()

# Build context and run step-by-step
context = runner._build_context(code, data, [], '')
print(f"\nInitial context: {len(context)} tokens")

# Set initial opcode (JMP)
if len(code) > 0:
    runner.model.set_active_opcode(code[0] & 0xFF)
    print(f"Set opcode: {code[0] & 0xFF}")

# Generate steps until we see an AX marker
print("\nGenerating tokens...")
for step in range(5):
    print(f"\nStep {step}:")
    # Generate 35 tokens for one VM step
    step_start = len(context)
    for i in range(35):
        token = runner.model.generate_next(context)
        context.append(token)

    # Extract what happened in this step
    step_tokens = context[step_start:step_start+35]

    # Find REG_AX
    ax_idx = None
    for i, tok in enumerate(step_tokens):
        if tok == Token.REG_AX:
            ax_idx = i
            break

    if ax_idx and ax_idx + 4 < len(step_tokens):
        ax_bytes = step_tokens[ax_idx+1:ax_idx+5]
        ax_val = sum((b & 0xFF) << (j*8) for j, b in enumerate(ax_bytes))
        print(f"  AX bytes: {ax_bytes}")
        print(f"  AX value: {ax_val} (0x{ax_val:08x})")

    # Check PC
    pc_idx = None
    for i, tok in enumerate(step_tokens):
        if tok == Token.REG_PC:
            pc_idx = i
            break

    if pc_idx and pc_idx + 4 < len(step_tokens):
        pc_bytes = step_tokens[pc_idx+1:pc_idx+5]
        pc_val = sum((b & 0xFF) << (j*8) for j, b in enumerate(pc_bytes))
        print(f"  PC value: {pc_val} (0x{pc_val:08x})")

        # Determine which instruction this is
        instr_idx = pc_val // 4
        if 0 <= instr_idx < len(code):
            instr = code[instr_idx]
            op = instr & 0xFF
            imm = instr >> 8
            print(f"  Next instruction: {instr_idx} (op={op}, imm={imm})")
            # Set opcode for next step
            runner.model.set_active_opcode(op)
