"""Test if IO_IS_PRTF is set when PRTF is active."""

import sys
import os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token, _SetDim as BD
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

c_code = 'int main() { printf("Hi"); return 0; }'

code, data = compile_c(c_code)
runner = AutoregressiveVMRunner(conversational_io=True)

# Build proper context
tokens = [Token.CODE_START]
for instr in code:
    op = instr & 0xFF
    imm = instr >> 8
    tokens.append(op)
    for i in range(IMMEDIATE_SIZE):
        tokens.append((imm >> (i * 8)) & 0xFF)
    for _ in range(PADDING_SIZE):
        tokens.append(0)
tokens.append(Token.CODE_END)
tokens.append(Token.DATA_START)
tokens.extend(data)
tokens.append(Token.DATA_END)
tokens.append(Token.THINKING_START)

# Add one VM step
for _ in range(35):
    tokens.append(0)

context = torch.tensor([tokens], dtype=torch.long)

# Set PRTF as active opcode
runner.model.set_active_opcode(33)

print("Running forward pass with PRTF active...")
with torch.no_grad():
    # Get embedding output
    x = runner.model.embed(context, active_opcode=33)

    print("\nEmbedding output (last position):")
    print(f"  ACTIVE_OPCODE_PRTF: {x[0, -1, BD.ACTIVE_OPCODE_PRTF]:.2f}")

    # Run through layers
    for i in range(16):
        x = runner.model.blocks[i](x)
        if i == 5:  # After L5 (where IO_IS_PRTF should be set)
            print(f"\nAfter L5 FFN (last position):")
            print(f"  IO_IS_PRTF: {x[0, -1, BD.IO_IS_PRTF]:.2f} (expected ~5.0)")

            # Check at AX marker position (where IO_IS_PRTF should be written)
            # Find AX marker in the step we added
            step_start = len(tokens) - 35
            ax_pos = step_start + 5  # REG_AX is 6th token in step (PC=5 tokens, then AX)
            print(f"  IO_IS_PRTF at AX pos ({ax_pos}): {x[0, ax_pos, BD.IO_IS_PRTF]:.2f}")

        if i == 6:  # After L6 (where it should be relayed)
            print(f"\nAfter L6 (last position):")
            print(f"  CMP[3]: {x[0, -1, BD.CMP + 3]:.2f}")

        if i == 15:  # After L15
            print(f"\nAfter L15 (last position):")
            print(f"  NEXT_THINKING_END: {x[0, -1, BD.NEXT_THINKING_END]:.2f}")

    # Check output
    logits = runner.model.head(x)[0, -1, :]
    print(f"\nOutput logits:")
    print(f"  THINKING_END: {logits[Token.THINKING_END]:.2f}")
    print(f"  REG_PC: {logits[Token.REG_PC]:.2f}")
    print(f"  STEP_END: {logits[Token.STEP_END]:.2f}")

    winner = logits.argmax().item()
    print(f"\n=> Winner: {winner}")
    if winner == Token.THINKING_END:
        print("✅ THINKING_END would be generated!")
    else:
        print(f"❌ {winner} would be generated instead of THINKING_END")
