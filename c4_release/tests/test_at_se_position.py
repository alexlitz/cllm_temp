"""Test THINKING_END at STEP_END position."""

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

# Add one complete VM step (35 tokens, ending at STEP_END position)
step_tokens = []
step_tokens.append(Token.REG_PC)
step_tokens.extend([0, 0, 0, 0])  # PC bytes
step_tokens.append(Token.REG_AX)
step_tokens.extend([0, 0, 0, 0])  # AX bytes
step_tokens.append(Token.REG_SP)
step_tokens.extend([0, 0, 0, 0])  # SP bytes
step_tokens.append(Token.REG_BP)
step_tokens.extend([0, 0, 0, 0])  # BP bytes
step_tokens.append(Token.STACK0)
step_tokens.extend([0, 0, 0, 0])  # STACK0 bytes
step_tokens.append(Token.MEM)
step_tokens.extend([0, 0, 0, 0, 0, 0, 0, 0])  # MEM section
# Now we're at 34 tokens, next should be STEP_END (position 34)

tokens.extend(step_tokens)

context = torch.tensor([tokens], dtype=torch.long)

# Set PRTF as active opcode
runner.model.set_active_opcode(33)

print(f"Context length: {len(tokens)}")
print(f"Last position (should be at STEP_END slot): {len(tokens) - 1}")

print("\nRunning forward pass with PRTF active...")
with torch.no_grad():
    x = runner.model.forward(context)
    last_x = runner.model.blocks[-1](runner.model.embed(context, active_opcode=33))
    for i in range(16):
        last_x = runner.model.blocks[i](last_x)

    print(f"\nAt STEP_END position (last token):")
    print(f"  NEXT_SE: {last_x[0, -1, BD.NEXT_SE]:.2f} (should be > 0 for STEP_END)")
    print(f"  CMP[3]: {last_x[0, -1, BD.CMP + 3]:.2f} (PRTF flag)")
    print(f"  NEXT_THINKING_END: {last_x[0, -1, BD.NEXT_THINKING_END]:.2f} (should be > 0)")

    logits = x[0, -1, :]
    print(f"\nOutput logits:")
    print(f"  THINKING_END: {logits[Token.THINKING_END]:.2f}")
    print(f"  STEP_END: {logits[Token.STEP_END]:.2f}")
    print(f"  REG_PC: {logits[Token.REG_PC]:.2f}")

    winner = logits.argmax().item()
    print(f"\n=> Winner: {winner}")
    if winner == Token.THINKING_END:
        print("✅ THINKING_END would be generated!")
    elif winner == Token.STEP_END:
        print("⚠️  STEP_END would be generated (normal VM behavior)")
    else:
        tok_name = "?"
        for attr in dir(Token):
            if not attr.startswith('_') and getattr(Token, attr) == winner:
                tok_name = attr
                break
        print(f"❌ {winner} ({tok_name}) would be generated")
