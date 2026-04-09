"""Debug what happens at first token generation."""

import sys
import os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token, _SetDim as BD

c_code = 'int main() { return 0; }'

code, data = compile_c(c_code)
runner = AutoregressiveVMRunner(conversational_io=True)

# Build prefix
prefix = [Token.CODE_START]
for instr in code:
    op = instr & 0xFF
    imm = [(instr >> (8 + i*8)) & 0xFF for i in range(4)]
    prefix.append(op)
    prefix.extend(imm)
prefix.append(Token.CODE_END)
prefix.append(Token.DATA_START)
prefix.extend(data)
prefix.append(Token.DATA_END)
prefix.append(Token.THINKING_START)

context = torch.tensor([prefix], dtype=torch.long)
runner.model.set_active_opcode(code[0] & 0xFF)

print(f"Last token in context: {context[0, -1].item()} (THINKING_START)")

# Run forward and inspect activations layer by layer
with torch.no_grad():
    token_ids = context
    x = runner.model.embed(token_ids, active_opcode=runner.model._active_opcode)

    # Check embedding outputs at last position
    print(f"\nEmbedding output at last position:")
    print(f"  MARK_THINKING_START: {x[0, -1, BD.MARK_THINKING_START]:.2f}")
    print(f"  MARK_THINKING_END: {x[0, -1, BD.MARK_THINKING_END]:.2f}")

    # L0
    x = runner.model.blocks[0](x)

    # L1
    x = runner.model.blocks[1](x)

    # L2 (has lookback detection)
    x = runner.model.blocks[2](x)
    print(f"\nAfter L2 (lookback detection) at last position:")
    print(f"  LAST_WAS_THINKING_START: {x[0, -1, BD.LAST_WAS_THINKING_START]:.2f}")
    print(f"  LAST_WAS_THINKING_END: {x[0, -1, BD.LAST_WAS_THINKING_END]:.2f}")
    print(f"  LAST_WAS_BYTE: {x[0, -1, BD.LAST_WAS_BYTE]:.2f}")

    # Also check at t-1 (second to last position)
    print(f"\nAt position t-1 (the THINKING_START token):")
    print(f"  MARK_THINKING_START: {x[0, -2, BD.MARK_THINKING_START]:.2f}")

    # L3 (has state init)
    x = runner.model.blocks[3](x)
    print(f"\nAfter L3 (state init) at last position:")
    print(f"  IO_IN_OUTPUT_MODE: {x[0, -1, BD.IO_IN_OUTPUT_MODE]:.2f}")

    # Run through remaining layers
    for i in range(4, 16):
        x = runner.model.blocks[i](x)

    print(f"\nAfter all blocks at last position:")
    print(f"  NEXT_PC: {x[0, -1, BD.NEXT_PC]:.2f}")
    print(f"  NEXT_THINKING_START: {x[0, -1, BD.NEXT_THINKING_START]:.2f}")
    print(f"  NEXT_THINKING_END: {x[0, -1, BD.NEXT_THINKING_END]:.2f}")
    print(f"  OUTPUT_LO[0]: {x[0, -1, BD.OUTPUT_LO]:.2f}")
    print(f"  OUTPUT_HI[0]: {x[0, -1, BD.OUTPUT_HI]:.2f}")

    # Check output head
    logits = runner.model.head(x)[0, -1, :]
    print(f"\nOutput logits:")
    print(f"  REG_PC: {logits[Token.REG_PC]:.2f}")
    print(f"  THINKING_START: {logits[Token.THINKING_START]:.2f}")
    print(f"  byte_0: {logits[0]:.2f}")

    winner = logits.argmax().item()
    print(f"\n=> Winner: {winner} (should be {Token.REG_PC})")
