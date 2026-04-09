"""Step by step trace to find where NEXT_THINKING_START is set."""

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

context = torch.tensor([prefix], dtype=torch.long, device=next(runner.model.parameters()).device)

# Set opcode for first instruction
runner.model.set_active_opcode(code[0] & 0xFF)

print("Generating first token (should be REG_PC)...")
with torch.no_grad():
    logits = runner.model.forward(context)
    last_logits = logits[0, -1, :]

# Check key dimensions at the last position
print("\nActivations at last position:")
# Need to run forward again to get intermediate activations
token_ids = context
x = runner.model.embed(token_ids, active_opcode=runner.model._active_opcode)

print(f"  LAST_WAS_THINKING_START: {x[0, -1, BD.LAST_WAS_THINKING_START]:.4f}")
print(f"  LAST_WAS_THINKING_END: {x[0, -1, BD.LAST_WAS_THINKING_END]:.4f}")
print(f"  LAST_WAS_BYTE: {x[0, -1, BD.LAST_WAS_BYTE]:.4f}")

# Run through layers and check L3 output
for i, block in enumerate(runner.model.blocks):
    x = block(x)
    if i == 2:  # After L2 (which has lookback detection)
        print(f"\nAfter L2 (lookback detection):")
        print(f"  LAST_WAS_THINKING_START: {x[0, -1, BD.LAST_WAS_THINKING_START]:.4f}")
        print(f"  LAST_WAS_THINKING_END: {x[0, -1, BD.LAST_WAS_THINKING_END]:.4f}")
    if i == 3:  # After L3 (which sets IO_IN_OUTPUT_MODE)
        print(f"\nAfter L3 (state init):")
        print(f"  IO_IN_OUTPUT_MODE: {x[0, -1, BD.IO_IN_OUTPUT_MODE]:.4f}")
    if i == 15:  # After L15
        print(f"\nAfter L15:")
        print(f"  NEXT_THINKING_START: {x[0, -1, BD.NEXT_THINKING_START]:.4f}")
        print(f"  NEXT_PC: {x[0, -1, BD.NEXT_PC]:.4f}")

# Check top predictions
top5 = torch.topk(last_logits, 5)
print(f"\nTop 5 predictions:")
for val, idx in zip(top5.values, top5.indices):
    tok_name = "?"
    for attr in dir(Token):
        if not attr.startswith('_') and getattr(Token, attr) == idx.item():
            tok_name = attr
            break
    if idx < 256:
        tok_name = f"byte_{idx.item()}"
    print(f"  {idx.item():3d} ({tok_name:20s}): {val.item():7.2f}")

reg_pc_logit = last_logits[Token.REG_PC].item()
thinking_start_logit = last_logits[Token.THINKING_START].item()
print(f"\nKey logits:")
print(f"  REG_PC: {reg_pc_logit:7.2f}")
print(f"  THINKING_START: {thinking_start_logit:7.2f}")

if reg_pc_logit > thinking_start_logit:
    print("\n✅ REG_PC would be generated (correct)")
else:
    print(f"\n❌ THINKING_START would be generated (wrong!)")
    print(f"   Difference: {thinking_start_logit - reg_pc_logit:.2f}")
