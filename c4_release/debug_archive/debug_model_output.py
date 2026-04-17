#!/usr/bin/env python3
"""Debug why the model generates garbage tokens instead of correct values."""

import torch
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token, _SetDim as BD
from neural_vm.embedding import Opcode
from neural_vm.constants import INSTR_WIDTH
from src.compiler import compile_c

# Simple program
code = 'int main() { return 42; }'
bytecode, data = compile_c(code)
print(f"Bytecode[0]: JSR to {bytecode[0] >> 8}")
print(f"Main at instruction {(bytecode[0] >> 8 - 2) // 8}")

# Create runner with NO handlers
runner = AutoregressiveVMRunner()
runner._func_call_handlers = {}  # Remove ALL handlers
print(f"Handlers: {list(runner._func_call_handlers.keys())}")

# Build context
runner._bytecode = bytecode
runner._last_sp = 0x1F800
runner._last_bp = 0x10000
ctx = runner._build_context(bytecode, data, [])
print(f"Initial context length: {len(ctx)}")

# Set active opcode for first instruction (JSR)
first_op = bytecode[0] & 0xFF
runner.model.set_active_opcode(first_op)
print(f"Active opcode: {first_op} (JSR)")

# Generate tokens and examine logits
print("\n--- Generating first step ---")
print("Token | Logits top-5 | Expected")
print("-" * 60)

expected_tokens = [
    (Token.REG_PC, "REG_PC marker"),
    (None, "PC byte 0"),  # Should be target PC low byte
    (None, "PC byte 1"),
    (None, "PC byte 2"),
    (None, "PC byte 3"),
    (Token.REG_AX, "REG_AX marker"),
    (None, "AX byte 0"),  # Should preserve AX
    (None, "AX byte 1"),
    (None, "AX byte 2"),
    (None, "AX byte 3"),
    (Token.REG_SP, "REG_SP marker"),
]

for i, (expected, desc) in enumerate(expected_tokens):
    # Get logits before generating
    with torch.no_grad():
        device = next(runner.model.parameters()).device
        token_ids = torch.tensor([ctx[-512:] if len(ctx) > 512 else ctx], dtype=torch.long, device=device)
        logits = runner.model.forward(token_ids)[0, -1, :]

        # Get top 5 predictions
        top5_vals, top5_ids = torch.topk(logits, 5)
        top5_list = [(idx.item(), val.item()) for idx, val in zip(top5_ids, top5_vals)]

    # Generate token
    next_token = runner.model.generate_next(ctx)
    ctx.append(next_token)

    # Check if correct
    if expected is not None:
        status = "✓" if next_token == expected else "✗"
    else:
        status = "?"

    print(f"{i:2d}: {next_token:3d} {status} | top5: {[t[0] for t in top5_list]} | {desc}")

    if i >= 10:
        break

# Check what internal dimensions look like at generation positions
print("\n--- Internal state at last position ---")
with torch.no_grad():
    token_ids = torch.tensor([ctx[-512:]], dtype=torch.long, device=device)
    # Get hidden state after all layers
    x = runner.model.embed(token_ids)
    for block in runner.model.blocks:
        x = block(x)

    # Check key dimensions at last position
    last_pos = x[0, -1, :]
    output_lo = [last_pos[BD.OUTPUT_LO + i].item() for i in range(4)]
    output_hi = [last_pos[BD.OUTPUT_HI + i].item() for i in range(4)]
    print(f"OUTPUT_LO[0:4]: {output_lo}")
    print(f"OUTPUT_HI[0:4]: {output_hi}")
    print(f"MARK_PC: {last_pos[BD.MARK_PC].item():.2f}")
    print(f"MARK_AX: {last_pos[BD.MARK_AX].item():.2f}")
    print(f"OP_JSR: {last_pos[BD.OP_JSR].item():.2f}")
    print(f"OP_IMM: {last_pos[BD.OP_IMM].item():.2f}")
