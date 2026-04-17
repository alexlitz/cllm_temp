#!/usr/bin/env python3
"""Debug FETCH dimensions - do they contain the JSR target address?"""

import torch
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token, _SetDim as BD
from neural_vm.embedding import Opcode
from src.compiler import compile_c

# Simple program
code = 'int main() { return 42; }'
bytecode, data = compile_c(code)
target_pc = bytecode[0] >> 8
print(f"JSR target PC: {target_pc} = 0x{target_pc:04x}")
print(f"  Low byte: {target_pc & 0xFF} = 0x{target_pc & 0xFF:02x}")
print(f"  High byte: {(target_pc >> 8) & 0xFF} = 0x{(target_pc >> 8) & 0xFF:02x}")

# Create runner
runner = AutoregressiveVMRunner()
runner._func_call_handlers = {}

# Build context
runner._bytecode = bytecode
runner._last_sp = 0x1F800
runner._last_bp = 0x10000
ctx = runner._build_context(bytecode, data, [])

# Set active opcode
runner.model.set_active_opcode(Opcode.JSR)

# Generate first token (REG_PC marker)
next_token = runner.model.generate_next(ctx)
ctx.append(next_token)
print(f"\nGenerated token 0: {next_token} (REG_PC marker)")

# Now we're at the position where PC byte 0 will be generated
# Check internal state at this position
print("\n--- Internal state before PC byte 0 ---")
with torch.no_grad():
    device = next(runner.model.parameters()).device
    token_ids = torch.tensor([ctx[-512:] if len(ctx) > 512 else ctx], dtype=torch.long, device=device)

    # Get hidden state after embedding
    x = runner.model.embed(token_ids)
    print(f"After embedding, last pos FETCH_LO[0:8]: {[x[0, -1, BD.FETCH_LO + i].item() for i in range(8)]}")

    # After each layer
    for i, block in enumerate(runner.model.blocks):
        x = block(x)
        if i in [4, 5, 6]:  # L5 does fetch, L6 does routing
            fetch_lo = [x[0, -1, BD.FETCH_LO + j].item() for j in range(8)]
            fetch_hi = [x[0, -1, BD.FETCH_HI + j].item() for j in range(8)]
            output_lo = [x[0, -1, BD.OUTPUT_LO + j].item() for j in range(8)]
            print(f"After L{i}: FETCH_LO={fetch_lo[:4]}, OUTPUT_LO={output_lo[:4]}")

    # Final state
    last_pos = x[0, -1, :]
    print(f"\nFinal state at last position:")
    print(f"  FETCH_LO[0:16]: {[last_pos[BD.FETCH_LO + i].item() for i in range(16)]}")
    print(f"  FETCH_HI[0:16]: {[last_pos[BD.FETCH_HI + i].item() for i in range(16)]}")
    print(f"  OUTPUT_LO[0:16]: {[last_pos[BD.OUTPUT_LO + i].item() for i in range(16)]}")
    print(f"  OUTPUT_HI[0:16]: {[last_pos[BD.OUTPUT_HI + i].item() for i in range(16)]}")

    # Decode FETCH as a value
    fetch_val = 0
    for i in range(16):
        if last_pos[BD.FETCH_LO + i].item() > 0.5:
            fetch_val |= (1 << i)
        if last_pos[BD.FETCH_HI + i].item() > 0.5:
            fetch_val |= (1 << (i + 16))
    print(f"\n  Decoded FETCH value: {fetch_val} = 0x{fetch_val:08x}")
    print(f"  Expected target:     {target_pc} = 0x{target_pc:08x}")
