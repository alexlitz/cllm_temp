#!/usr/bin/env python3
"""Trace OUTPUT corruption at PC byte 0 position."""
import sys
sys.path.insert(0, '.')
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode

BD = _SetDim

# Simple test: IMM 42, EXIT
bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]

# Build context
context = [Token.CODE_START]
for instr in bytecode:
    opcode = instr & 0xFF
    imm = (instr >> 8) & 0xFFFFFFFF
    context.append(opcode)
    for i in range(4):
        context.append((imm >> (i * 8)) & 0xFF)
    context.extend([0, 0, 0])
context.extend([Token.CODE_END, Token.DATA_START, Token.DATA_END])

# Generate draft tokens
draft = DraftVM(bytecode)
draft.step()  # Step 0: IMM 42
draft_tokens = draft.draft_tokens()

# Forward pass
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

full_context = context + draft_tokens
device = model.embed.weight.device
x = torch.tensor([full_context], dtype=torch.long, device=device)

with torch.no_grad():
    h = model.embed(x)
    model._inject_code_addr_keys(x, h)
    model._inject_mem_store(x, h)

    ctx_len = len(context)
    pc_b0_pos = ctx_len + 1  # Position 21

    def show_output(layer_name, h_state):
        """Show OUTPUT at PC byte 0 position."""
        lo_vals = [h_state[0, pc_b0_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
        hi_vals = [h_state[0, pc_b0_pos, BD.OUTPUT_HI + k].item() for k in range(16)]
        # Find strongest values
        lo_max = max(enumerate(lo_vals), key=lambda x: abs(x[1]))
        hi_max = max(enumerate(hi_vals), key=lambda x: abs(x[1]))
        return f'{layer_name}: LO[{lo_max[0]}]={lo_max[1]:.2f}, HI[{hi_max[0]}]={hi_max[1]:.2f}'

    print(f'Tracing OUTPUT at PC byte 0 (position {pc_b0_pos}):')
    print(f'Expected: OUTPUT should encode byte 0 (next PC byte)')
    print(f'Problem: OUTPUT encodes byte 42 (AX value) instead\n')

    # Trace through layers
    for i in range(len(model.blocks)):
        h = model.blocks[i](h)
        print(show_output(f'After L{i}', h))
        # Stop if we see byte 42 appear
        lo_vals = [h[0, pc_b0_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
        hi_vals = [h[0, pc_b0_pos, BD.OUTPUT_HI + k].item() for k in range(16)]
        if abs(lo_vals[10]) > 1.5 and abs(hi_vals[2]) > 1.5:
            print(f'\n→ Byte 42 (0x2A) appeared at L{i}! LO[10]={lo_vals[10]:.2f}, HI[2]={hi_vals[2]:.2f}')
            print(f'   This layer is causing the corruption.')
            break
