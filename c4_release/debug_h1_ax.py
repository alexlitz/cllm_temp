#!/usr/bin/env python3
"""Debug H1[AX_IDX] at PC vs AX byte positions."""
import sys
sys.path.insert(0, '.')
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode

BD = _SetDim
AX_IDX = 1

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
    pc_marker_pos = ctx_len  # Position 20
    pc_b0_pos = pc_marker_pos + 1  # Position 21
    ax_marker_pos = pc_marker_pos + 5  # Position 25
    ax_b0_pos = ax_marker_pos + 1  # Position 26

    # Run through L0 and L1
    h = model.blocks[0](h)
    h = model.blocks[1](h)

    print(f'After L1 (byte index assignment):')
    print(f'\nPC marker (pos {pc_marker_pos}):')
    print(f'  IS_BYTE: {h[0, pc_marker_pos, BD.IS_BYTE]:.2f}')
    print(f'  H1[AX_IDX]: {h[0, pc_marker_pos, BD.H1 + AX_IDX]:.2f}')

    print(f'\nPC byte 0 (pos {pc_b0_pos}):')
    print(f'  IS_BYTE: {h[0, pc_b0_pos, BD.IS_BYTE]:.2f}')
    print(f'  H1[AX_IDX]: {h[0, pc_b0_pos, BD.H1 + AX_IDX]:.2f}  ← Should be 0 (not AX)')
    print(f'  BYTE_INDEX_0: {h[0, pc_b0_pos, BD.BYTE_INDEX_0]:.2f}')
    print(f'  BYTE_INDEX_1: {h[0, pc_b0_pos, BD.BYTE_INDEX_1]:.2f}')
    print(f'  BYTE_INDEX_2: {h[0, pc_b0_pos, BD.BYTE_INDEX_2]:.2f}')
    print(f'  BYTE_INDEX_3: {h[0, pc_b0_pos, BD.BYTE_INDEX_3]:.2f}')

    print(f'\nAX marker (pos {ax_marker_pos}):')
    print(f'  IS_BYTE: {h[0, ax_marker_pos, BD.IS_BYTE]:.2f}')
    print(f'  H1[AX_IDX]: {h[0, ax_marker_pos, BD.H1 + AX_IDX]:.2f}')

    print(f'\nAX byte 0 (pos {ax_b0_pos}):')
    print(f'  IS_BYTE: {h[0, ax_b0_pos, BD.IS_BYTE]:.2f}')
    print(f'  H1[AX_IDX]: {h[0, ax_b0_pos, BD.H1 + AX_IDX]:.2f}  ← Should be 1 (is AX)')
    print(f'  BYTE_INDEX_0: {h[0, ax_b0_pos, BD.BYTE_INDEX_0]:.2f}')
    print(f'  BYTE_INDEX_1: {h[0, ax_b0_pos, BD.BYTE_INDEX_1]:.2f}')
    print(f'  BYTE_INDEX_2: {h[0, ax_b0_pos, BD.BYTE_INDEX_2]:.2f}')
    print(f'  BYTE_INDEX_3: {h[0, ax_b0_pos, BD.BYTE_INDEX_3]:.2f}')

    print(f'\n\nIf H1[AX_IDX] is 1 at PC byte 0, that\'s the bug - L10 byte passthrough will fire there!')
