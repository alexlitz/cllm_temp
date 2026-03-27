#!/usr/bin/env python3
"""Debug PSH instruction at step 1."""
import sys
sys.path.insert(0, '.')
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode

BD = _SetDim

# Program: IMM 5, PSH
bytecode = [Opcode.IMM | (5 << 8), Opcode.PSH]

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

# Execute step 0 (IMM 5)
draft = DraftVM(bytecode)
draft.step()
draft_tokens_0 = draft.draft_tokens()
context = context + draft_tokens_0

# Execute step 1 (PSH)
draft.step()
draft_tokens_1 = draft.draft_tokens()
print(f'DraftVM step 1: PC={draft.pc}, AX={draft.ax}, SP={draft.sp}')
print(f'PSH should: AX=5 (unchanged), SP=-8 (0xFFFFFFF8)')

# Forward pass
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

full_context = context + draft_tokens_1
device = model.embed.weight.device
x = torch.tensor([full_context], dtype=torch.long, device=device)

with torch.no_grad():
    h = model.embed(x)
    model._inject_code_addr_keys(x, h)
    model._inject_mem_store(x, h)

    ctx_len = len(context)
    pc_marker_pos = ctx_len
    ax_marker_pos = ctx_len + 5

    # Run through L3 to check PC carry-forward
    for i in range(4):
        h = model.blocks[i](h)

    print(f'\nAfter L3 (PC carry-forward) at PC marker (pos {pc_marker_pos}):')
    print(f'  EMBED_LO: {[f"{h[0, pc_marker_pos, BD.EMBED_LO + k]:.2f}" for k in range(16)]}')
    print(f'  EMBED_HI: {[f"{h[0, pc_marker_pos, BD.EMBED_HI + k]:.2f}" for k in range(4)]}')
    print(f'  Expected: PC=10 from step 0 (LO[10]=1, HI[0]=1)')

    # Run L4 attention (should relay PC to AX marker)
    h = model.blocks[4](h)

    print(f'\nAfter L4 (PC relay) at AX marker (pos {ax_marker_pos}):')
    print(f'  EMBED_LO: {[f"{h[0, ax_marker_pos, BD.EMBED_LO + k]:.2f}" for k in range(16)]}')
    print(f'  EMBED_HI: {[f"{h[0, ax_marker_pos, BD.EMBED_HI + k]:.2f}" for k in range(4)]}')
    print(f'  Expected: PC=10 (LO[10]=1, HI[0]=1) - but this gets modified by L4 FFN to PC-2=8')
    print(f'  Fetch address = PC-2 = 8, which is where PSH opcode is!')

    # Run L5
    h = model.blocks[5](h)

    print(f'\nAfter L5 (opcode decode) at AX marker (pos {ax_marker_pos}):')
    print(f'  OPCODE_BYTE_LO: {[f"{h[0, ax_marker_pos, BD.OPCODE_BYTE_LO + k]:.2f}" for k in range(16)]}')
    print(f'  OPCODE_BYTE_HI: {[f"{h[0, ax_marker_pos, BD.OPCODE_BYTE_HI + k]:.2f}" for k in range(16)]}')
    print(f'  Expected: PSH=13=0x0D, so LO[13]=1, HI[0]=1')
    print(f'  OP_PSH: {h[0, ax_marker_pos, BD.OP_PSH]:.4f} (should be ~5.0)')
    print(f'  OP_IMM: {h[0, ax_marker_pos, BD.OP_IMM]:.4f} (should be 0)')
    print(f'  AX_CARRY_LO: {[f"{h[0, ax_marker_pos, BD.AX_CARRY_LO + k]:.2f}" for k in range(16)]}')
    print(f'  AX_CARRY should be 5 from L3 carry-forward')

    # Run L6 (routing)
    h = model.blocks[6](h)

    print(f'\nAfter L6 (routing) at AX marker:')
    print(f'  OUTPUT_LO: {[f"{h[0, ax_marker_pos, BD.OUTPUT_LO + k]:.2f}" for k in range(16)]}')
    print(f'  OUTPUT_HI: {[f"{h[0, ax_marker_pos, BD.OUTPUT_HI + k]:.2f}" for k in range(4)]}')
    print(f'  Expected: OUTPUT should have AX=5 (LO[5]=1, HI[0]=1)')

    # Check SP marker too
    sp_marker_pos = ctx_len + 10
    print(f'\nAfter L6 at SP marker (pos {sp_marker_pos}):')
    print(f'  OUTPUT_LO: {[f"{h[0, sp_marker_pos, BD.OUTPUT_LO + k]:.2f}" for k in range(16)]}')
    print(f'  OUTPUT_HI: {[f"{h[0, sp_marker_pos, BD.OUTPUT_HI + k]:.2f}" for k in range(4)]}')
    print(f'  Expected: SP=-8 (0xFFFFFFF8 = LO[8]=1, HI[15]=1)')
    print(f'  (SP should be decremented by 8 for PSH)')
