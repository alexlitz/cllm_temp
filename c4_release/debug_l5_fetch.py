#!/usr/bin/env python3
"""Debug L5 fetch addressing."""
import sys
sys.path.insert(0, '.')
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.embedding import Opcode

BD = _SetDim

# Program: IMM 42
bytecode = [Opcode.IMM | (42 << 8)]

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

print(f'Context: {context}')
print(f'  Position 0: CODE_START={Token.CODE_START}')
print(f'  Position 1: opcode={context[1]} (should be 1 for IMM)')
print(f'  Position 2: imm_byte0={context[2]} (should be 42)')

# Run neural VM through L5
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

device = model.embed.weight.device
x = torch.tensor([context], dtype=torch.long, device=device)

with torch.no_grad():
    h = model.embed(x)
    model._inject_code_addr_keys(x, h)

    print(f'\\nADDR_KEY after injection (first 3 positions):')
    for i in range(1, 4):
        addr_lo = h[0, i, BD.ADDR_KEY:BD.ADDR_KEY+16].tolist()
        addr_hi = h[0, i, BD.ADDR_KEY+16:BD.ADDR_KEY+32].tolist()
        # Decode address from one-hot
        lo_idx = addr_lo.index(max(addr_lo)) if max(addr_lo) > 0.5 else -1
        hi_idx = addr_hi.index(max(addr_hi)) if max(addr_hi) > 0.5 else -1
        addr_value = lo_idx + hi_idx * 16 if lo_idx >= 0 and hi_idx >= 0 else -1
        print(f'  Position {i} (token {x[0,i].item()}): addr_lo[{lo_idx}]=1, addr_hi[{hi_idx}]=1 → address={addr_value}')

    model._inject_mem_store(x, h)

    # Run through L3 (PC default)
    for i in range(4):
        h = model.blocks[i](h)

    pc_marker = 13  # Position after CODE_START, 8 bytes, CODE_END, DATA_START, DATA_END
    ax_marker = pc_marker + 5

    print(f'\\nAfter L3 (PC=0 default):')
    pc_lo = h[0, pc_marker, BD.OUTPUT_LO:BD.OUTPUT_LO+16].tolist()
    pc_hi = h[0, pc_marker, BD.OUTPUT_HI:BD.OUTPUT_HI+16].tolist()
    pc_lo_idx = pc_lo.index(max(pc_lo))
    pc_hi_idx = pc_hi.index(max(pc_hi))
    print(f'  PC marker: OUTPUT_LO[{pc_lo_idx}]=1, OUTPUT_HI[{pc_hi_idx}]=1 → PC={pc_lo_idx + pc_hi_idx*16}')

    # Run L4 (PC relay and PC+1)
    h = model.blocks[4](h)

    print(f'\\nAfter L4 (PC relay and PC+1):')
    embed_lo = h[0, ax_marker, BD.EMBED_LO:BD.EMBED_LO+16].tolist()
    embed_hi = h[0, ax_marker, BD.EMBED_HI:BD.EMBED_HI+16].tolist()
    temp_lo = h[0, ax_marker, BD.TEMP:BD.TEMP+16].tolist()
    temp_hi = h[0, ax_marker, BD.TEMP+16:BD.TEMP+32].tolist()

    embed_lo_idx = embed_lo.index(max(embed_lo))
    embed_hi_idx = embed_hi.index(max(embed_hi))
    temp_lo_idx = temp_lo.index(max(temp_lo))
    temp_hi_idx = temp_hi.index(max(temp_hi))

    print(f'  EMBED (opcode addr): EMBED_LO[{embed_lo_idx}]=1, EMBED_HI[{embed_hi_idx}]=1 → addr={embed_lo_idx + embed_hi_idx*16}')
    print(f'  TEMP (imm addr): TEMP[{temp_lo_idx}]=1, TEMP[{temp_hi_idx}]=1 → addr={temp_lo_idx + temp_hi_idx*16}')
    print(f'  Expected: EMBED=0 (opcode at addr 0), TEMP=1 (immediate at addr 1)')

    # Run L5 (fetch)
    h = model.blocks[5](h)

    print(f'\\nAfter L5 (fetch):')
    opcode_lo = h[0, ax_marker, BD.OPCODE_BYTE_LO:BD.OPCODE_BYTE_LO+16].tolist()
    opcode_hi = h[0, ax_marker, BD.OPCODE_BYTE_HI:BD.OPCODE_BYTE_HI+16].tolist()
    fetch_lo = h[0, ax_marker, BD.FETCH_LO:BD.FETCH_LO+16].tolist()
    fetch_hi = h[0, ax_marker, BD.FETCH_HI:BD.FETCH_HI+16].tolist()

    opcode_lo_idx = opcode_lo.index(max(opcode_lo))
    opcode_hi_idx = opcode_hi.index(max(opcode_hi))
    fetch_lo_idx = fetch_lo.index(max(fetch_lo))
    fetch_hi_idx = fetch_hi.index(max(fetch_hi))

    print(f'  OPCODE: OPCODE_LO[{opcode_lo_idx}]=1, OPCODE_HI[{opcode_hi_idx}]=1 → opcode={opcode_lo_idx + opcode_hi_idx*16}')
    print(f'  FETCH: FETCH_LO[{fetch_lo_idx}]=1, FETCH_HI[{fetch_hi_idx}]=1 → value={fetch_lo_idx + fetch_hi_idx*16}')
    print(f'  Expected: opcode=1 (IMM), immediate=42')
