#!/usr/bin/env python3
"""Debug step 0 to understand EMBED population."""
import sys
sys.path.insert(0, '.')
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode

def build_context(bytecode):
    context = [Token.CODE_START]
    for instr in bytecode:
        opcode = instr & 0xFF
        imm = (instr >> 8) & 0xFFFFFFFF
        context.append(opcode)
        for i in range(4):
            context.append((imm >> (i * 8)) & 0xFF)
        context.extend([0, 0, 0])
    context.append(Token.CODE_END)
    context.append(Token.DATA_START)
    context.append(Token.DATA_END)
    return context

def main():
    BD = _SetDim
    print('Building model...')
    model = AutoregressiveVM()
    set_vm_weights(model)
    model.eval()

    bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]
    context = build_context(bytecode)

    draft = DraftVM(bytecode)
    draft.step()
    draft_tokens = draft.draft_tokens()

    print(f'Context: {len(context)} tokens')
    print(f'Draft PC: {draft_tokens[1]} (should be 7)')
    print(f'Draft AX: {draft_tokens[6]} (should be 42)')

    full_context = context + draft_tokens

    with torch.no_grad():
        x = torch.tensor([full_context], dtype=torch.long)
        h = model.embed(x)
        model._inject_code_addr_keys(x, h)
        model._inject_mem_store(x, h)

        # Check CODE section for ADDR_KEY
        print(f'\\n=== CODE section (positions 1-8) ===')
        for i in range(1, 9):
            tok = x[0, i].item()
            addr_key = h[0, i, BD.ADDR_KEY:BD.ADDR_KEY+16].tolist()
            print(f'Pos {i}: token={tok}, ADDR_KEY={addr_key[:4]}...')

        # Run through layers with correct numbering
        ax_marker_pos = len(context) + 5

        # L0-L2
        for i in range(3):
            h = model.blocks[i](h)
        print(f'\\n=== After L2 (register carry) at AX marker pos {ax_marker_pos} ===')
        print(f'EMBED_LO: {h[0, ax_marker_pos, BD.EMBED_LO:BD.EMBED_LO+4].tolist()}')

        # L3
        h = model.blocks[3](h)
        print(f'\\n=== After L3 (PC relay) at AX marker pos {ax_marker_pos} ===')
        print(f'AX_CARRY_LO: {h[0, ax_marker_pos, BD.AX_CARRY_LO:BD.AX_CARRY_LO+4].tolist()}')

        # L4
        h = model.blocks[4](h)
        print(f'\\n=== After L4 (PC+1 relay) at AX marker pos {ax_marker_pos} ===')
        print(f'TEMP: {h[0, ax_marker_pos, BD.TEMP:BD.TEMP+4].tolist()}')

        # L5
        h = model.blocks[5](h)
        print(f'\\n=== After L5 (fetch+decode) at AX marker pos {ax_marker_pos} ===')
        print(f'OPCODE_BYTE_LO: {h[0, ax_marker_pos, BD.OPCODE_BYTE_LO:BD.OPCODE_BYTE_LO+4].tolist()}')
        print(f'FETCH_LO: {h[0, ax_marker_pos, BD.FETCH_LO:BD.FETCH_LO+4].tolist()}')
        print(f'OP_IMM: {h[0, ax_marker_pos, BD.OP_IMM]:.4f}')

        # L6
        h = model.blocks[6](h)
        print(f'\\n=== After L6 (routing) at AX_B0 pos {ax_marker_pos+1} ===')
        print(f'OUTPUT_LO[0]: {h[0, ax_marker_pos+1, BD.OUTPUT_LO]:.4f}')

        # Get final prediction
        for i in range(6, len(model.blocks)):
            h = model.blocks[i](h)
        logits = model.head(h)

        pc_b0_pred = logits[0, len(context), :].argmax().item()
        ax_b0_pred = logits[0, len(context) + 6, :].argmax().item()

        print(f'\\n=== Final predictions ===')
        print(f'PC_B0: pred={pc_b0_pred}, draft={draft_tokens[1]}')
        print(f'AX_B0: pred={ax_b0_pred}, draft={draft_tokens[6]}')

if __name__ == "__main__":
    main()
