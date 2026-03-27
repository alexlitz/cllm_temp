#!/usr/bin/env python3
"""Debug IMM opcode execution through layers."""
import sys
sys.path.insert(0, '.')
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode

def _le_bytes(val):
    v = val & 0xFFFFFFFF
    return [(v >> (i * 8)) & 0xFF for i in range(4)]

def build_context(bytecode, data=[]):
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
    for b in data:
        context.append(b)
    context.append(Token.DATA_END)

    # Initial step
    context.append(Token.REG_PC)
    context.extend(_le_bytes(2))
    context.append(Token.REG_AX)
    context.extend(_le_bytes(0))
    context.append(Token.REG_SP)
    context.extend(_le_bytes(0))
    context.append(Token.REG_BP)
    context.extend(_le_bytes(0))
    context.append(Token.STACK0)
    context.extend(_le_bytes(0))
    context.append(Token.MEM)
    context.extend(_le_bytes(0))
    context.extend(_le_bytes(0))
    context.append(Token.STEP_END)
    return context

def main():
    print('Building model...')
    model = AutoregressiveVM()
    set_vm_weights(model)
    model.eval()

    bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]
    context = build_context(bytecode)

    with torch.no_grad():
        x = torch.tensor([context], dtype=torch.long)

        # Get embeddings
        h = model.embed(x)
        print(f'Context length: {len(context)}')
        print(f'Embedding shape: {h.shape}')

        # Check key dimensions at last position (where we predict next token)
        last_pos = len(context) - 1

        # Get dimension indices from _SetDim
        from neural_vm.vm_step import _SetDim
        BD = _SetDim

        # Also check PC marker position and EMBED
        pc_marker_pos = len(context) - 35  # Position of PC marker in initial step
        print(f'\n=== PC from initial step (position {pc_marker_pos}) ===')
        print(f'PC_MARKER at {pc_marker_pos}: token={context[pc_marker_pos]}')
        print(f'PC bytes at {pc_marker_pos+1}-{pc_marker_pos+4}: {context[pc_marker_pos+1:pc_marker_pos+5]}')
        print(f'EMBED_LO at {last_pos}: {h[0, last_pos, BD.EMBED_LO:BD.EMBED_LO+4]}')

        print(f'\n=== Embeddings at position {last_pos} (predicting PC marker) ===')
        print(f'OP_IMM (dim {BD.OP_IMM}): {h[0, last_pos, BD.OP_IMM]:.4f}')
        print(f'MARK_AX (dim {BD.MARK_AX}): {h[0, last_pos, BD.MARK_AX]:.4f}')
        print(f'MARK_PC (dim {BD.MARK_PC}): {h[0, last_pos, BD.MARK_PC]:.4f}')
        print(f'FETCH_LO[0-3] (dims {BD.FETCH_LO}-{BD.FETCH_LO+3}): {h[0, last_pos, BD.FETCH_LO:BD.FETCH_LO+4]}')

        # Run through blocks
        for i, block in enumerate(model.blocks):
            h = block(h)
            if i in [2, 5]:  # Check key layers
                print(f'\n=== After Layer {i} ===')
                print(f'OP_IMM: {h[0, last_pos, BD.OP_IMM]:.4f}')
                print(f'MARK_AX: {h[0, last_pos, BD.MARK_AX]:.4f}')
                print(f'FETCH_LO[0]: {h[0, last_pos, BD.FETCH_LO]:.4f}')
                if i == 5:
                    print(f'OUTPUT_LO[0-3]: {h[0, last_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+4]}')
                    print(f'AX_CARRY_LO[0-3]: {h[0, last_pos, BD.AX_CARRY_LO:BD.AX_CARRY_LO+4]}')

        # Final logits - only predict from last position
        logits = model.head(h)
        pred_pc_marker = logits[0, last_pos, :].argmax().item()

        print(f'\n=== Predictions ===')
        print(f'Next token from pos {last_pos}: {pred_pc_marker} (expected: {Token.REG_PC}={Token.REG_PC})')

        # To see AX prediction, we'd need to run autoreggressively for 6 more tokens
        # For now, just show that PC marker prediction is correct or not

if __name__ == "__main__":
    main()
