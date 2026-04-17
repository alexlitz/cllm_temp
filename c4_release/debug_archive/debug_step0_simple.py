#!/usr/bin/env python3
"""Simple debug of step 0 PC behavior."""
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
    model = AutoregressiveVM()
    set_vm_weights(model)
    model.eval()

    bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]
    context = build_context(bytecode)

    draft = DraftVM(bytecode)
    draft.step()
    draft_tokens = draft.draft_tokens()

    print(f'DraftVM: PC={draft.pc}, AX={draft.ax}')
    print(f'Draft tokens: PC_B0={draft_tokens[1]}, AX_B0={draft_tokens[6]}')

    print(f'\\nContext structure:')
    print(f'  Position 0: {context[0]} (CODE_START)')
    for i in range(1, min(10, len(context))):
        print(f'  Position {i}: {context[i]} (address {i})')
    print(f'  ...')
    print(f'  Context length: {len(context)}')

    full_context = context + draft_tokens

    with torch.no_grad():
        x = torch.tensor([full_context], dtype=torch.long)
        h = model.embed(x)
        model._inject_code_addr_keys(x, h)
        model._inject_mem_store(x, h)

        # Check positions and ADDR_KEY
        pc_pos = len(context)
        ax_pos = pc_pos + 5

        print(f'\\nADDR_KEY in code section:')
        for i in range(1, 6):
            addr_key = h[0, i, BD.ADDR_KEY:BD.ADDR_KEY+4].tolist()
            print(f'  Position {i} (token {x[0,i].item()}): ADDR_KEY={addr_key}')

        # L0-L2
        for i in range(3):
            h = model.blocks[i](h)
        print(f'\\nAfter L2:')
        print(f'  PC marker pos {pc_pos}, HAS_SE: {h[0, pc_pos, BD.HAS_SE]:.4f}')
        print(f'  AX marker pos {ax_pos}, MARK_AX: {h[0, ax_pos, BD.MARK_AX]:.4f}')

        # L3
        h = model.blocks[3](h)
        print(f'\\nAfter L3 FFN (PC default/increment):')
        print(f'  OUTPUT_LO at PC marker: {[f"{h[0, pc_pos, BD.OUTPUT_LO + k]:.2f}" for k in range(8)]}...')
        print(f'  Expected: OUTPUT_LO[2]=1 for PC=2')

        # L4
        h = model.blocks[4](h)
        print(f'\\nAfter L4 (PC-2 and PC-1 for fetch):')
        print(f'  EMBED_LO at AX (all 16): {[f"{h[0, ax_pos, BD.EMBED_LO + k]:.2f}" for k in range(16)]}')
        print(f'  Expected EMBED (PC-2=0): LO[0]=1')
        print(f'  TEMP at AX (all 16): {[f"{h[0, ax_pos, BD.TEMP + k]:.2f}" for k in range(16)]}')
        print(f'  Expected TEMP (PC-1=1): TEMP[1]=1')

        # L5
        h = model.blocks[5](h)
        print(f'\\nAfter L5 (fetch+decode):')
        print(f'  OPCODE_BYTE_LO at AX: {h[0, ax_pos, BD.OPCODE_BYTE_LO:BD.OPCODE_BYTE_LO+4].tolist()}')
        print(f'  FETCH_LO at AX (all 16): {[f"{h[0, ax_pos, BD.FETCH_LO + k]:.2f}" for k in range(16)]}')
        print(f'  FETCH_HI at AX (all 16): {[f"{h[0, ax_pos, BD.FETCH_HI + k]:.2f}" for k in range(16)]}')
        print(f'  Expected FETCH for byte 42 (0x2A): LO[10]=1, HI[2]=1')
        print(f'  OP_IMM at AX: {h[0, ax_pos, BD.OP_IMM]:.4f} (should be ~5.0)')

        # L6
        h = model.blocks[6](h)
        ax_b0_pos = ax_pos + 1  # AX byte 0 position
        print(f'\\nAfter L6 FFN:')
        print(f'  PC OUTPUT_LO: {[f"{h[0, pc_pos, BD.OUTPUT_LO + k]:.2f}" for k in range(16)]}')
        print(f'  AX marker OUTPUT_LO (all 16): {[f"{h[0, ax_pos, BD.OUTPUT_LO + k]:.2f}" for k in range(16)]}')
        print(f'  AX marker OUTPUT_HI (all 16): {[f"{h[0, ax_pos, BD.OUTPUT_HI + k]:.2f}" for k in range(16)]}')
        print(f'  AX byte0 OUTPUT_LO (first 4): {[f"{h[0, ax_b0_pos, BD.OUTPUT_LO + k]:.2f}" for k in range(4)]}')
        print(f'  Expected at marker for byte 42 (0x2A): OUTPUT_LO[10]=1, OUTPUT_HI[2]=1')

        # Run rest, checking PC after key layers
        for i in range(7, len(model.blocks)):
            h = model.blocks[i](h)
            if i in [10, 14]:
                print(f'\\nAfter L{i}:')
                print(f'  OUTPUT_LO (all 16): {[f"{h[0, pc_pos, BD.OUTPUT_LO + k]:.2f}" for k in range(16)]}')

        print(f'\\nAfter L15 (final layer):')
        print(f'  OUTPUT_LO (all 16): {[f"{h[0, pc_pos, BD.OUTPUT_LO + k]:.2f}" for k in range(16)]}')
        print(f'  OUTPUT_HI (all 16): {[f"{h[0, pc_pos, BD.OUTPUT_HI + k]:.2f}" for k in range(16)]}')
        print(f'  Expected: OUTPUT_LO[7]=1, OUTPUT_HI[0]=1 for byte 7')

        logits = model.head(h)

        # Autoregressive offset: position K predicts token K+1
        # So to predict AX_B0, we read from AX marker position
        pred_pc_b0 = logits[0, pc_pos, :].argmax().item()
        pred_ax_b0 = logits[0, ax_pos, :].argmax().item()  # Read from marker, not byte position!

        print(f'\\nPredictions (with autoregressive offset):')
        print(f'  PC_B0: pred={pred_pc_b0}, draft={draft_tokens[1]}')
        print(f'  AX_B0: pred={pred_ax_b0}, draft={draft_tokens[6]}')

        # Check logits at AX marker
        ax_logits = logits[0, ax_pos, :256]
        print(f'\\nLogits at AX marker (predicts AX_B0), top 10:')
        top_logits, top_indices = ax_logits.topk(10)
        for i in range(10):
            print(f'  Token {top_indices[i].item()}: {top_logits[i].item():.4f}')

if __name__ == "__main__":
    main()
