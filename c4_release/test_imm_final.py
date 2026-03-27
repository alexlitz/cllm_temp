#!/usr/bin/env python3
"""Final test: IMM instruction with PC=idx*8+2 formula."""
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

    print(f'DraftVM after step 0:')
    print(f'  PC={draft.pc}, AX={draft.ax}')
    print(f'  Draft tokens: PC_B0={draft_tokens[1]}, AX_B0={draft_tokens[6]}')

    full_context = context + draft_tokens

    with torch.no_grad():
        x = torch.tensor([full_context], dtype=torch.long)
        h = model.embed(x)
        model._inject_code_addr_keys(x, h)
        model._inject_mem_store(x, h)

        # Run through all layers
        for i in range(len(model.blocks)):
            h = model.blocks[i](h)

        logits = model.head(h)

        # Autoregressive offset: position K predicts token K+1
        pc_marker_pos = len(context)
        ax_marker_pos = pc_marker_pos + 5

        pred_pc_b0 = logits[0, pc_marker_pos, :].argmax().item()
        pred_ax_b0 = logits[0, ax_marker_pos, :].argmax().item()

        print(f'\\nNeural VM predictions:')
        print(f'  PC_B0: pred={pred_pc_b0}, expected={draft_tokens[1]} {"✓" if pred_pc_b0 == draft_tokens[1] else "✗"}')
        print(f'  AX_B0: pred={pred_ax_b0}, expected={draft_tokens[6]} {"✓" if pred_ax_b0 == draft_tokens[6] else "✗"}')

        # Check OUTPUT at markers
        print(f'\\nOUTPUT at PC marker (position {pc_marker_pos}):')
        pc_output_lo = [f"{h[0, pc_marker_pos, BD.OUTPUT_LO + k]:.2f}" for k in range(16)]
        pc_output_hi = [f"{h[0, pc_marker_pos, BD.OUTPUT_HI + k]:.2f}" for k in range(16)]
        print(f'  OUTPUT_LO: {pc_output_lo}')
        print(f'  OUTPUT_HI: {pc_output_hi}')
        print(f'  Expected for byte 10 (0x0A): LO[10]≈4, HI[0]≈4')

        print(f'\\nOUTPUT at AX marker (position {ax_marker_pos}):')
        ax_output_lo = [f"{h[0, ax_marker_pos, BD.OUTPUT_LO + k]:.2f}" for k in range(16)]
        ax_output_hi = [f"{h[0, ax_marker_pos, BD.OUTPUT_HI + k]:.2f}" for k in range(16)]
        print(f'  OUTPUT_LO: {ax_output_lo}')
        print(f'  OUTPUT_HI: {ax_output_hi}')
        print(f'  Expected for byte 42 (0x2A): LO[10]≈4, HI[2]≈4')

if __name__ == "__main__":
    main()
