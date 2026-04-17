#!/usr/bin/env python3
"""Debug L6 PC increment."""
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

    full_context = context + draft_tokens

    with torch.no_grad():
        x = torch.tensor([full_context], dtype=torch.long)
        h = model.embed(x)
        model._inject_code_addr_keys(x, h)
        model._inject_mem_store(x, h)

        pc_pos = len(context)

        # Run through L0-L5
        for i in range(6):
            h = model.blocks[i](h)

        print(f'Before L6 FFN at PC marker (position {pc_pos}):')
        print(f'  MARK_PC: {h[0, pc_pos, BD.MARK_PC]:.4f}')
        print(f'  HAS_SE: {h[0, pc_pos, BD.HAS_SE]:.4f} (should be 0 for step 0)')
        print(f'  OUTPUT_LO: {[f"{h[0, pc_pos, BD.OUTPUT_LO + k]:.2f}" for k in range(16)]}')
        print(f'  OUTPUT_HI: {[f"{h[0, pc_pos, BD.OUTPUT_HI + k]:.2f}" for k in range(16)]}')
        print(f'  Current value: OUTPUT_LO[2]=1 means PC=2')

        # Check if increment should fire
        # L6 increment: fires when MARK_PC=1 AND NOT HAS_SE
        # Threshold T=0.5: MARK_PC(1) - 0.5 = 0.5 > 0 → should fire
        # Suppress when HAS_SE: W_up[HAS_SE] = -2*S → adds -2*S*0 = 0 (no suppression)
        mark_pc = h[0, pc_pos, BD.MARK_PC].item()
        has_se = h[0, pc_pos, BD.HAS_SE].item()
        activation = mark_pc - 0.5 - 2.0 * has_se
        print(f'\nL6 PC increment activation check:')
        print(f'  MARK_PC={mark_pc:.4f}, HAS_SE={has_se:.4f}')
        print(f'  Activation = MARK_PC - 0.5 - 2*HAS_SE = {mark_pc:.4f} - 0.5 - 2*{has_se:.4f} = {activation:.4f}')
        print(f'  Should fire: {activation > 0} (needs > 0)')

        # Run L6 FFN
        h = model.blocks[6](h)

        print(f'\nAfter L6 FFN at PC marker:')
        print(f'  OUTPUT_LO: {[f"{h[0, pc_pos, BD.OUTPUT_LO + k]:.2f}" for k in range(16)]}')
        print(f'  OUTPUT_HI: {[f"{h[0, pc_pos, BD.OUTPUT_HI + k]:.2f}" for k in range(16)]}')
        print(f'  Expected: OUTPUT_LO[10]=1, OUTPUT_HI[0]=1 for PC=10 (2+8)')

        # Check which OUTPUT_LO values are high
        output_lo_vals = [h[0, pc_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
        max_idx = output_lo_vals.index(max(output_lo_vals))
        print(f'  Strongest OUTPUT_LO: index {max_idx} with value {output_lo_vals[max_idx]:.4f}')

if __name__ == "__main__":
    main()
