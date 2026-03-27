#!/usr/bin/env python3
"""Test speculative mode with default passthroughs."""
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
    print(f'Bytecode: {[hex(b) for b in bytecode]}')

    draft = DraftVM(bytecode)
    context = build_context(bytecode)

    print(f'\\nStep 0: Executing IMM 42')
    draft.step()
    print(f'DraftVM after step: PC={draft.pc}, AX={draft.ax}')
    draft_tokens = draft.draft_tokens()

    # SPECULATIVE MODE: Give neural VM full 35 draft tokens to validate
    full_context = context + draft_tokens
    with torch.no_grad():
        x = torch.tensor([full_context], dtype=torch.long)

        # Check intermediate activations
        from neural_vm.vm_step import _SetDim
        BD = _SetDim
        h = model.embed(x)
        model._inject_code_addr_keys(x, h)
        model._inject_mem_store(x, h)

        # Run through first few layers
        for i, block in enumerate(model.blocks):
            h = block(h)
            if i == 2:  # After L3 (register carry)
                # Check EMBED at STEP_END of initial step
                step_end_pos = len(context) - 1
                print(f'\\nAfter L2 at STEP_END pos {step_end_pos}:')
                print(f'  EMBED_LO[0-3]: {h[0, step_end_pos, BD.EMBED_LO:BD.EMBED_LO+4].tolist()}')

                # Check at AX marker in draft step
                ax_marker_pos = len(context) + 5
                print(f'\\nAfter L2 at AX marker pos {ax_marker_pos}:')
                print(f'  OP_IMM: {h[0, ax_marker_pos, BD.OP_IMM]:.4f}')
                print(f'  MARK_AX: {h[0, ax_marker_pos, BD.MARK_AX]:.4f}')
            if i == 3:  # After L4 (PC relay)
                ax_marker_pos = len(context) + 5
                print(f'\\nAfter L3 (PC relay) at AX marker pos {ax_marker_pos}:')
                print(f'  AX_CARRY_LO[0-3]: {h[0, ax_marker_pos, BD.AX_CARRY_LO:BD.AX_CARRY_LO+4].tolist()}')
            if i == 4:  # After L5 (fetch)
                ax_marker_pos = len(context) + 5
                print(f'\\nAfter L4 (fetch) at AX marker pos {ax_marker_pos}:')
                print(f'  FETCH_LO[0]: {h[0, ax_marker_pos, BD.FETCH_LO]:.4f}')
            if i == 5:  # After L6 (routing)
                ax_b0_pos = len(context) + 5 + 1
                print(f'\\nAfter L5 (routing) at AX_B0 pos {ax_b0_pos}:')
                print(f'  OUTPUT_LO[0]: {h[0, ax_b0_pos, BD.OUTPUT_LO]:.4f}')

        logits = model.head(h)

        # Check predictions for each draft token
        start = len(context)
        matches = 0
        mismatches = []

        for i in range(35):
            draft_tok = draft_tokens[i]
            pred_tok = logits[0, start + i - 1, :].argmax().item()
            if draft_tok == pred_tok:
                matches += 1
            else:
                mismatches.append((i, draft_tok, pred_tok))

    print(f'\\nResults:')
    print(f'  Matches: {matches}/35')
    print(f'  Mismatches: {len(mismatches)}/35')

    if len(mismatches) > 0:
        print(f'\\nFirst 5 mismatches:')
        for i, draft, pred in mismatches[:5]:
            token_names = ['PC_MRK', 'PC_B0', 'PC_B1', 'PC_B2', 'PC_B3',
                          'AX_MRK', 'AX_B0', 'AX_B1', 'AX_B2', 'AX_B3',
                          'SP_MRK', 'SP_B0', 'SP_B1', 'SP_B2', 'SP_B3',
                          'BP_MRK', 'BP_B0', 'BP_B1', 'BP_B2', 'BP_B3',
                          'S0_MRK', 'S0_B0', 'S0_B1', 'S0_B2', 'S0_B3',
                          'MEM_MRK', 'MEM_A0', 'MEM_A1', 'MEM_A2', 'MEM_A3',
                          'MEM_V0', 'MEM_V1', 'MEM_V2', 'MEM_V3', 'STEP_END']
            print(f'  {token_names[i]}: draft={draft}, neural={pred}')

    if matches == 35:
        print('\\n✓ ALL TOKENS MATCH! Speculative mode working correctly!')
        return 0
    else:
        print(f'\\n✗ {len(mismatches)} mismatches in speculative mode')
        return 1

if __name__ == "__main__":
    sys.exit(main())
