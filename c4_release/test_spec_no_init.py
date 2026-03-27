#!/usr/bin/env python3
"""Test speculative mode WITHOUT initial step (like batch_runner)."""
import sys
sys.path.insert(0, '.')
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode

def build_context(bytecode, data=[]):
    """Build context like batch_runner does - NO initial step."""
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
    # NO INITIAL STEP - just like batch_runner
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

    print(f'\\nContext length (no initial step): {len(context)}')
    print(f'Step 0: Executing IMM 42')
    draft.step()
    print(f'DraftVM after step: PC={draft.pc}, AX={draft.ax}')
    draft_tokens = draft.draft_tokens()

    # Validate like batch_runner does
    full_context = context + draft_tokens
    with torch.no_grad():
        x = torch.tensor([full_context], dtype=torch.long)
        logits = model.forward(x)

        # Check predictions
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
        token_names = ['PC_MRK', 'PC_B0', 'PC_B1', 'PC_B2', 'PC_B3',
                      'AX_MRK', 'AX_B0', 'AX_B1', 'AX_B2', 'AX_B3',
                      'SP_MRK', 'SP_B0', 'SP_B1', 'SP_B2', 'SP_B3',
                      'BP_MRK', 'BP_B0', 'BP_B1', 'BP_B2', 'BP_B3',
                      'S0_MRK', 'S0_B0', 'S0_B1', 'S0_B2', 'S0_B3',
                      'MEM_MRK', 'MEM_A0', 'MEM_A1', 'MEM_A2', 'MEM_A3',
                      'MEM_V0', 'MEM_V1', 'MEM_V2', 'MEM_V3', 'STEP_END']
        for i, draft, pred in mismatches[:5]:
            print(f'  {token_names[i]}: draft={draft}, neural={pred}')

    if matches == 35:
        print('\\n✓ ALL TOKENS MATCH!')
        return 0
    else:
        print(f'\\n✗ {len(mismatches)} mismatches')
        return 1

if __name__ == "__main__":
    sys.exit(main())
