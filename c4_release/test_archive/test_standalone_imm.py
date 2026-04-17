#!/usr/bin/env python3
"""Test standalone neural VM with IMM opcode."""
import sys
sys.path.insert(0, '.')
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode

def _le_bytes(val):
    """32-bit value -> 4 little-endian byte tokens."""
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

    # Add initial step (step -1) with PC=2, all registers = 0
    # This bootstraps the neural VM with the starting state
    context.append(Token.REG_PC)
    context.extend(_le_bytes(2))  # Initial PC = 2 (start of first instruction)
    context.append(Token.REG_AX)
    context.extend(_le_bytes(0))  # AX = 0
    context.append(Token.REG_SP)
    context.extend(_le_bytes(0))  # SP = 0
    context.append(Token.REG_BP)
    context.extend(_le_bytes(0))  # BP = 0
    context.append(Token.STACK0)
    context.extend(_le_bytes(0))  # STACK0 = 0
    context.append(Token.MEM)
    context.extend(_le_bytes(0))  # MEM addr = 0
    context.extend(_le_bytes(0))  # MEM val = 0
    context.append(Token.STEP_END)

    return context

def main():
    print('Building model...')
    model = AutoregressiveVM()
    set_vm_weights(model)
    model.eval()

    bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]
    print(f'Bytecode: {[hex(b) for b in bytecode]}')
    print(f'IMM value should be: 42 = 0x{42:02x}')

    draft = DraftVM(bytecode)
    context = build_context(bytecode)

    print('\nTesting IMM opcode (step 0)...')
    draft.step()
    print(f'DraftVM AX after step: {draft.ax}')
    draft_tokens = draft.draft_tokens()

    # Run neural prediction
    full_context = context + draft_tokens
    with torch.no_grad():
        x = torch.tensor([full_context], dtype=torch.long)
        logits = model.forward(x)
        start = len(context) - 1
        neural_tokens = logits[0, start:start+35, :].argmax(dim=-1).tolist()

    # Compare key positions
    TOKEN_NAMES = [
        'PC_MARKER', 'PC_B0', 'PC_B1', 'PC_B2', 'PC_B3',
        'AX_MARKER', 'AX_B0', 'AX_B1', 'AX_B2', 'AX_B3',
        'SP_MARKER', 'SP_B0', 'SP_B1', 'SP_B2', 'SP_B3',
        'BP_MARKER', 'BP_B0', 'BP_B1', 'BP_B2', 'BP_B3',
        'STACK0_MARKER', 'STACK0_B0', 'STACK0_B1', 'STACK0_B2', 'STACK0_B3',
        'MEM_MARKER', 'MEM_ADDR_B0', 'MEM_ADDR_B1', 'MEM_ADDR_B2', 'MEM_ADDR_B3',
        'MEM_VAL_B0', 'MEM_VAL_B1', 'MEM_VAL_B2', 'MEM_VAL_B3',
        'STEP_END'
    ]

    print(f'\nDraft tokens for AX register:')
    print(f'  AX_MARKER: {draft_tokens[5]}')
    print(f'  AX_B0: {draft_tokens[6]}')
    print(f'  AX_B1: {draft_tokens[7]}')
    print(f'  AX_B2: {draft_tokens[8]}')
    print(f'  AX_B3: {draft_tokens[9]}')
    print(f'\nNeural predictions for AX register:')
    print(f'  AX_MARKER: {neural_tokens[5]}')
    print(f'  AX_B0: {neural_tokens[6]}')
    print(f'  AX_B1: {neural_tokens[7]}')
    print(f'  AX_B2: {neural_tokens[8]}')
    print(f'  AX_B3: {neural_tokens[9]}')
    print()

    mismatches = []
    for i in range(35):
        if draft_tokens[i] != neural_tokens[i]:
            mismatches.append((i, TOKEN_NAMES[i], draft_tokens[i], neural_tokens[i]))

    if len(mismatches) == 0:
        print('✓ ALL TOKENS MATCH! Neural VM is working standalone!')
        return 0
    else:
        print(f'✗ {len(mismatches)} mismatches:')
        for i, name, draft_val, neural_val in mismatches[:10]:
            print(f'  {name}: draft={draft_val}, neural={neural_val}')
        return 1

if __name__ == "__main__":
    sys.exit(main())
