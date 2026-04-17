#!/usr/bin/env python3
"""Debug ADDR_KEY layout to understand bytecode addressing."""
import sys
sys.path.insert(0, '.')
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.embedding import Opcode

def build_context(bytecode):
    context = [Token.CODE_START]
    for instr in bytecode:
        opcode = instr & 0xFF
        imm = (instr >> 8) & 0xFFFFFFFF
        context.append(opcode)
        for i in range(4):
            context.append((imm >> (i * 8)) & 0xFF)
        context.extend([0, 0, 0])  # 3 bytes padding
    context.append(Token.CODE_END)
    context.append(Token.DATA_START)
    context.append(Token.DATA_END)
    return context

def main():
    BD = _SetDim
    model = AutoregressiveVM()
    set_vm_weights(model)

    bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]
    context = build_context(bytecode)

    x = torch.tensor([context], dtype=torch.long)
    h = model.embed(x)
    model._inject_code_addr_keys(x, h)

    print('Context layout:')
    print('=' * 70)
    for i in range(len(context)):
        tok = context[i]
        if tok == Token.CODE_START:
            print(f'Position {i:2d}: CODE_START')
        elif tok == Token.CODE_END:
            print(f'Position {i:2d}: CODE_END')
            break
        elif tok < 256:
            # Check ADDR_KEY
            addr_lo = 0
            addr_hi = 0
            for k in range(16):
                if h[0, i, BD.ADDR_KEY + k] > 0.5:
                    addr_lo = k
                if h[0, i, BD.ADDR_KEY + 16 + k] > 0.5:
                    addr_hi = k
            addr = addr_lo + addr_hi * 16
            print(f'Position {i:2d}: byte {tok:3d} (0x{tok:02x}) → ADDR_KEY = {addr:2d} (0x{addr:02x})')

    print('\nInstruction layout analysis:')
    print('  Instruction 0: positions 1-8 (opcode + 4 imm + 3 padding)')
    print('  Instruction 1: positions 9-16')
    print('')
    print('ADDR_KEY assignment:')
    print('  Position 1 (opcode 0): address 0')
    print('  Position 2 (imm byte 0): address 1')
    print('  ...check if padding has ADDR_KEY...')
    print('')
    print('Expected PC increment:')
    print('  If padding has no ADDR_KEY: increment by 5 (opcode + 4 imm)')
    print('  If padding has ADDR_KEY: increment by 8 (full instruction)')

if __name__ == "__main__":
    main()
