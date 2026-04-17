#!/usr/bin/env python3
"""Trace ADD operation step by step."""

from neural_vm.run_vm import AutoregressiveVMRunner, Token
from neural_vm.embedding import Opcode
from neural_vm.constants import INSTR_WIDTH, PC_OFFSET

# ADD should: pop from stack, add to AX
# Pattern: IMM a, PSH, IMM b, ADD -> AX = a + b
bytecode = [
    Opcode.IMM | (10 << 8),   # [0] AX = 10
    Opcode.PSH,               # [1] push AX (SP-=8, *SP=10)
    Opcode.IMM | (32 << 8),   # [2] AX = 32
    Opcode.ADD,               # [3] AX = *SP + AX = 10 + 32 = 42 (SP+=8)
    Opcode.EXIT,              # [4] EXIT
]

print("Bytecode:")
for i, instr in enumerate(bytecode):
    op = instr & 0xFF
    imm = (instr >> 8) & 0xFFFFFF
    op_names = {1: 'IMM', 13: 'PSH', 25: 'ADD', 38: 'EXIT'}
    print(f"  [{i}] {op_names.get(op, f'OP{op}')} {imm}")

runner = AutoregressiveVMRunner()
runner._func_call_handlers = {}

ctx = runner._build_context(bytecode, b"", [])
runner._last_sp = 0x10000
runner._last_bp = 0x10000
runner.model.set_active_opcode(bytecode[0] & 0xFF)

print(f"\nStep-by-step execution (expected: AX should become 42 after ADD):")
for step in range(6):
    step_tokens = []
    for _ in range(35):
        next_token = runner.model.generate_next(ctx, max_context_window=2048)
        ctx.append(next_token)
        step_tokens.append(next_token)
        if next_token == Token.HALT:
            break

    def extract_reg(tokens, marker):
        try:
            idx = tokens.index(marker)
            if idx + 4 < len(tokens):
                b = tokens[idx+1:idx+5]
                return b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)
        except ValueError:
            pass
        return None

    pc = extract_reg(step_tokens, Token.REG_PC)
    ax = extract_reg(step_tokens, Token.REG_AX)
    sp = extract_reg(step_tokens, Token.REG_SP)

    # Also extract STACK0 (value at *SP)
    stack0 = None
    try:
        idx = step_tokens.index(Token.STACK0)
        if idx + 4 < len(step_tokens):
            b = step_tokens[idx+1:idx+5]
            stack0 = b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)
    except ValueError:
        pass

    if pc is not None:
        instr_idx = (pc - PC_OFFSET) // INSTR_WIDTH
        if 0 <= instr_idx < len(bytecode):
            op = bytecode[instr_idx] & 0xFF
            op_names = {1: 'IMM', 13: 'PSH', 25: 'ADD', 38: 'EXIT'}
            op_name = op_names.get(op, f'OP{op}')
            runner.model.set_active_opcode(op)
        else:
            op_name = "OUT_OF_RANGE"
        print(f"Step {step}: PC=0x{pc:04x} ({op_name}), AX={ax}, SP=0x{sp:05x}, STACK0={stack0}")

    if Token.HALT in step_tokens:
        print(f"\nHALT: AX={ax} (expected 42)")
        break
