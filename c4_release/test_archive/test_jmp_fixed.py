#!/usr/bin/env python3
"""Test JMP after fix."""

from neural_vm.run_vm import AutoregressiveVMRunner, Token
from neural_vm.embedding import Opcode
from neural_vm.constants import INSTR_WIDTH, PC_OFFSET

# JMP 4: should jump to instruction 4, skip instructions 1-3
bytecode = [
    Opcode.JMP | (4 << 8),    # [0] JMP to instruction 4
    Opcode.IMM | (99 << 8),   # [1] IMM 99 (should be skipped)
    Opcode.EXIT,              # [2] EXIT 99 (should be skipped)
    Opcode.NOP,               # [3] NOP padding
    Opcode.IMM | (42 << 8),   # [4] IMM 42 (target)
    Opcode.EXIT,              # [5] EXIT with 42
]

print("Test: JMP 4 should jump to instruction 4 (IMM 42) and exit with 42")
print()

runner = AutoregressiveVMRunner()
runner._func_call_handlers = {}

# Run step by step
ctx = runner._build_context(bytecode, b"", [])
runner._last_sp = 0x1F800
runner._last_bp = 0x10000
runner.model.set_active_opcode(bytecode[0] & 0xFF)

print("Step-by-step execution:")
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

    if pc is not None:
        instr_idx = (pc - PC_OFFSET) // INSTR_WIDTH
        if 0 <= instr_idx < len(bytecode):
            op = bytecode[instr_idx] & 0xFF
            imm = (bytecode[instr_idx] >> 8) & 0xFFFFFF
            op_names = {1: 'IMM', 2: 'JMP', 38: 'EXIT', 39: 'NOP'}
            op_name = op_names.get(op, f'OP{op}')
            if op == 1:
                op_name = f'IMM {imm}'
            runner.model.set_active_opcode(op)
        else:
            op_name = f"OUT_OF_RANGE (idx={instr_idx})"
        print(f"  Step {step}: PC=0x{pc:04x} (instr {instr_idx}: {op_name}), AX={ax}")

    if Token.HALT in step_tokens:
        print(f"\nHALT: AX={ax} (expected 42)")
        if ax == 42:
            print("SUCCESS! JMP correctly jumped to instruction 4")
        else:
            print(f"FAILURE: expected 42, got {ax}")
        break
