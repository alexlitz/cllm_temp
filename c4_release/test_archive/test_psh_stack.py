#!/usr/bin/env python3
"""Test PSH operation and stack."""

from neural_vm.run_vm import AutoregressiveVMRunner, Token
from neural_vm.embedding import Opcode
from neural_vm.constants import INSTR_WIDTH, PC_OFFSET

# Test: IMM 42, PSH, IMM 0, LI (load from stack), EXIT
# This should: push 42, set AX=0, load 42 from stack into AX, exit with 42
bytecode = [
    Opcode.IMM | (42 << 8),   # [0] AX = 42
    Opcode.PSH,               # [1] push AX (SP -= 8, *SP = AX)
    Opcode.IMM | (0 << 8),    # [2] AX = 0
    Opcode.LI,                # [3] AX = *(int*)*SP = load 42 from stack
    Opcode.EXIT,              # [4] EXIT with AX
]

print("Bytecode:")
for i, instr in enumerate(bytecode):
    op = instr & 0xFF
    imm = (instr >> 8) & 0xFFFFFF
    op_names = {1: 'IMM', 10: 'LI', 14: 'PSH', 38: 'EXIT'}
    print(f"  [{i}] {op_names.get(op, f'OP{op}')} {imm}")

runner = AutoregressiveVMRunner()
runner._func_call_handlers = {}

# Trace execution
ctx = runner._build_context(bytecode, b"", [])
runner._last_sp = 0x1F800
runner._last_bp = 0x10000
runner.model.set_active_opcode(bytecode[0] & 0xFF)

print(f"\nStep-by-step execution:")
for step in range(6):
    step_tokens = []
    for _ in range(35):
        next_token = runner.model.generate_next(ctx, max_context_window=2048)
        ctx.append(next_token)
        step_tokens.append(next_token)
        if next_token == Token.HALT:
            break

    # Extract registers
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
    bp = extract_reg(step_tokens, Token.REG_BP)

    if pc is not None:
        instr_idx = (pc - PC_OFFSET) // INSTR_WIDTH
        if 0 <= instr_idx < len(bytecode):
            op = bytecode[instr_idx] & 0xFF
            op_names = {1: 'IMM', 10: 'LI', 14: 'PSH', 38: 'EXIT', 39: 'NOP'}
            op_name = op_names.get(op, f'OP{op}')
            runner.model.set_active_opcode(op)
        else:
            op_name = "OUT_OF_RANGE"
        print(f"Step {step}: PC=0x{pc:04x} ({op_name}), AX={ax}, SP=0x{sp:05x}, BP=0x{bp:05x}")

    if Token.HALT in step_tokens:
        print(f"\nHALT: AX={ax}")
        break

print(f"\n--- Full runner execution ---")
result = runner.run(bytecode, b"", [], "", max_steps=10)
output, exit_code = result if isinstance(result, tuple) else ("", result)
print(f"Result: exit_code={exit_code}, expected=42")
