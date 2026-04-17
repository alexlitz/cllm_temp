#!/usr/bin/env python3
"""Test simple JMP with small target to verify basic functionality."""

from neural_vm.run_vm import AutoregressiveVMRunner, Token
from neural_vm.embedding import Opcode

# JMP to instruction 4 (PC = 4*8+2 = 34 = 0x22)
# At instruction 4: IMM 42, EXIT
bytecode = [
    Opcode.JMP | (4 << 8),    # [0] JMP to instruction 4
    Opcode.IMM | (99 << 8),   # [1] IMM 99 (should be skipped)
    Opcode.EXIT,              # [2] EXIT 99 (should be skipped)
    Opcode.NOP,               # [3] NOP padding
    Opcode.IMM | (42 << 8),   # [4] IMM 42 (target)
    Opcode.EXIT,              # [5] EXIT with 42
]

print("Bytecode:")
for i, instr in enumerate(bytecode):
    op = instr & 0xFF
    imm = (instr >> 8) & 0xFFFFFF
    # From embedding.py: IMM=1, JMP=2, JSR=3, etc.
    op_names = {1: 'IMM', 2: 'JMP', 3: 'JSR', 4: 'BZ', 5: 'BNZ', 6: 'ENT', 7: 'ADJ', 8: 'LEV',
                9: 'LEA', 10: 'LI', 11: 'LC', 12: 'SI', 13: 'SC', 14: 'PSH', 38: 'EXIT', 39: 'NOP'}
    print(f"  [{i}] {op_names.get(op, f'OP{op}')} {imm}")

print(f"\nJMP target: instruction 4, PC = 4*8+2 = 34 = 0x22")
print(f"JMP immediate bytes: [{4 & 0xFF}, {(4 >> 8) & 0xFF}, {(4 >> 16) & 0xFF}, 0]")
print(f"Expected PC bytes: [0x22, 0x00, 0x00, 0x00]")

runner = AutoregressiveVMRunner()
runner._func_call_handlers = {}  # 100% neural

# Build context and trace execution
from neural_vm.constants import INSTR_WIDTH, PC_OFFSET
ctx = runner._build_context(bytecode, b"", [])
prefix_len = len(ctx)
print(f"\nContext length: {prefix_len} tokens")

runner._last_sp = 0x1F800
runner._last_bp = 0x10000
runner.model.set_active_opcode(bytecode[0] & 0xFF)

# Generate a few steps and trace PC
for step in range(6):
    # Generate 35 tokens
    step_tokens = []
    for _ in range(35):
        next_token = runner.model.generate_next(ctx, max_context_window=2048)
        ctx.append(next_token)
        step_tokens.append(next_token)
        if next_token == Token.HALT:
            break

    # Extract PC
    pc_idx = step_tokens.index(Token.REG_PC) if Token.REG_PC in step_tokens else -1
    if pc_idx >= 0 and pc_idx + 4 < len(step_tokens):
        pc_bytes = step_tokens[pc_idx+1:pc_idx+5]
        pc = pc_bytes[0] | (pc_bytes[1] << 8) | (pc_bytes[2] << 16) | (pc_bytes[3] << 24)
        instr_idx = (pc - PC_OFFSET) // INSTR_WIDTH
        if 0 <= instr_idx < len(bytecode):
            op = bytecode[instr_idx] & 0xFF
            op_names = {1: 'IMM', 2: 'JMP', 3: 'JSR', 4: 'BZ', 5: 'BNZ', 6: 'ENT', 7: 'ADJ', 8: 'LEV',
                        9: 'LEA', 10: 'LI', 11: 'LC', 12: 'SI', 13: 'SC', 14: 'PSH', 38: 'EXIT', 39: 'NOP'}
            op_name = op_names.get(op, f'OP{op}')
        else:
            op_name = "OUT_OF_RANGE"
        print(f"Step {step}: PC=0x{pc:04x} ({pc}), instr_idx={instr_idx}, op={op_name}, pc_bytes={pc_bytes}")

        # Set next opcode
        if 0 <= instr_idx < len(bytecode):
            runner.model.set_active_opcode(bytecode[instr_idx] & 0xFF)

    if Token.HALT in step_tokens:
        ax_idx = step_tokens.index(Token.REG_AX) if Token.REG_AX in step_tokens else -1
        if ax_idx >= 0 and ax_idx + 4 < len(step_tokens):
            ax_bytes = step_tokens[ax_idx+1:ax_idx+5]
            ax = ax_bytes[0] | (ax_bytes[1] << 8) | (ax_bytes[2] << 16) | (ax_bytes[3] << 24)
            print(f"\nHALT: AX={ax}")
        break

print("\n--- Full runner execution ---")
result = runner.run(bytecode, b"", [], "", max_steps=20)
output, exit_code = result if isinstance(result, tuple) else ("", result)

print(f"\nResult: exit_code={exit_code}")
print(f"Expected: 42 (if JMP works), 99 (if JMP fails)")
print(f"JMP works: {exit_code == 42}")
