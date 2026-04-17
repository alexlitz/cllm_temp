#!/usr/bin/env python3
"""Trace JSR execution step by step."""

from neural_vm.run_vm import AutoregressiveVMRunner, Token
from neural_vm.vm_step import _SetDim as BD
from src.compiler import compile_c
import torch

# Simple function call
code = '''
int add(int a, int b) {
    return a + b;
}

int main() {
    return add(10, 32);
}
'''

bytecode, data = compile_c(code)
print(f"Bytecode: {len(bytecode)} instructions")

runner = AutoregressiveVMRunner()
runner._func_call_handlers = {}

# Build initial context
ctx = runner._build_context(bytecode, data, [])
prefix_len = len(ctx)
print(f"Initial context: {prefix_len} tokens")

# Run a few steps manually
runner._last_sp = 0x1F800
runner._last_bp = 0x10000

# Set initial opcode
from neural_vm.constants import INSTR_WIDTH, PC_OFFSET
init_exec = runner._exec_pc() // INSTR_WIDTH
print(f"Initial PC: exec_idx={init_exec}")
if 0 <= init_exec < len(bytecode):
    opcode = bytecode[init_exec] & 0xFF
    print(f"Initial opcode: {opcode}")
    runner.model.set_active_opcode(opcode)

# Generate steps
max_steps = 20
for step in range(max_steps):
    # Generate 35 tokens for this step
    step_tokens = []
    for _ in range(Token.STEP_TOKENS):
        next_token = runner.model.generate_next(ctx)
        ctx.append(next_token)
        step_tokens.append(next_token)

        if next_token == Token.HALT:
            break

    # Decode step
    pc_bytes = []
    ax_bytes = []
    sp_bytes = []
    bp_bytes = []

    i = 0
    while i < len(step_tokens):
        tok = step_tokens[i]
        if tok == Token.REG_PC:
            pc_bytes = step_tokens[i+1:i+5]
            i += 5
        elif tok == Token.REG_AX:
            ax_bytes = step_tokens[i+1:i+5]
            i += 5
        elif tok == Token.REG_SP:
            sp_bytes = step_tokens[i+1:i+5]
            i += 5
        elif tok == Token.REG_BP:
            bp_bytes = step_tokens[i+1:i+5]
            i += 5
        elif tok == Token.STACK0:
            i += 5
        elif tok == Token.MEM:
            i += 9
        elif tok in [Token.STEP_END, Token.HALT]:
            break
        else:
            i += 1

    def bytes_to_int(b):
        if len(b) < 4:
            return 0
        return b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)

    pc = bytes_to_int(pc_bytes)
    ax = bytes_to_int(ax_bytes)
    sp = bytes_to_int(sp_bytes)
    bp = bytes_to_int(bp_bytes)

    # Get opcode at current PC
    pc_idx = pc // INSTR_WIDTH - PC_OFFSET // INSTR_WIDTH
    instr = bytecode[pc_idx] if 0 <= pc_idx < len(bytecode) else 0
    op = instr & 0xFF
    imm = (instr >> 8) & 0xFFFFFF

    op_names = {1: 'LEA', 2: 'IMM', 3: 'JMP', 4: 'JSR', 5: 'BZ', 6: 'BNZ', 7: 'ENT', 8: 'ADJ', 9: 'LEV',
                10: 'LI', 11: 'LC', 12: 'SI', 13: 'SC', 14: 'PSH', 15: 'OR', 16: 'XOR', 17: 'AND',
                18: 'EQ', 19: 'NE', 20: 'LT', 21: 'GT', 22: 'LE', 23: 'GE', 24: 'SHL', 25: 'SHR',
                26: 'ADD', 27: 'SUB', 28: 'MUL', 29: 'DIV', 30: 'MOD', 31: 'EXIT'}
    op_name = op_names.get(op, f'OP{op}')

    print(f"Step {step}: PC=0x{pc:08x} ({op_name} {imm}), AX={ax}, SP=0x{sp:08x}, BP=0x{bp:08x}")

    # Set next opcode
    runner.model.set_active_opcode(op)

    if step_tokens[-1] == Token.HALT or (op == 31 and step > 0):  # EXIT
        print(f"\nHALT/EXIT: AX = {ax}")
        break
