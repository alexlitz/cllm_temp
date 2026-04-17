#!/usr/bin/env python3
"""Debug JSR execution after ALiBi fix."""

from neural_vm.run_vm import AutoregressiveVMRunner, Token
from neural_vm.vm_step import _SetDim as BD
from neural_vm.embedding import Opcode
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
print(f"Bytecode length: {len(bytecode)} instructions")

# Find JSR instruction
for i, instr in enumerate(bytecode[:20]):
    op = instr & 0xFF
    imm = (instr >> 8) & 0xFFFFFF
    # Map opcode number to name
    op_names = {1: 'LEA', 2: 'IMM', 3: 'JMP', 4: 'JSR', 5: 'BZ', 6: 'BNZ', 7: 'ENT', 8: 'ADJ', 9: 'LEV',
                10: 'LI', 11: 'LC', 12: 'SI', 13: 'SC', 14: 'PSH', 15: 'OR', 16: 'XOR', 17: 'AND',
                18: 'EQ', 19: 'NE', 20: 'LT', 21: 'GT', 22: 'LE', 23: 'GE', 24: 'SHL', 25: 'SHR',
                26: 'ADD', 27: 'SUB', 28: 'MUL', 29: 'DIV', 30: 'MOD', 31: 'EXIT'}
    op_name = op_names.get(op, f'OP{op}')
    print(f"  [{i}] 0x{instr:08x}: {op_name}, imm=0x{imm:06x} ({imm})")

runner = AutoregressiveVMRunner()
runner._func_call_handlers = {}  # Disable all handlers
runner._debug_ent = False
runner._debug_lev = False

# Run with step limit and trace
result = runner.run(bytecode, data, [], "", max_steps=100)
output, exit_code = result if isinstance(result, tuple) else ("", result)

print(f"\nResult: exit_code={exit_code}, output='{output}'")
print(f"Expected: 42")
