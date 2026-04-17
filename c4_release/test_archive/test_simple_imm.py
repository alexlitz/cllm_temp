#!/usr/bin/env python3
"""Test simple IMM instruction without JMP/JSR."""

from neural_vm.run_vm import AutoregressiveVMRunner, Token
from neural_vm.embedding import Opcode

# Simple: IMM 42, EXIT
bytecode = [
    Opcode.IMM | (42 << 8),   # [0] AX = 42
    Opcode.EXIT,              # [1] EXIT with AX
]

print("Bytecode:")
for i, instr in enumerate(bytecode):
    op = instr & 0xFF
    imm = (instr >> 8) & 0xFFFFFF
    op_names = {1: 'IMM', 2: 'JMP', 3: 'JSR', 38: 'EXIT', 39: 'NOP'}
    print(f"  [{i}] {op_names.get(op, f'OP{op}')} {imm}")

runner = AutoregressiveVMRunner()
runner._func_call_handlers = {}  # 100% neural

result = runner.run(bytecode, b"", [], "", max_steps=10)
output, exit_code = result if isinstance(result, tuple) else ("", result)

print(f"\nResult: exit_code={exit_code}")
print(f"Expected: 42")
print(f"IMM works: {exit_code == 42}")
