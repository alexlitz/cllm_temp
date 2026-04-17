#!/usr/bin/env python3
"""Test various opcodes with 100% neural execution (no handlers)."""

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode

def test_bytecode(name, bytecode, expected):
    runner = AutoregressiveVMRunner()
    runner._func_call_handlers = {}  # 100% neural
    result = runner.run(bytecode, b"", [], "", max_steps=50)
    exit_code = result[1] if isinstance(result, tuple) else result
    success = exit_code == expected
    print(f"{name}: exit_code={exit_code}, expected={expected}, {'PASS' if success else 'FAIL'}")
    return success

results = []

# Test 1: IMM
results.append(test_bytecode("IMM 42", [
    Opcode.IMM | (42 << 8),
    Opcode.EXIT,
], 42))

# Test 2: IMM + ADD
results.append(test_bytecode("10 + 32 = 42", [
    Opcode.IMM | (10 << 8),   # AX = 10
    Opcode.PSH,               # push AX
    Opcode.IMM | (32 << 8),   # AX = 32
    Opcode.ADD,               # AX = *SP + AX = 10 + 32 = 42
    Opcode.EXIT,
], 42))

# Test 3: IMM + SUB
results.append(test_bytecode("50 - 8 = 42", [
    Opcode.IMM | (50 << 8),   # AX = 50
    Opcode.PSH,               # push AX
    Opcode.IMM | (8 << 8),    # AX = 8
    Opcode.SUB,               # AX = *SP - AX = 50 - 8 = 42
    Opcode.EXIT,
], 42))

# Test 4: IMM + MUL
results.append(test_bytecode("6 * 7 = 42", [
    Opcode.IMM | (6 << 8),    # AX = 6
    Opcode.PSH,               # push AX
    Opcode.IMM | (7 << 8),    # AX = 7
    Opcode.MUL,               # AX = *SP * AX = 6 * 7 = 42
    Opcode.EXIT,
], 42))

# Test 5: Sequential IMM (last value wins)
results.append(test_bytecode("IMM sequence", [
    Opcode.IMM | (10 << 8),   # AX = 10
    Opcode.IMM | (20 << 8),   # AX = 20
    Opcode.IMM | (42 << 8),   # AX = 42
    Opcode.EXIT,
], 42))

# Test 6: Zero
results.append(test_bytecode("IMM 0", [
    Opcode.IMM | (0 << 8),
    Opcode.EXIT,
], 0))

# Test 7: Large value
results.append(test_bytecode("IMM 255", [
    Opcode.IMM | (255 << 8),
    Opcode.EXIT,
], 255))

print(f"\n{'='*40}")
print(f"Results: {sum(results)}/{len(results)} tests passed")
