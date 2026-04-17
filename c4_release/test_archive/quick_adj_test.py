#!/usr/bin/env python3
"""Quick inline ADJ test."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode

# Test: Function with arguments (triggers ADJ)
code = """
int add(int a, int b) {
    return a + b;
}
int main() {
    return add(10, 32);
}
"""

print("Compiling...")
bytecode, data = compile_c(code)

# Check for ADJ
adj_count = sum(1 for instr in bytecode if (instr & 0xFF) == Opcode.ADJ)
print(f"ADJ instructions: {adj_count}")

if adj_count == 0:
    print("✗ No ADJ - test invalid")
    sys.exit(1)

print("Running (max 200 steps)...")
runner = AutoregressiveVMRunner()
result = runner.run(bytecode, data, [], max_steps=200)
print(f"Result: {result}")
print(f"Expected: ('', 42) or 42")

# Check if it's a tuple or just exit code
if isinstance(result, tuple):
    output, exit_code = result
    if exit_code == 42:
        print("✓ SUCCESS!")
    else:
        print(f"✗ FAIL: exit code {exit_code}")
elif result == 42:
    print("✓ SUCCESS!")
else:
    print(f"✗ FAIL: result {result}")
