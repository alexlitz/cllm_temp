#!/usr/bin/env python3
"""
Correct ADJ test - ADJ is used to clean up arguments after function calls.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode

print("=" * 70)
print("NEURAL ADJ TEST - Function Call Argument Cleanup")
print("=" * 70)

# Test: Function with arguments (triggers ADJ for cleanup)
code = """
int add(int a, int b) {
    return a + b;
}

int main() {
    return add(10, 32);
}
"""

print("\nTest Code:")
print(code)

bytecode, data = compile_c(code)

# Verify ADJ instructions exist
adj_count = sum(1 for instr in bytecode if (instr & 0xFF) == Opcode.ADJ)
print(f"\nADJ instructions in bytecode: {adj_count}")

if adj_count == 0:
    print("✗ No ADJ instructions found - test invalid!")
    sys.exit(1)

print("✓ ADJ instructions found, proceeding with test...")

# Run the program
runner = AutoregressiveVMRunner()
print("\nRunning program...")
output, exit_code = runner.run(bytecode, data, [], max_steps=100)

print(f"\nResult:")
print(f"  Output: '{output}'")
print(f"  Exit code: {exit_code}")
print(f"  Expected: 42")

if exit_code == 42:
    print("\n✓✓✓ SUCCESS! ADJ works neurally!")
else:
    print(f"\n✗ FAIL: got {exit_code}, expected 42")
    sys.exit(1)
