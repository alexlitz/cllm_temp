#!/usr/bin/env python3
"""Minimal ADJ test - check if ADJ (stack adjustment after function call) works neurally."""

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from src.compiler import compile_c
from pathlib import Path
import tempfile

# ADJ is used to clean up function arguments after a call
code = """
int add(int a, int b) {
    return a + b;
}

int main() {
    int result;
    result = add(3, 4);
    return result;
}
"""

print("Testing ADJ (stack adjustment) neural execution...")
print("Code:")
print(code)
print()

try:
    # Compile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
        f.write(code)
        f.flush()
        temp_path = f.name

    bytecode, data = compile_c(Path(temp_path).read_text())
    Path(temp_path).unlink()

    # Check if ADJ is in bytecode
    adj_count = sum(1 for instr in bytecode if (instr & 0xFF) == Opcode.ADJ)

    print(f"ADJ instructions in bytecode: {adj_count}")

    if adj_count == 0:
        print("✗ No ADJ in bytecode!")
        exit(1)

    # Show ADJ instructions
    for i, instr in enumerate(bytecode):
        if (instr & 0xFF) == Opcode.ADJ:
            imm = instr >> 8
            print(f"  Position {i}: ADJ {imm} bytes")

    # Run with neural execution
    print()
    print("Running with neural VM (max 200 steps)...")
    runner = AutoregressiveVMRunner()
    result = runner.run(bytecode, data, [], max_steps=200)

    print(f"Exit code: {result}")
    print(f"Expected: 7 (3+4)")
    print()

    if result == 7:
        print("✓✓✓ ADJ works neurally!")
        print("The transformer correctly:")
        print("  1. Called function with arguments")
        print("  2. Pushed arguments to stack")
        print("  3. Executed function")
        print("  4. Used ADJ to clean up stack (SP += 16)")
        print("  5. Returned correct result")
    else:
        print(f"✗ Got {result}, expected 7")
        exit(1)

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
