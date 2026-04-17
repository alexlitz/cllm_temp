#!/usr/bin/env python3
"""Debug LEV execution to see why L15 heads 4-11 aren't reading memory correctly."""

from neural_vm.run_vm import AutoregressiveVMRunner
from src.compiler import compile_c
from pathlib import Path
import tempfile

# Simple function that triggers ENT (has local) and LEV (returns)
code = """
int helper(int x) {
    int local;
    local = x * 2;
    return local;
}

int main() {
    return helper(21);
}
"""

print("=" * 60)
print("LEV Debug Test - Simple Function Call")
print("=" * 60)
print(f"\nCode:\n{code}")

# Compile
with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
    f.write(code)
    f.flush()
    temp_path = f.name

bytecode, data = compile_c(Path(temp_path).read_text())
Path(temp_path).unlink()

print(f"\nBytecode: {len(bytecode)} instructions")
print(f"Data: {len(data)} bytes")

# Create runner with debug enabled
print("\nCreating runner with LEV debug enabled...")
runner = AutoregressiveVMRunner()
runner._debug_lev = True  # Enable LEV handler debug output

print("\nRunning program...")
result = runner.run(bytecode, data, [], max_steps=500)

print(f"\n{'=' * 60}")
print(f"Result: {result}")
print(f"Expected: 42")
print(f"{'=' * 60}")

if result == 42:
    print("✓ PASS: Function call/return works!")
else:
    print(f"✗ FAIL: Got {result}, expected 42")
    exit(1)
