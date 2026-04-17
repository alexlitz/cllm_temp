#!/usr/bin/env python3
"""Test if L14 addr heads fix works with fresh model load."""

from neural_vm.run_vm import AutoregressiveVMRunner
from src.compiler import compile_c
from pathlib import Path
import tempfile

code = """
int main() {
    return 42;
}
"""

print("Testing L14 addr heads fix with fresh model load")
print("=" * 70)

# Compile
with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
    f.write(code)
    f.flush()
    temp_path = f.name

bytecode, data = compile_c(Path(temp_path).read_text())
Path(temp_path).unlink()

# Create runner (forces fresh model weight initialization)
runner = AutoregressiveVMRunner()

# Monkey-patch to inspect memory writes
original_mem_store = runner._mem_store_word

step_count = 0

def debug_mem_store(addr, value):
    global step_count
    result = original_mem_store(addr, value)
    if step_count < 5:
        print(f"  [STEP {step_count}] MEM WRITE addr=0x{addr:08x}, value=0x{value:08x}")
    step_count += 1
    return result

runner._mem_store_word = debug_mem_store

print("\nRunning simple program (no function calls)...\n")
try:
    result = runner.run(bytecode, data, [], max_steps=10)
    print(f"\nResult: {result}")
    print(f"Expected: ('', 42)")

    if result == ('', 42):
        print("\n✓ Program executed correctly")
    else:
        print(f"\n✗ FAIL: Got {result}, expected ('', 42)")
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
