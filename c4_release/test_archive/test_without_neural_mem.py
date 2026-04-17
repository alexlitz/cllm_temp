#!/usr/bin/env python3
"""Test if JSR/LEV work when we disable neural MEM extraction (use handler memory only)."""

from neural_vm.run_vm import AutoregressiveVMRunner
from src.compiler import compile_c
from pathlib import Path
import tempfile

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

print("Testing JSR/LEV with handler-only memory (disable neural MEM extraction)")
print("=" * 70)

# Compile
with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
    f.write(code)
    f.flush()
    temp_path = f.name

bytecode, data = compile_c(Path(temp_path).read_text())
Path(temp_path).unlink()

# Create runner
runner = AutoregressiveVMRunner()

# Disable neural MEM extraction by making _track_memory_write a no-op
def noop_track_memory_write(context, op):
    pass

runner._track_memory_write = noop_track_memory_write

print("Running with neural MEM extraction disabled...")
print("(JSR/LEV handlers will use only handler-written memory)\n")

try:
    result = runner.run(bytecode, data, [], max_steps=100)
    print(f"\nResult: {result}")
    print(f"Expected: 42")

    if result == 42:
        print("\n✓✓✓ SUCCESS! JSR/LEV work when neural MEM is disabled!")
        print("This confirms the issue is in L14 neural MEM generation, not LEV.")
    else:
        print(f"\n✗ FAIL: Got {result}, expected 42")
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
