#!/usr/bin/env python3
"""
Test LEV neural implementation with a simple function return.
"""

import sys
sys.path.insert(0, '/home/alexlitz/Documents/misc/c4_release/c4_release')

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner

# Simple function that returns a value
code = '''
int helper() {
    return 42;
}

int main() {
    int result;
    result = helper();
    return result;
}
'''

print("Compiling code...")
try:
    bytecode, data = compile_c(code, link_stdlib=False)
    print(f"✓ Compiled: {len(bytecode)} bytes of bytecode")
except Exception as e:
    print(f"✗ Compilation failed: {e}")
    sys.exit(1)

print("\nRunning with neural VM...")
try:
    runner = AutoregressiveVMRunner(
        n_layers=17,  # Updated for LEV Phase 3
        pure_attention_memory=False,
        debug_ent_lev=True  # Enable LEV debug output
    )
    result = runner.run(bytecode, data, max_steps=1000)

    print(f"\n{'='*60}")
    print(f"Exit code: {result.exit_code}")
    print(f"Steps: {result.steps}")
    print(f"Expected: 42")

    if result.exit_code == 42:
        print("✓ LEV TEST PASSED!")
        sys.exit(0)
    else:
        print(f"✗ LEV TEST FAILED: got {result.exit_code}, expected 42")
        sys.exit(1)

except Exception as e:
    print(f"✗ Runtime error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
