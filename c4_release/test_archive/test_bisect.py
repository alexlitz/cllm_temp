#!/usr/bin/env python3
"""Bisect test: Check if basic IMM/EXIT execution works.

Good commit: Exit code = 42
Bad commit: Exit code = 16843009 (0x01010101) or other wrong value
"""

import sys
import os

# Ensure we're testing the current checkout
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from neural_vm.run_vm import AutoregressiveVMRunner
    import inspect

    # Simple program: IMM 42, EXIT
    # Opcode 1 = IMM, immediate value 42 in upper bits
    # Opcode 34 = EXIT, immediate 0
    bytecode = [1 | (42 << 8), 34 | (0 << 8)]

    # Check if conversational_io parameter exists (backwards compatibility)
    sig = inspect.signature(AutoregressiveVMRunner.__init__)
    if 'conversational_io' in sig.parameters:
        runner = AutoregressiveVMRunner(conversational_io=False)
    else:
        runner = AutoregressiveVMRunner()

    _, exit_code = runner.run(bytecode, b'', [], max_steps=5)

    print(f"Exit code: {exit_code}")

    if exit_code == 42:
        print("✅ GOOD: Exit code correct")
        sys.exit(0)  # Good commit
    else:
        print(f"❌ BAD: Exit code should be 42, got {exit_code}")
        sys.exit(1)  # Bad commit

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)  # Treat errors as bad
