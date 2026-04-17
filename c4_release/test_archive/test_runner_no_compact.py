#!/usr/bin/env python3
"""Test runner without compact/compact_moe."""

import sys
import signal
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import set_vm_weights
from src.compiler import compile_c

def timeout_handler(signum, frame):
    print("\nTIMEOUT!")
    sys.exit(124)

print("Creating runner...")
runner = AutoregressiveVMRunner()

print("Setting weights (no compact)...")
set_vm_weights(runner.model)
# NO compact() or compact_moe() calls

print("Compiling program...")
source = "int main() { return 0; }"
bytecode, data = compile_c(source)
print(f"Bytecode: {bytecode}")

print("\nRunning with max_steps=3...")
sys.stdout.flush()

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(60)  # 60 second timeout

try:
    output, exit_code = runner.run(bytecode, data or b"", argv=[], max_steps=3)
    signal.alarm(0)

    print(f"Success!")
    print(f"  Output: {output!r}")
    print(f"  Exit code: {exit_code}")
    print(f"  Expected exit code: 0")
    print(f"  Match: {exit_code == 0}")

except Exception as e:
    signal.alarm(0)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
