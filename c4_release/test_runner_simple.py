#!/usr/bin/env python3
"""Test runner.run() with minimal steps."""

import sys
import signal
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import set_vm_weights
from src.compiler import compile_c

def timeout_handler(signum, frame):
    print("\nTIMEOUT!")
    sys.exit(124)

print("Creating runner...")
sys.stdout.flush()
runner = AutoregressiveVMRunner()

print("Setting weights...")
sys.stdout.flush()
set_vm_weights(runner.model)

print("Compiling program...")
sys.stdout.flush()
source = "int main() { return 0; }"
bytecode, data = compile_c(source)
print(f"Bytecode: {bytecode}")

print("\nRunning with max_steps=2...")
sys.stdout.flush()

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)  # 30 second timeout

try:
    output, exit_code = runner.run(bytecode, data or b"", argv=[], max_steps=2)
    signal.alarm(0)

    print(f"Success!")
    print(f"  Output: {output!r}")
    print(f"  Exit code: {exit_code}")

except Exception as e:
    signal.alarm(0)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
