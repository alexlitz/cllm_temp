"""Test ADD without handler to see current behavior."""

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Opcode
import sys

code = "int main() { return 10 + 32; }"
bytecode, data = compile_c(code)

print("=" * 70, file=sys.stderr)
print("TEST: ADD WITHOUT HANDLER", file=sys.stderr)
print("=" * 70, file=sys.stderr)
print(f"Code: {code}", file=sys.stderr)
print(f"Expected result: 42 (10 + 32)", file=sys.stderr)
print("", file=sys.stderr)

# Test WITHOUT handler
runner = AutoregressiveVMRunner()

# Remove ADD handler
if Opcode.ADD in runner._func_call_handlers:
    del runner._func_call_handlers[Opcode.ADD]
    print("✓ ADD handler removed", file=sys.stderr)

runner.model.cuda()

print("Running VM without ADD handler...", file=sys.stderr)
output, exit_code = runner.run(bytecode, max_steps=10)

print("", file=sys.stderr)
print("=" * 70, file=sys.stderr)
print("RESULT", file=sys.stderr)
print("=" * 70, file=sys.stderr)
print(f"Exit code: {exit_code}", file=sys.stderr)
print(f"Expected:  42", file=sys.stderr)
print("", file=sys.stderr)

if exit_code == 42:
    print("✓✓✓ SUCCESS! ADD IS WORKING WITHOUT HANDLER! ✓✓✓", file=sys.stderr)
elif exit_code == 10:
    print("✗ FAILED: Returns first operand (10)", file=sys.stderr)
elif exit_code == 32:
    print("✗ FAILED: Returns second operand (32)", file=sys.stderr)
else:
    print(f"✗ FAILED: Unexpected result ({exit_code})", file=sys.stderr)

print("=" * 70, file=sys.stderr)

import torch
del runner
torch.cuda.empty_cache()
