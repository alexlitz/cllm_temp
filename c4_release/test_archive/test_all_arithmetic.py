"""Test all arithmetic operations without handlers."""

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Opcode
import sys

test_cases = [
    ("ADD", "int main() { return 10 + 32; }", 42),
    ("SUB", "int main() { return 50 - 8; }", 42),
    ("MUL", "int main() { return 6 * 7; }", 42),
    ("DIV", "int main() { return 84 / 2; }", 42),
]

print("=" * 70, file=sys.stderr)
print("TESTING ARITHMETIC OPERATIONS WITHOUT HANDLERS", file=sys.stderr)
print("=" * 70, file=sys.stderr)

results = []

for op_name, code, expected in test_cases:
    bytecode, data = compile_c(code)

    runner = AutoregressiveVMRunner()

    # Remove handler for this operation
    opcode = getattr(Opcode, op_name)
    if opcode in runner._func_call_handlers:
        del runner._func_call_handlers[opcode]

    runner.model.cuda()

    output, exit_code = runner.run(bytecode, max_steps=10)

    status = "✓" if exit_code == expected else "✗"
    results.append((op_name, code, expected, exit_code, status))

    import torch
    del runner
    torch.cuda.empty_cache()

    print(f"{status} {op_name:3s}: {code:30s} => {exit_code:3d} (expected {expected})", file=sys.stderr)

print("", file=sys.stderr)
print("=" * 70, file=sys.stderr)
print("SUMMARY", file=sys.stderr)
print("=" * 70, file=sys.stderr)

working = sum(1 for _, _, _, _, s in results if s == "✓")
total = len(results)

print(f"Working: {working}/{total}", file=sys.stderr)

if working == total:
    print("✓✓✓ ALL ARITHMETIC OPERATIONS WORKING! ✓✓✓", file=sys.stderr)
elif working == 0:
    print("✗✗✗ NO ARITHMETIC OPERATIONS WORKING ✗✗✗", file=sys.stderr)
else:
    print(f"⚠ PARTIAL: {working} working, {total-working} broken", file=sys.stderr)

print("=" * 70, file=sys.stderr)
