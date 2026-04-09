"""
Quick test of arithmetic operations without handlers.
Tests a few simple programs to verify the fix works.
"""

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Opcode
import torch

print("="*80)
print("QUICK ARITHMETIC TEST (without handlers)")
print("="*80 + "\n")

test_cases = [
    ("return 42;", 42, "Literal"),
    ("return 10 + 32;", 42, "ADD"),
    ("return 50 - 8;", 42, "SUB"),
]

results = []

for code, expected, name in test_cases:
    full_code = f"int main() {{ {code} }}"
    print(f"{name:12s}: ", end="", flush=True)

    try:
        bytecode, data = compile_c(full_code)
        runner = AutoregressiveVMRunner()

        # Remove arithmetic handlers
        for op in [Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV]:
            if op in runner._func_call_handlers:
                del runner._func_call_handlers[op]

        output, exit_code = runner.run(bytecode, max_steps=20)

        if output == expected and exit_code == 0:
            print(f"✓ PASS (got {output})")
            results.append(True)
        else:
            print(f"✗ FAIL (expected {expected}, got {output}, exit {exit_code})")
            results.append(False)

    except Exception as e:
        print(f"✗ ERROR: {e}")
        results.append(False)

print("\n" + "="*80)
if all(results):
    print(f"✓✓✓ ALL {len(results)} TESTS PASSED! ✓✓✓")
else:
    passed = sum(results)
    print(f"⚠️  {passed}/{len(results)} tests passed")
print("="*80)

torch.cuda.empty_cache()
