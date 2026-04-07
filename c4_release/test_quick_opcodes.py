"""Quick opcode verification - tests only the most important untested opcodes"""

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
import sys
import torch

def test(code, expected, desc):
    """Single test with cleanup."""
    print(f"{desc}...", end=" ", file=sys.stderr)
    try:
        bytecode, data = compile_c(code)
        runner = AutoregressiveVMRunner()
        runner.model.cuda()
        _, exit_code = runner.run(bytecode, max_steps=100)
        del runner
        torch.cuda.empty_cache()

        if exit_code == expected:
            print(f"✓ PASS ({exit_code})", file=sys.stderr)
            return True
        else:
            print(f"✗ FAIL ({exit_code}, expected {expected})", file=sys.stderr)
            return False
    except Exception as e:
        print(f"✗ ERROR: {str(e)[:50]}", file=sys.stderr)
        return False

print("QUICK OPCODE TESTS\n" + "="*50, file=sys.stderr)

passed = failed = 0

# NE (not equal)
if test("int main() { return 42 != 42; }", 0, "NE-false"): passed += 1
else: failed += 1
if test("int main() { return 41 != 42; }", 1, "NE-true"): passed += 1
else: failed += 1

# GT (greater than)
if test("int main() { return 20 > 10; }", 1, "GT-true"): passed += 1
else: failed += 1
if test("int main() { return 10 > 10; }", 0, "GT-false"): passed += 1
else: failed += 1

# LE (less or equal)
if test("int main() { return 10 <= 20; }", 1, "LE-true"): passed += 1
else: failed += 1
if test("int main() { return 20 <= 10; }", 0, "LE-false"): passed += 1
else: failed += 1

# GE (greater or equal)
if test("int main() { return 20 >= 10; }", 1, "GE-true"): passed += 1
else: failed += 1
if test("int main() { return 10 >= 10; }", 1, "GE-equal"): passed += 1
else: failed += 1

# PRTF
if test('int main() { printf("Hi"); return 42; }', 42, "PRTF"): passed += 1
else: failed += 1

# PUTCHAR
if test('int main() { putchar(65); return 0; }', 0, "PUTCHAR"): passed += 1
else: failed += 1

print(f"\n{'='*50}", file=sys.stderr)
print(f"RESULTS: {passed}/{passed+failed} passed ({100*passed//(passed+failed)}%)", file=sys.stderr)
