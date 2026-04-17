"""Test first-step opcode decode for arithmetic, bitwise, and comparison operations

Tests the 14 opcodes added in commit 63a5d78.
"""

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
import sys

def test_program(code, expected_exit_code, description):
    """Test a single program."""
    print(f"\n{description}", file=sys.stderr)
    print(f"  Code: {code.strip()}", file=sys.stderr)

    try:
        bytecode, data = compile_c(code)

        runner = AutoregressiveVMRunner()
        runner.model.cuda()
        output, exit_code = runner.run(bytecode, max_steps=100)

        if exit_code == expected_exit_code:
            print(f"  ✓ PASS: exit_code={exit_code}", file=sys.stderr)
            return True
        else:
            print(f"  ✗ FAIL: exit_code={exit_code}, expected {expected_exit_code}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"  ✗ ERROR: {str(e)[:80]}", file=sys.stderr)
        return False

# Track results
results = {"passed": 0, "failed": 0, "total": 0}

def run_test(code, expected, desc):
    results["total"] += 1
    if test_program(code, expected, f"[{results['total']}] {desc}"):
        results["passed"] += 1
    else:
        results["failed"] += 1

print("=" * 70, file=sys.stderr)
print("FIRST-STEP OPCODE DECODE TEST SUITE", file=sys.stderr)
print("Testing 14 opcodes from commit 63a5d78", file=sys.stderr)
print("=" * 70, file=sys.stderr)

# Baseline
run_test("int main() { return 42; }", 42, "Baseline - simple return")

# Arithmetic operations
run_test("int main() { return 10 + 32; }", 42, "ADD operation")
run_test("int main() { return 50 - 8; }", 42, "SUB operation")
run_test("int main() { return 6 * 7; }", 42, "MUL operation")
run_test("int main() { return 84 / 2; }", 42, "DIV operation")
run_test("int main() { return 142 % 100; }", 42, "MOD operation")

# Bitwise operations
run_test("int main() { return 32 | 10; }", 42, "OR operation")
run_test("int main() { return 40 ^ 2; }", 42, "XOR operation")
run_test("int main() { return 63 & 42; }", 42, "AND operation")

# Comparison operations
run_test("int main() { return 42 == 42; }", 1, "EQ operation (true)")
run_test("int main() { return 41 == 42; }", 0, "EQ operation (false)")
run_test("int main() { return 10 < 20; }", 1, "LT operation (true)")
run_test("int main() { return 20 < 10; }", 0, "LT operation (false)")

# Shift operations
run_test("int main() { return 21 << 1; }", 42, "SHL operation")
run_test("int main() { return 84 >> 1; }", 42, "SHR operation")

# Multiple operations
run_test("int main() { return (10 + 20) * 2 - 18; }", 42, "Multiple ops (ADD, MUL, SUB)")
run_test("int main() { return (100 / 2) - 8; }", 42, "Multiple ops (DIV, SUB)")

print("\n" + "=" * 70, file=sys.stderr)
print(f"RESULTS: {results['passed']}/{results['total']} tests passed", file=sys.stderr)
if results["failed"] > 0:
    print(f"FAILED: {results['failed']} tests", file=sys.stderr)
    sys.exit(1)
else:
    print("✓ ALL TESTS PASSED!", file=sys.stderr)
    sys.exit(0)
print("=" * 70, file=sys.stderr)
