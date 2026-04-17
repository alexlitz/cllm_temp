"""Test first-step opcode decode with simpler programs

Focus on testing opcodes directly without complex C constructs.
"""

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
import sys

def test_program(code, expected_exit_code, description):
    """Test a single program."""
    print(f"\nTesting: {description}", file=sys.stderr)
    print(f"Code: {code.strip()}", file=sys.stderr)

    try:
        bytecode, data = compile_c(code)
        print(f"Compiled: {len(bytecode)} instructions", file=sys.stderr)

        runner = AutoregressiveVMRunner()
        runner.model.cuda()
        output, exit_code = runner.run(bytecode, max_steps=100)

        if exit_code == expected_exit_code:
            print(f"✓ PASS: exit_code={exit_code}", file=sys.stderr)
            return True
        else:
            print(f"✗ FAIL: exit_code={exit_code}, expected {expected_exit_code}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"✗ ERROR: {str(e)[:100]}", file=sys.stderr)
        return False

# Track results
passed = 0
failed = 0

print("=" * 60, file=sys.stderr)
print("FIRST-STEP OPCODE DECODE TEST SUITE (SIMPLE)", file=sys.stderr)
print("=" * 60, file=sys.stderr)

# Test 1: Simple return (baseline)
if test_program("int main() { return 42; }", 42, "Baseline - simple return"):
    passed += 1
else:
    failed += 1

# Test 2: Return 0 (EXIT opcode)
if test_program("int main() { return 0; }", 0, "EXIT with 0"):
    passed += 1
else:
    failed += 1

# Test 3: Return different value
if test_program("int main() { return 100; }", 100, "IMM with 100"):
    passed += 1
else:
    failed += 1

# Test 4: Empty main (should return 0)
if test_program("int main() { }", 0, "Empty main"):
    passed += 1
else:
    failed += 1

print("\n" + "=" * 60, file=sys.stderr)
print(f"RESULTS: {passed}/4 tests passed", file=sys.stderr)
if failed > 0:
    print(f"FAILED: {failed} tests", file=sys.stderr)
else:
    print("✓ ALL TESTS PASSED!", file=sys.stderr)
print("=" * 60, file=sys.stderr)
