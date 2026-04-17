"""Test first-step opcode decode for 14 opcodes (commit 63a5d78)

Tests opcodes that can execute on the first step (when HAS_SE flag is not set):
- Control: EXIT (38), NOP (39)
- Arithmetic: ADD (25), SUB (26), MUL (27), DIV (28), MOD (29)
- Bitwise: OR (14), XOR (15), AND (16)
- Comparison: EQ (17), LT (19)
- Shift: SHL (23), SHR (24)
"""

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
import sys

def test_program(code, expected_exit_code, description):
    """Test a single program."""
    print(f"\nTesting: {description}", file=sys.stderr)
    print(f"Code: {code}", file=sys.stderr)

    try:
        bytecode, data = compile_c(code)
        print(f"Bytecode: {bytecode[:10]}", file=sys.stderr)

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
        print(f"✗ ERROR: {e}", file=sys.stderr)
        return False

# Track results
passed = 0
failed = 0
tests = []

# Test 1: Simple return (EXIT + IMM)
tests.append(("int main() { return 42; }", 42, "Simple return"))

# Test 2: Arithmetic - ADD
tests.append(("""
int main() {
    int a = 10;
    int b = 20;
    return a + b;
}
""", 30, "ADD operation"))

# Test 3: Arithmetic - SUB
tests.append(("""
int main() {
    int a = 50;
    int b = 8;
    return a - b;
}
""", 42, "SUB operation"))

# Test 4: Arithmetic - MUL
tests.append(("""
int main() {
    int a = 6;
    int b = 7;
    return a * b;
}
""", 42, "MUL operation"))

# Test 5: Arithmetic - DIV
tests.append(("""
int main() {
    int a = 84;
    int b = 2;
    return a / b;
}
""", 42, "DIV operation"))

# Test 6: Arithmetic - MOD
tests.append(("""
int main() {
    int a = 142;
    int b = 100;
    return a % b;
}
""", 42, "MOD operation"))

# Test 7: Bitwise - OR
tests.append(("""
int main() {
    int a = 32;
    int b = 10;
    return a | b;
}
""", 42, "OR operation"))

# Test 8: Bitwise - XOR
tests.append(("""
int main() {
    int a = 40;
    int b = 2;
    return a ^ b;
}
""", 42, "XOR operation"))

# Test 9: Bitwise - AND
tests.append(("""
int main() {
    int a = 63;
    int b = 42;
    return a & b;
}
""", 42, "AND operation"))

# Test 10: Comparison - EQ
tests.append(("""
int main() {
    int a = 42;
    int b = 42;
    return a == b;
}
""", 1, "EQ operation (true)"))

# Test 11: Comparison - EQ (false)
tests.append(("""
int main() {
    int a = 41;
    int b = 42;
    return a == b;
}
""", 0, "EQ operation (false)"))

# Test 12: Comparison - LT
tests.append(("""
int main() {
    int a = 10;
    int b = 20;
    return a < b;
}
""", 1, "LT operation (true)"))

# Test 13: Shift - SHL
tests.append(("""
int main() {
    int a = 21;
    return a << 1;
}
""", 42, "SHL operation"))

# Test 14: Shift - SHR
tests.append(("""
int main() {
    int a = 84;
    return a >> 1;
}
""", 42, "SHR operation"))

# Test 15: NOP (if it doesn't crash, it works)
# Note: NOP is hard to test directly in C, skip for now

# Run all tests
print("=" * 60, file=sys.stderr)
print("FIRST-STEP OPCODE DECODE TEST SUITE", file=sys.stderr)
print("=" * 60, file=sys.stderr)

for code, expected, description in tests:
    if test_program(code, expected, description):
        passed += 1
    else:
        failed += 1

print("\n" + "=" * 60, file=sys.stderr)
print(f"RESULTS: {passed}/{len(tests)} tests passed", file=sys.stderr)
if failed > 0:
    print(f"FAILED: {failed} tests", file=sys.stderr)
else:
    print("✓ ALL TESTS PASSED!", file=sys.stderr)
print("=" * 60, file=sys.stderr)
