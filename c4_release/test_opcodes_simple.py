"""Test opcodes with simple C programs that avoid compiler limitations

Focus on direct opcode testing without local variables or arrays.
"""

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
import sys

def test_program(code, expected_exit_code, description):
    """Test a single program."""
    print(f"\n{description}", file=sys.stderr)

    try:
        bytecode, data = compile_c(code)

        runner = AutoregressiveVMRunner()
        runner.model.cuda()
        output_str, exit_code = runner.run(bytecode, max_steps=200)

        if exit_code == expected_exit_code:
            print(f"  ✓ PASS: exit_code={exit_code}", file=sys.stderr)
            if output_str:
                print(f"    Output: {repr(output_str[:50])}", file=sys.stderr)
            return True
        else:
            print(f"  ✗ FAIL: exit_code={exit_code}, expected {expected_exit_code}", file=sys.stderr)
            return False
    except Exception as e:
        error_msg = str(e)[:80]
        print(f"  ✗ ERROR: {error_msg}", file=sys.stderr)
        return False

results = {"passed": 0, "failed": 0, "total": 0}

def run_test(code, expected, desc):
    results["total"] += 1
    if test_program(code, expected, f"[{results['total']}] {desc}"):
        results["passed"] += 1
    else:
        results["failed"] += 1

print("=" * 70, file=sys.stderr)
print("SIMPLE OPCODE VERIFICATION TESTS", file=sys.stderr)
print("=" * 70, file=sys.stderr)

# ========== COMPARISON OPERATIONS ==========
print("\n--- COMPARISON OPERATIONS ---", file=sys.stderr)

# NE (not equal)
run_test("int main() { return 42 != 42; }", 0, "NE: equal values")
run_test("int main() { return 41 != 42; }", 1, "NE: different values")

# GT (greater than)
run_test("int main() { return 20 > 10; }", 1, "GT: true case")
run_test("int main() { return 10 > 20; }", 0, "GT: false case")
run_test("int main() { return 10 > 10; }", 0, "GT: equal case")

# LE (less or equal)
run_test("int main() { return 10 <= 20; }", 1, "LE: less case")
run_test("int main() { return 10 <= 10; }", 1, "LE: equal case")
run_test("int main() { return 20 <= 10; }", 0, "LE: greater case")

# GE (greater or equal)
run_test("int main() { return 20 >= 10; }", 1, "GE: greater case")
run_test("int main() { return 10 >= 10; }", 1, "GE: equal case")
run_test("int main() { return 10 >= 20; }", 0, "GE: less case")

# ========== MEMORY OPERATIONS ==========
print("\n--- MEMORY OPERATIONS ---", file=sys.stderr)

# Test PSH indirectly through expression evaluation
run_test("int main() { return 5 + 10 + 27; }", 42, "PSH: multi-value expression")

# ========== SYSTEM CALLS ==========
print("\n--- SYSTEM CALLS ---", file=sys.stderr)

# PRTF (printf)
run_test('int main() { printf("Test"); return 42; }', 42, "PRTF: simple string")
run_test('int main() { printf("Value: %d", 42); return 0; }', 0, "PRTF: with format")

# ========== I/O OPERATIONS ==========
print("\n--- I/O OPERATIONS ---", file=sys.stderr)

# PUTCHAR
run_test('int main() { putchar(65); return 0; }', 0, "PUTCHAR: output char")

# GETCHAR would need input, skip for now

# ========== CONTROL FLOW ==========
print("\n--- CONTROL FLOW ---", file=sys.stderr)

# Test branching
run_test("""
int main() {
    if (1) return 42;
    return 0;
}
""", 42, "BZ/BNZ: if statement (true)")

run_test("""
int main() {
    if (0) return 0;
    return 42;
}
""", 42, "BZ/BNZ: if statement (false)")

# ========== COMBINED OPERATIONS ==========
print("\n--- COMBINED OPERATIONS ---", file=sys.stderr)

run_test("""
int main() {
    if (10 < 20) return 42;
    return 0;
}
""", 42, "Combined: LT + BNZ")

run_test("""
int main() {
    if (20 == 20) return 42;
    return 0;
}
""", 42, "Combined: EQ + BNZ")

run_test("""
int main() {
    if (30 != 20) return 42;
    return 0;
}
""", 42, "Combined: NE + BNZ")

print("\n" + "=" * 70, file=sys.stderr)
print(f"RESULTS: {results['passed']}/{results['total']} tests passed", file=sys.stderr)
if results["failed"] > 0:
    print(f"FAILED: {results['failed']} tests", file=sys.stderr)
    print(f"SUCCESS RATE: {100*results['passed']//results['total']}%", file=sys.stderr)
else:
    print("✓ ALL TESTS PASSED!", file=sys.stderr)
print("=" * 70, file=sys.stderr)
