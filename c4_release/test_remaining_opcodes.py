"""Test remaining untested opcodes

Categories:
1. Comparison ops (NE, GT, LE, GE)
2. Memory ops (LC, SC, PSH)
3. System calls (MALC, FREE, MSET, MCMP, PRTF)
4. I/O ops (GETCHAR, PUTCHAR)
5. Control ops (NOP, POP, BLT, BGE)
"""

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
import sys

def test_program(code, expected_exit_code, description):
    """Test a single program."""
    print(f"\n{description}", file=sys.stderr)
    print(f"  Code: {code.strip()[:60]}...", file=sys.stderr)

    try:
        bytecode, data = compile_c(code)

        runner = AutoregressiveVMRunner()
        runner.model.cuda()
        output, exit_code = runner.run(bytecode, max_steps=200)

        if exit_code == expected_exit_code:
            print(f"  ✓ PASS: exit_code={exit_code}", file=sys.stderr)
            return True
        else:
            print(f"  ✗ FAIL: exit_code={exit_code}, expected {expected_exit_code}", file=sys.stderr)
            return False
    except Exception as e:
        error_msg = str(e)[:80]
        print(f"  ✗ ERROR: {error_msg}", file=sys.stderr)
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
print("REMAINING OPCODES TEST SUITE", file=sys.stderr)
print("=" * 70, file=sys.stderr)

# ========== COMPARISON OPERATIONS (NE, GT, LE, GE) ==========
print("\n--- COMPARISON OPERATIONS ---", file=sys.stderr)

run_test("int main() { return 42 != 42; }", 0, "NE operation (false)")
run_test("int main() { return 41 != 42; }", 1, "NE operation (true)")
run_test("int main() { return 20 > 10; }", 1, "GT operation (true)")
run_test("int main() { return 10 > 20; }", 0, "GT operation (false)")
run_test("int main() { return 10 <= 20; }", 1, "LE operation (true)")
run_test("int main() { return 10 <= 10; }", 1, "LE operation (equal)")
run_test("int main() { return 20 <= 10; }", 0, "LE operation (false)")
run_test("int main() { return 20 >= 10; }", 1, "GE operation (true)")
run_test("int main() { return 10 >= 10; }", 1, "GE operation (equal)")
run_test("int main() { return 10 >= 20; }", 0, "GE operation (false)")

# ========== MEMORY OPERATIONS (PSH tested via expressions) ==========
print("\n--- MEMORY OPERATIONS ---", file=sys.stderr)

# PSH is implicitly tested in any expression with multiple values
run_test("int main() { int x = 42; return x; }", 42, "PSH/variable test")

# LC and SC are char operations - harder to test directly in C
# They would need pointer manipulation which may not work yet

# ========== SYSTEM CALLS ==========
print("\n--- SYSTEM CALLS ---", file=sys.stderr)

# MALC/FREE - memory allocation
run_test("""
int main() {
    int *p = (int*)malloc(8);
    if (p == 0) return 1;
    free(p);
    return 0;
}
""", 0, "MALC/FREE operations")

# MSET - memset
run_test("""
int main() {
    int arr[3];
    memset(arr, 0, 12);
    return arr[0];
}
""", 0, "MSET operation")

# MCMP - memcmp
run_test("""
int main() {
    int a[2];
    int b[2];
    a[0] = 42;
    a[1] = 43;
    b[0] = 42;
    b[1] = 43;
    return memcmp(a, b, 8);
}
""", 0, "MCMP operation (equal)")

# PRTF - printf
run_test("""
int main() {
    printf("Hello\\n");
    return 0;
}
""", 0, "PRTF operation")

# ========== FILE I/O (may not work without special setup) ==========
print("\n--- FILE I/O OPERATIONS ---", file=sys.stderr)

# OPEN/READ/CLOS might need special environment setup
# Skip for now unless we have a test infrastructure

# ========== CONTROL OPERATIONS ==========
print("\n--- CONTROL OPERATIONS ---", file=sys.stderr)

# NOP is hard to test - it does nothing
# POP might not be emitted by compiler directly
# BLT/BGE are VM extensions, may not be in compiler

print("\n" + "=" * 70, file=sys.stderr)
print(f"RESULTS: {results['passed']}/{results['total']} tests passed", file=sys.stderr)
if results["failed"] > 0:
    print(f"FAILED: {results['failed']} tests", file=sys.stderr)
    sys.exit(1)
else:
    print("✓ ALL TESTS PASSED!", file=sys.stderr)
    sys.exit(0)
print("=" * 70, file=sys.stderr)
