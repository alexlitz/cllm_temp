"""Test opcodes with GPU memory management

Cleans up model between tests to avoid OOM errors.
"""

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
import sys
import torch

def test_program(code, expected_exit_code, description):
    """Test a single program with GPU cleanup."""
    print(f"\n{description}", file=sys.stderr)

    try:
        bytecode, data = compile_c(code)

        runner = AutoregressiveVMRunner()
        runner.model.cuda()
        output_str, exit_code = runner.run(bytecode, max_steps=200)

        # Clean up GPU memory
        del runner
        torch.cuda.empty_cache()

        if exit_code == expected_exit_code:
            print(f"  ✓ PASS: exit_code={exit_code}", file=sys.stderr)
            return True
        else:
            print(f"  ✗ FAIL: exit_code={exit_code}, expected {expected_exit_code}", file=sys.stderr)
            return False
    except Exception as e:
        error_msg = str(e)[:80]
        print(f"  ✗ ERROR: {error_msg}", file=sys.stderr)
        # Clean up even on error
        try:
            torch.cuda.empty_cache()
        except:
            pass
        return False

results = {"passed": 0, "failed": 0, "total": 0}

def run_test(code, expected, desc):
    results["total"] += 1
    if test_program(code, expected, f"[{results['total']}] {desc}"):
        results["passed"] += 1
    else:
        results["failed"] += 1

print("=" * 70, file=sys.stderr)
print("OPCODE VERIFICATION (Memory Efficient)", file=sys.stderr)
print("=" * 70, file=sys.stderr)

# ========== COMPARISON OPERATIONS ==========
print("\n--- COMPARISON OPERATIONS ---", file=sys.stderr)

run_test("int main() { return 42 != 42; }", 0, "NE: equal")
run_test("int main() { return 41 != 42; }", 1, "NE: different")
run_test("int main() { return 20 > 10; }", 1, "GT: true")
run_test("int main() { return 10 > 20; }", 0, "GT: false")
run_test("int main() { return 10 <= 20; }", 1, "LE: true")
run_test("int main() { return 20 <= 10; }", 0, "LE: false")
run_test("int main() { return 20 >= 10; }", 1, "GE: true")
run_test("int main() { return 10 >= 20; }", 0, "GE: false")

# ========== SYSTEM CALLS ==========
print("\n--- SYSTEM CALLS ---", file=sys.stderr)

run_test('int main() { printf("Test"); return 42; }', 42, "PRTF: simple")
run_test('int main() { putchar(65); return 0; }', 0, "PUTCHAR: char")

# ========== CONTROL FLOW ==========
print("\n--- CONTROL FLOW ---", file=sys.stderr)

run_test("int main() { if (1) return 42; return 0; }", 42, "BZ: true branch")
run_test("int main() { if (0) return 0; return 42; }", 42, "BZ: false branch")

print("\n" + "=" * 70, file=sys.stderr)
print(f"RESULTS: {results['passed']}/{results['total']} tests passed", file=sys.stderr)
if results["failed"] > 0:
    print(f"FAILED: {results['failed']} tests", file=sys.stderr)
    print(f"SUCCESS RATE: {100*results['passed']//results['total']}%", file=sys.stderr)
else:
    print("✓ ALL TESTS PASSED!", file=sys.stderr)
print("=" * 70, file=sys.stderr)
