#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

from src.compiler import compile_c
from neural_vm.batch_runner_v2 import UltraBatchRunner

# Test programs matching C4 syntax
programs = [
    ("int main() { return 42; }", 42),
    ("int main() { int x; x = 10; return x + 32; }", 42),
    ("int main() { int x; int y; x = 6; y = 7; return x * y; }", 42),
    ("int main() { int x; x = 100; x = x - 58; return x; }", 42),
    ("int main() { int x; x = 20; return x + 22; }", 42),
]

print("Testing multiple programs with batch runner...")
print("=" * 60)

bytecodes = []
expected_results = []
for prog, exp in programs:
    bc, _ = compile_c(prog)
    bytecodes.append(bc)
    expected_results.append(exp)

runner = UltraBatchRunner(batch_size=8, strict=False)
results = runner.run_batch(bytecodes, max_steps=200)

print("\nResults:")
all_pass = True
for i, (res, exp, (prog, _)) in enumerate(zip(results, expected_results, programs)):
    status = "PASS" if res == exp else "FAIL"
    if res != exp:
        all_pass = False
    prog_short = prog[:50] + "..." if len(prog) > 50 else prog
    print(f"  {i+1}. [{status}] Result: {res:3d} (expected {exp:3d})")
    print(f"      {prog_short}")

print("\n" + "=" * 60)
if all_pass:
    print("ALL TESTS PASSED!")
    print("Batch runner executes multiple different programs correctly.")
else:
    print("Some tests failed")
