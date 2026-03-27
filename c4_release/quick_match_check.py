#!/usr/bin/env python3
"""Quick check of pass and match rates."""

from src.speculator import SpeculativeVM, FastLogicalVM
from src.transformer_vm import C4TransformerVM
from src.compiler import compile_c

print("Quick Pass & Match Rate Check")
print("=" * 50)

# Create VMs
transformer_vm = C4TransformerVM()
speculator = SpeculativeVM(
    transformer_vm=transformer_vm,
    validate_ratio=1.0  # Validate all for accurate stats
)

# Simple test programs
tests = [
    ("int main() { return 0; }", 0),
    ("int main() { return 1; }", 1),
    ("int main() { return 42; }", 42),
    ("int main() { return 100; }", 100),
    ("int main() { return 5 + 3; }", 8),
    ("int main() { return 10 - 3; }", 7),
    ("int main() { return 4 * 5; }", 20),
    ("int main() { return 20 / 4; }", 5),
    ("int main() { return 17 % 5; }", 2),
    ("int main() { return 2 + 3 * 4; }", 14),
]

print(f"\nRunning {len(tests)} tests with 100% validation")
print()

passed = 0
for i, (source, expected) in enumerate(tests):
    try:
        bytecode, data = compile_c(source)
        result = speculator.run(bytecode, data, validate=True, raise_on_mismatch=False)

        if result == expected:
            passed += 1
            status = "PASS"
        else:
            status = "FAIL"

        print(f"  [{i+1:2d}] {status}: {source[:35]:35s} -> {result}")

    except Exception as e:
        print(f"  [{i+1:2d}] ERROR: {str(e)[:50]}")

print()
print("=" * 50)
print("RESULTS")
print("=" * 50)

pass_rate = (passed / len(tests)) * 100
total_validated = speculator.validations
total_mismatches = speculator.mismatches
match_count = total_validated - total_mismatches
match_rate = (match_count / total_validated * 100) if total_validated > 0 else 0

print(f"\nFast VM (Speculator):")
print(f"  Pass rate: {passed}/{len(tests)} = {pass_rate:.1f}%")

print(f"\nNeural Model:")
print(f"  Validated: {total_validated}/{len(tests)}")
print(f"  Matches: {match_count}")
print(f"  Mismatches: {total_mismatches}")
print(f"  Match rate: {match_rate:.1f}%")

print()
if match_rate == 0:
    print("⚠ Neural model: COMPLETELY BROKEN (0% match)")
elif match_rate < 50:
    print("⚠ Neural model: MOSTLY BROKEN (<50% match)")
elif match_rate < 99:
    print("⚠ Neural model: PARTIALLY WORKING")
else:
    print("✓ Neural model: WORKING CORRECTLY")
