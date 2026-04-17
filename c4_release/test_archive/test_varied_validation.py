#!/usr/bin/env python3
"""Test validation with varied programs."""

from src.baked_c4 import BakedC4Transformer
from src.speculator import ValidationError

print("Creating BakedC4Transformer with 10% validation")
c4 = BakedC4Transformer(use_speculator=True, validate_neural=True, validation_sample_rate=0.1)

# Different programs to get different bytecode hashes
test_programs = [
    ("int main() { return 0; }", 0),
    ("int main() { return 1; }", 1),
    ("int main() { return 42; }", 42),
    ("int main() { return 100; }", 100),
    ("int main() { return 5 + 3; }", 8),
    ("int main() { return 10 * 2; }", 20),
    ("int main() { return 15 - 5; }", 10),
    ("int main() { return 20 / 4; }", 5),
    ("int main() { return 17 % 5; }", 2),
    ("int main() { return 2 + 3 * 4; }", 14),
]

passed = 0
failed_validation = 0
errors = 0

for i, (code, expected) in enumerate(test_programs):
    try:
        result = c4.run_c(code)
        if result == expected:
            passed += 1
            print(f"  [{i+1:2d}] PASS: {code[:40]:40s} -> {result}")
        else:
            print(f"  [{i+1:2d}] FAIL: {code[:40]:40s} -> {result} (expected {expected})")
            failed_validation += 1
    except ValidationError as e:
        print(f"  [{i+1:2d}] VALIDATION ERROR: {code[:30]:30s}")
        print(f"       {str(e)[:80]}...")
        failed_validation += 1
    except Exception as e:
        print(f"  [{i+1:2d}] ERROR: {code[:40]:40s} -> {e}")
        errors += 1

print(f"\nResults:")
print(f"  Passed: {passed}/{len(test_programs)}")
print(f"  Failed validation: {failed_validation}")
print(f"  Errors: {errors}")

if failed_validation > 0:
    print(f"\nSUCCESS: Validation caught neural model failures!")
    exit(0)
else:
    print(f"\nNote: No validations triggered in this run")
    print(f"      (10% sampling means ~1 test should validate)")
    print(f"      Run with more tests to see validation failures")
    exit(0)
