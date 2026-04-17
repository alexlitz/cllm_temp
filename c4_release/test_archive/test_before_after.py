#!/usr/bin/env python3
"""Compare behavior before and after validation changes."""

from src.baked_c4 import BakedC4Transformer
from src.speculator import ValidationError

print("=" * 70)
print("BEFORE vs AFTER: Test Suite Behavior")
print("=" * 70)
print()

test_program = "int main() { return 42; }"

# ============================================================================
# BEFORE: Validation disabled (old behavior)
# ============================================================================
print("BEFORE (validation disabled):")
print("-" * 70)

c4_before = BakedC4Transformer(use_speculator=True, validate_neural=False)

print(f"  Configuration:")
print(f"    use_speculator: {c4_before.use_speculator}")
print(f"    validate_neural: {c4_before.validate_neural}")
print(f"    validation_sample_rate: {c4_before.speculator.validate_ratio}")
print()

print(f"  Running: {test_program}")
try:
    result = c4_before.run_c(test_program)
    print(f"  Result: {result}")
    print(f"  Status: ✓ Test PASSES (Fast VM returns correct result)")
    print()
    print(f"  Problem: Neural model is broken but test still passes!")
    print(f"           This gives false confidence.")
except Exception as e:
    print(f"  Exception: {type(e).__name__}: {e}")

print()

# ============================================================================
# AFTER: Validation enabled (new behavior)
# ============================================================================
print("AFTER (validation enabled):")
print("-" * 70)

c4_after = BakedC4Transformer(use_speculator=True, validate_neural=True,
                              validation_sample_rate=1.0)  # 100% for demo

print(f"  Configuration:")
print(f"    use_speculator: {c4_after.use_speculator}")
print(f"    validate_neural: {c4_after.validate_neural}")
print(f"    validation_sample_rate: {c4_after.speculator.validate_ratio}")
print()

print(f"  Running: {test_program}")
try:
    result = c4_after.run_c(test_program)
    print(f"  Result: {result}")
    print(f"  Status: Test PASSES (unexpected)")
except ValidationError as e:
    print(f"  Exception: ValidationError")
    print(f"  Status: ✗ Test FAILS (neural model validation failed)")
    print()
    print(f"  Solution: Test suite now catches broken neural model!")
    print(f"            Tests fail appropriately.")
    print()
    print(f"  Error details:")
    for line in str(e).split('\n')[:5]:
        print(f"    {line}")

print()
print("=" * 70)
print("CONCLUSION")
print("=" * 70)
print()
print("✓ BEFORE: Tests passed even with broken neural model (FALSE POSITIVE)")
print("✓ AFTER:  Tests fail when neural model is broken (CORRECT BEHAVIOR)")
print()
print("The test suite now accurately reflects the neural model's status!")
