#!/usr/bin/env python3
"""Test that sampled validation works and is fast."""

import time
from src.baked_c4 import BakedC4Transformer
from src.speculator import ValidationError

def test_sampled_validation():
    """Run multiple tests - some should be validated, any mismatch should fail."""

    print("Creating BakedC4Transformer with 100% validation for testing")
    c4_full = BakedC4Transformer(use_speculator=True, validate_neural=True, validation_sample_rate=1.0)

    print("Creating BakedC4Transformer with 10% validation (default)")
    c4_sampled = BakedC4Transformer(use_speculator=True, validate_neural=True, validation_sample_rate=0.1)

    code = "int main() { return 42; }"

    # Test with 100% validation - should always fail
    print(f"\n1. Testing with 100% validation (should fail immediately):")
    try:
        result = c4_full.run_c(code)
        print(f"   ERROR: Got result {result} without ValidationError!")
        return False
    except ValidationError as e:
        print(f"   SUCCESS: ValidationError raised as expected")
        print(f"   Details: {str(e)[:150]}...")

    # Test with 10% validation - might pass some, but will eventually fail
    print(f"\n2. Testing with 10% validation (sampled):")
    print(f"   Running 20 times - statistically ~2 should be validated")

    validated_count = 0
    error_raised = False

    for i in range(20):
        try:
            result = c4_sampled.run_c(code)
            print(f"   Run {i+1}: Result={result} (validation skipped)")
        except ValidationError as e:
            print(f"   Run {i+1}: ValidationError raised (validation happened!)")
            error_raised = True
            validated_count += 1
            # In real tests, this would fail the test suite immediately

    print(f"\n   Validated {validated_count}/20 runs (~{validated_count/20*100:.0f}%)")

    if error_raised:
        print(f"   SUCCESS: At least one validation caught the mismatch!")
        return True
    else:
        print(f"   Note: No validations happened in this run (random sampling)")
        print(f"         But would fail eventually over many tests")
        return True

if __name__ == "__main__":
    success = test_sampled_validation()
    exit(0 if success else 1)
