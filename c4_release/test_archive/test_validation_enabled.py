#!/usr/bin/env python3
"""Test that validation is now enabled and catches neural model failures."""

from src.baked_c4 import BakedC4Transformer
from src.speculator import ValidationError

def test_validation():
    """Test that validation catches mismatches."""

    print("Creating BakedC4Transformer with validation enabled (default)")
    c4 = BakedC4Transformer(use_speculator=True, validate_neural=True)

    print(f"  use_speculator: {c4.use_speculator}")
    print(f"  validate_neural: {c4.validate_neural}")
    print(f"  speculator.validate_ratio: {c4.speculator.validate_ratio}")

    # Simple test that should fail validation
    code = "int main() { return 42; }"

    print(f"\nRunning: {code}")
    print("This should raise ValidationError because neural model is broken!\n")

    try:
        result = c4.run_c(code)
        print(f"ERROR: Test returned {result} without raising ValidationError!")
        print("Validation is NOT working correctly")
        return False
    except ValidationError as e:
        print(f"SUCCESS: ValidationError raised as expected!")
        print(f"\nError details:\n{e}")
        return True
    except Exception as e:
        print(f"ERROR: Unexpected exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_validation()
    exit(0 if success else 1)
