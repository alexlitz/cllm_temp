#!/usr/bin/env python3
"""Test the neural model WITHOUT DraftVM - pure autonomous execution."""

from src.baked_c4 import BakedC4Transformer

def test_pure_neural():
    """Test neural model autonomously (no DraftVM)."""

    # Build model WITHOUT speculator
    print("Building model without speculator...")
    model = BakedC4Transformer(use_speculator=False)

    # Simple test
    code = "int main() { return 100 + 200; }"
    expected = 300

    print(f"\nRunning: {code}")
    print("Expected:", expected)

    try:
        result = model.run_c(code, max_steps=10000)
        print("Result:", result)

        if result == expected:
            print("✓ TEST PASSED - Neural model computed correctly!")
            return True
        else:
            print(f"✗ TEST FAILED - Got {result} instead of {expected}")
            return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pure_neural()
    exit(0 if success else 1)
