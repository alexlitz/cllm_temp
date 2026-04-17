#!/usr/bin/env python3
"""Debug test for pure neural execution."""

from src.baked_c4 import BakedC4Transformer

def test_pure_neural_debug():
    """Test neural model with debugging output."""

    print("Building model without speculator...")
    model = BakedC4Transformer(use_speculator=False)

    # Simpler test first
    code1 = "int main() { return 42; }"
    print(f"\nTest 1: {code1}")
    try:
        result = model.run_c(code1, max_steps=1000)
        print(f"Result: {result}")
        print(f"Expected: 42")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Original test
    code2 = "int main() { return 100 + 200; }"
    print(f"\nTest 2: {code2}")
    try:
        result = model.run_c(code2, max_steps=10000)
        print(f"Result: {result}")
        print(f"Expected: 300")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pure_neural_debug()
