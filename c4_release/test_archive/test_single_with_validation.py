#!/usr/bin/env python3
"""Test single program with validation."""

from src.baked_c4 import BakedC4Transformer
from src.speculator import ValidationError

print("Creating BakedC4Transformer with validation...")
c4 = BakedC4Transformer(use_speculator=True, validate_neural=True)

print("Running test: int main() { return 42; }")

try:
    result = c4.run_c("int main() { return 42; }")
    print(f"Result: {result}")
    print("ERROR: Should have raised ValidationError!")
except ValidationError as e:
    print(f"SUCCESS: Validation caught the mismatch!")
    print(f"Details: {str(e)[:200]}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
