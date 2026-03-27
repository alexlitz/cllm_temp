#!/usr/bin/env python3
"""Test if run_c hangs."""

import sys
print("Creating BakedC4Transformer without validation...")
sys.stdout.flush()

from src.baked_c4 import BakedC4Transformer

c4 = BakedC4Transformer(use_speculator=True, validate_neural=False)
print("Created")
sys.stdout.flush()

print("Running: int main() { return 0; }")
sys.stdout.flush()

result = c4.run_c("int main() { return 0; }")
print(f"Result: {result}")
sys.stdout.flush()

print("SUCCESS without validation")
print()

print("Now creating with validation enabled...")
sys.stdout.flush()

c4_val = BakedC4Transformer(use_speculator=True, validate_neural=True, validation_sample_rate=1.0)
print("Created")
sys.stdout.flush()

print("Running: int main() { return 0; } with validation...")
sys.stdout.flush()

result = c4_val.run_c("int main() { return 0; }")
print(f"Result: {result}")
sys.stdout.flush()

print("SUCCESS with validation")
