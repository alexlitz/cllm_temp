#!/usr/bin/env python3
"""Test if just creating BakedC4Transformer hangs."""

import sys
print("Starting test...")
sys.stdout.flush()

print("Importing BakedC4Transformer...")
sys.stdout.flush()
from src.baked_c4 import BakedC4Transformer

print("Creating BakedC4Transformer...")
sys.stdout.flush()
c4 = BakedC4Transformer(use_speculator=True, validate_neural=False)

print("SUCCESS: BakedC4Transformer created")
sys.stdout.flush()

print("Now testing with validation enabled...")
sys.stdout.flush()
c4_val = BakedC4Transformer(use_speculator=True, validate_neural=True, validation_sample_rate=0.1)

print("SUCCESS: BakedC4Transformer with validation created")
sys.stdout.flush()
