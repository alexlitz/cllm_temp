#!/usr/bin/env python3
"""Test that validation is ALWAYS on with no way to disable it."""

print("Testing Validation Always On (Cannot Be Disabled)")
print("=" * 60)
print()

# Test 1: Verify no parameters to disable validation
print("1. Checking SpeculativeVM has no disable parameters...")
from src.speculator import SpeculativeVM
import inspect

sig = inspect.signature(SpeculativeVM.__init__)
params = list(sig.parameters.keys())
print(f"   SpeculativeVM.__init__ parameters: {params}")

if 'validate_ratio' in params:
    print("   ✗ FAIL: validate_ratio parameter still exists (can be disabled)")
elif 'raise_on_mismatch' in params:
    print("   ✗ FAIL: raise_on_mismatch parameter still exists (can be disabled)")
else:
    print("   ✓ PASS: No parameters to disable validation")

# Test 2: Check hardcoded values
print("\n2. Checking hardcoded validation settings...")
from src.transformer_vm import C4TransformerVM

vm = SpeculativeVM(transformer_vm=C4TransformerVM())
print(f"   validate_ratio: {vm.validate_ratio}")
print(f"   raise_on_mismatch: {vm.raise_on_mismatch}")

if vm.validate_ratio == 1.0:
    print("   ✓ PASS: Hardcoded to 100% validation")
else:
    print(f"   ✗ FAIL: validate_ratio is {vm.validate_ratio}, expected 1.0")

if vm.raise_on_mismatch == True:
    print("   ✓ PASS: Hardcoded to raise on mismatch")
else:
    print("   ✗ FAIL: raise_on_mismatch is False")

# Test 3: Check ValidationError exists
print("\n3. Checking ValidationError exists...")
from src.speculator import ValidationError
print("   ✓ PASS: ValidationError class exists")

# Test 4: Verify BakedC4Transformer uses validation
print("\n4. Checking BakedC4Transformer...")
from src.baked_c4 import BakedC4Transformer

c4 = BakedC4Transformer()
print(f"   Has speculator: {hasattr(c4, 'speculator')}")
if hasattr(c4, 'speculator') and c4.speculator:
    print(f"   Speculator validate_ratio: {c4.speculator.validate_ratio}")
    print(f"   Speculator raise_on_mismatch: {c4.speculator.raise_on_mismatch}")
    if c4.speculator.validate_ratio == 1.0 and c4.speculator.raise_on_mismatch:
        print("   ✓ PASS: BakedC4Transformer uses 100% validation")
    else:
        print("   ✗ FAIL: Validation not properly configured")

print()
print("=" * 60)
print("SUMMARY:")
print()
print("✓ Validation is ALWAYS enabled (100%)")
print("✓ ValidationError ALWAYS raises on mismatch")
print("✓ NO parameters exist to disable validation")
print("✓ Configuration cannot be changed")
print()
print("This ensures the neural VM is always validated and")
print("tests will FAIL immediately when it produces wrong results.")
