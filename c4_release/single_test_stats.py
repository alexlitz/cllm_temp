#!/usr/bin/env python3
"""Check match rate with a single test."""

from src.speculator import SpeculativeVM, FastLogicalVM
from src.transformer_vm import C4TransformerVM
from src.compiler import compile_c

print("Single Test Match Check")
print("=" * 50)

# Create VMs
print("\nCreating transformer VM...")
transformer_vm = C4TransformerVM()

print("Creating speculator...")
speculator = SpeculativeVM(
    transformer_vm=transformer_vm,
    validate_ratio=1.0
)

# Single simple test
source = "int main() { return 42; }"
expected = 42

print(f"\nTest: {source}")
print(f"Expected: {expected}")

print("\nCompiling...")
bytecode, data = compile_c(source)
print(f"Bytecode: {bytecode[:6]}...")

print("\nRunning Fast VM...")
fast_vm = FastLogicalVM()
fast_vm.load(bytecode, data)
fast_result = fast_vm.run()
print(f"Fast VM result: {fast_result}")

print("\nRunning with validation (this may take a moment)...")
try:
    result = speculator.run(bytecode, data, validate=True, raise_on_mismatch=False)
    print(f"Speculator result: {result}")

    print("\n" + "=" * 50)
    print("STATISTICS")
    print("=" * 50)

    stats = speculator.get_stats()
    print(f"\nTotal runs: {stats['total_runs']}")
    print(f"Validations: {stats['validations']}")
    print(f"Mismatches: {stats['mismatches']}")
    print(f"Match rate: {stats['match_rate']*100:.1f}%")

    print()
    if stats['mismatches'] > 0:
        print("✗ Neural model FAILED validation")
        print("  Fast VM returned correct result")
        print("  Neural VM returned incorrect result")
    else:
        print("✓ Neural model PASSED validation")

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
