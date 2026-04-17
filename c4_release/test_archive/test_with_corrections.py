#!/usr/bin/env python3
"""Test if neural model works better with shadow state corrections."""

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import set_vm_weights
from src.compiler import compile_c

print("Testing Neural Model WITH shadow state corrections")
print("=" * 60)

# Create runner with corrections enabled (default)
runner = AutoregressiveVMRunner(pure_attention_memory=False)
set_vm_weights(runner.model)
runner.model.compact(block_size=32)
runner.model.compact_moe()

print(f"Configuration:")
print(f"  pure_attention_memory: {runner.pure_attention_memory}")
print(f"  (False = corrections enabled)")
print()

test_cases = [
    ("int main() { return 0; }", 0),
    ("int main() { return 1; }", 1),
    ("int main() { return 42; }", 42),
    ("int main() { return 100; }", 100),
    ("int main() { return 5 + 3; }", 8),
]

matches = 0
mismatches = 0

for source, expected in test_cases:
    bytecode, data = compile_c(source)

    try:
        output, exit_code = runner.run(bytecode, data, max_steps=100)

        if exit_code == expected:
            matches += 1
            status = "✓ MATCH"
        else:
            mismatches += 1
            status = "✗ MISMATCH"

        print(f"{status}: {source[:30]:30s}")
        print(f"         Expected: {expected}, Got: {exit_code}")

    except Exception as e:
        mismatches += 1
        print(f"✗ ERROR: {source[:30]:30s}")
        print(f"         {str(e)[:60]}")

print()
print("=" * 60)
print("RESULTS:")
print(f"  Matches: {matches}/{len(test_cases)}")
print(f"  Mismatches: {mismatches}/{len(test_cases)}")

if matches + mismatches > 0:
    match_rate = (matches / (matches + mismatches)) * 100
    print(f"  Match rate: {match_rate:.1f}%")

print()
if matches > 0:
    print(f"✓ SUCCESS: Neural model matched on {matches} test(s) with corrections!")
else:
    print(f"✗ FAILURE: Neural model still broken even with corrections enabled")
