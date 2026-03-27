#!/usr/bin/env python3
"""Quick test of batched GPU validation."""

import sys
import time
from src.compiler import compile_c
from src.baked_c4 import BakedC4Transformer
from src.speculator import ValidationError

print('Testing GPU + Batched Validation')
print('=' * 60)
print()

# Create VM with batching
print('Initializing with GPU + batching (batch_size=8)...')
c4 = BakedC4Transformer(use_speculator=True)
c4.speculator.use_batching = True
c4.speculator.batch_size = 8
print()

# Test programs
test_programs = [
    ('int main() { return 0; }', 0),
    ('int main() { return 1; }', 1),
    ('int main() { return 42; }', 42),
    ('int main() { return 100; }', 100),
    ('int main() { int x; x = 5; return x; }', 5),
    ('int main() { int x; x = 10; return x + 1; }', 11),
    ('int main() { return 2 + 2; }', 4),
    ('int main() { return 10 - 3; }', 7),
]

print(f'Running {len(test_programs)} tests in batch...')
print()

# Compile all programs
bytecodes = []
data_list = []
expected = []

for code, exp in test_programs:
    bc, data = compile_c(code)
    bytecodes.append(bc)
    data_list.append(data)
    expected.append(exp)

# Run batch
print('Executing batch with validation...')
sys.stdout.flush()

start = time.time()
try:
    results = c4.speculator.run_batch(bytecodes, data_list)
    elapsed = time.time() - start

    print(f'Completed in {elapsed:.2f}s')
    print()

    # Check results
    all_correct = True
    for i, (result, exp, (code, _)) in enumerate(zip(results, expected, test_programs)):
        status = '✓' if result == exp else '✗'
        print(f'  [{i+1}] {status} {code[:40]}... → {result} (expected {exp})')
        if result != exp:
            all_correct = False

    print()
    print('=' * 60)

    if all_correct:
        print('✓ ALL TESTS PASSED!')
        print(f'  Time: {elapsed:.2f}s')
        print(f'  Speed: {len(test_programs)/elapsed:.1f} tests/second')
        print()
        print('GPU + batching is working!')
    else:
        print('✗ SOME TESTS FAILED')
        print('  Fast VM vs Neural VM mismatch')

except ValidationError as e:
    elapsed = time.time() - start
    print(f'ValidationError after {elapsed:.2f}s:')
    print(str(e))
    print()
    print('This means neural VM doesn\'t match Fast VM (expected for broken neural VM)')

except Exception as e:
    elapsed = time.time() - start
    print(f'ERROR after {elapsed:.2f}s:')
    print(f'{type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()

# Show stats
stats = c4.speculator.get_stats()
print()
print('Validation statistics:')
print(f'  Total runs: {stats["total_runs"]}')
print(f'  Validations: {stats["validations"]}')
print(f'  Mismatches: {stats["mismatches"]}')
print(f'  Match rate: {stats["match_rate"]*100:.1f}%')
