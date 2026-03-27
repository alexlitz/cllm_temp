#!/usr/bin/env python3
"""Quick validation test with minimal steps."""

import sys
import time
from src.compiler import compile_c

print('Quick validation test (max_steps=20)...')
print('=' * 60)
print()

# Import after printing to show progress
from src.baked_c4 import BakedC4Transformer

c4 = BakedC4Transformer(use_speculator=True)

code = 'int main() { return 42; }'
expected = 42

print(f'Test: {code}')
print(f'Expected: {expected}')
print()

# Compile
print('Compiling...')
bytecode, data = compile_c(code)
print(f'Bytecode: {len(bytecode)} instructions')
print()

# Fast VM
print('Running Fast VM...')
start = time.time()
c4.speculator.fast_vm.reset()
c4.speculator.fast_vm.load(bytecode, data)
fast_result = c4.speculator.fast_vm.run()
fast_time = time.time() - start
print(f'  Result: {fast_result}')
print(f'  Time: {fast_time:.3f}s')
print()

# Neural VM with very limited steps
print('Running Neural VM (max_steps=20)...')
print('(Generating ~700 tokens, this will take 30-60 seconds)')
sys.stdout.flush()

start = time.time()
try:
    c4.transformer_vm.reset()
    c4.transformer_vm.load_bytecode(bytecode, data)
    neural_result = c4.transformer_vm.run(max_steps=20)
    neural_time = time.time() - start

    # Extract exit code
    if isinstance(neural_result, tuple):
        neural_output, neural_exit_code = neural_result
    else:
        neural_exit_code = neural_result

    print(f'  Result: {neural_exit_code}')
    print(f'  Time: {neural_time:.1f}s')
    print()

    # Compare
    print('=' * 60)
    print('COMPARISON:')
    print(f'  Fast VM:   {fast_result} (expected: {expected})')
    print(f'  Neural VM: {neural_exit_code}')
    print()

    if fast_result == neural_exit_code == expected:
        print('✓ MATCH: Both VMs agree and are correct!')
        sys.exit(0)
    elif fast_result == expected:
        print('✗ MISMATCH: Fast VM correct, Neural VM wrong')
        print(f'   This is expected - Neural VM is broken')
        sys.exit(1)
    else:
        print('✗ UNEXPECTED: Fast VM wrong')
        sys.exit(1)

except Exception as e:
    neural_time = time.time() - start
    print(f'  ERROR: {type(e).__name__} ({neural_time:.1f}s)')
    print(f'  {str(e)[:200]}')
    sys.exit(1)
