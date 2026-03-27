#!/usr/bin/env python3
"""Test neural VM directly with progress output."""

import sys
import time
from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner

print('Direct Neural VM Test')
print('=' * 60)
print()

# Create runner
print('Creating AutoregressiveVMRunner...')
runner = AutoregressiveVMRunner(
    d_model=512,
    n_layers=16,
    n_heads=8,
    ffn_hidden=4096,
    max_seq_len=4096,
)

# Import and set weights
from neural_vm.vm_step import set_vm_weights
print('Setting VM weights...')
set_vm_weights(runner.model)
print('Compacting model...')
runner.model.compact(block_size=32)
runner.model.compact_moe()
print()

# Test program
code = 'int main() { return 0; }'
print(f'Test: {code}')
print('Expected: 0')
print()

# Compile
bytecode_obj, data = compile_c(code)
print(f'Bytecode: {len(bytecode_obj)} instructions')

# Convert to list format
bytecode = [instr for instr in bytecode_obj]
print(f'Instructions: {bytecode}')
print()

# Run with limited steps and progress
print('Running (max_steps=5)...')
print('Expected ~175 tokens (5 steps * 35 tokens)')
print()

start = time.time()
try:
    output, exit_code = runner.run(
        bytecode=bytecode,
        data=data,
        argv=[],
        stdin="",
        max_steps=5,  # Very limited
    )
    elapsed = time.time() - start

    print()
    print(f'Completed in {elapsed:.1f}s')
    print(f'Output: {repr(output)}')
    print(f'Exit code: {exit_code}')
    print()

    if exit_code == 0:
        print('✓ CORRECT: Neural VM returned 0')
    else:
        print(f'✗ WRONG: Neural VM returned {exit_code}, expected 0')

except KeyboardInterrupt:
    elapsed = time.time() - start
    print(f'\n\nInterrupted after {elapsed:.1f}s')
    sys.exit(1)
except Exception as e:
    elapsed = time.time() - start
    print()
    print(f'ERROR after {elapsed:.1f}s:')
    print(f'  {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
