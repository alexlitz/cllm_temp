#!/usr/bin/env python3
"""Debug neural VM to see where it hangs."""

import sys
import time
from src.compiler import compile_c

print('Debugging Neural VM Token Generation')
print('=' * 60)
print()

# Patch the model to add progress logging
print('Setting up neural VM with debug logging...')

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import set_vm_weights, Token

# Create runner
runner = AutoregressiveVMRunner(
    d_model=512,
    n_layers=16,
    n_heads=8,
    ffn_hidden=4096,
    max_seq_len=4096,
)

# Patch generate_next to add logging
original_generate_next = runner.model.generate_next
token_count = [0]
last_print = [time.time()]

def logged_generate_next(context):
    token_count[0] += 1
    now = time.time()

    # Print progress every second
    if now - last_print[0] >= 1.0:
        print(f'  Tokens generated: {token_count[0]}', flush=True)
        last_print[0] = now

    result = original_generate_next(context)

    # Print special tokens
    if result == Token.STEP_END:
        print(f'  → STEP_END at token {token_count[0]}', flush=True)
    elif result == Token.HALT:
        print(f'  → HALT at token {token_count[0]}', flush=True)

    return result

runner.model.generate_next = logged_generate_next

print('Setting VM weights...')
set_vm_weights(runner.model)

print('Compacting model...')
sys.stdout.flush()
compact_start = time.time()
runner.model.compact(block_size=32)
compact_time = time.time() - compact_start
print(f'  compact() took {compact_time:.1f}s')

sys.stdout.flush()
moe_start = time.time()
runner.model.compact_moe()
moe_time = time.time() - moe_start
print(f'  compact_moe() took {moe_time:.1f}s')
print()

# Test simple program
code = 'int main() { return 0; }'
bytecode_obj, data = compile_c(code)
bytecode = [instr for instr in bytecode_obj]

print(f'Test: {code}')
print(f'Bytecode: {bytecode}')
print(f'Expected steps: 4-5 (IMM→LEV→ENT→LEV→EXIT)')
print(f'Expected tokens: ~140-175 (35 per step)')
print()

print('Starting execution (max_steps=3)...')
print('Progress will be shown every second:')
print()
sys.stdout.flush()

start = time.time()
try:
    output, exit_code = runner.run(
        bytecode=bytecode,
        data=data,
        argv=[],
        stdin="",
        max_steps=3,
    )
    elapsed = time.time() - start

    print()
    print('=' * 60)
    print(f'COMPLETED in {elapsed:.1f}s')
    print(f'Total tokens: {token_count[0]}')
    print(f'Output: {repr(output)}')
    print(f'Exit code: {exit_code}')
    print(f'Tokens/second: {token_count[0]/elapsed:.1f}')

    if exit_code == 0:
        print('✓ CORRECT RESULT')
    else:
        print(f'✗ WRONG RESULT (expected 0, got {exit_code})')

except KeyboardInterrupt:
    elapsed = time.time() - start
    print()
    print(f'\nInterrupted after {elapsed:.1f}s')
    print(f'Tokens generated: {token_count[0]}')
    print(f'Rate: {token_count[0]/elapsed:.1f} tokens/second')
    sys.exit(1)

except Exception as e:
    elapsed = time.time() - start
    print()
    print(f'ERROR after {elapsed:.1f}s:')
    print(f'Tokens generated: {token_count[0]}')
    print(f'{type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
