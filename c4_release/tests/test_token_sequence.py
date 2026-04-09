"""Show what tokens are actually generated."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token

c_code = 'int main() { return 0; }'

code, data = compile_c(c_code)
runner = AutoregressiveVMRunner(conversational_io=True)

# Move to GPU if available
import torch
if torch.cuda.is_available():
    print(f'Moving model to GPU: {torch.cuda.get_device_name(0)}')
    runner.model = runner.model.cuda()
else:
    print('Running on CPU (will be slow)')

# Build context
context = runner._build_context(code, data, [], '')
prefix_len = len(context)
print(f'Prefix length: {prefix_len}')

# Set initial opcode
if len(code) > 0:
    runner.model.set_active_opcode(code[0] & 0xFF)

# Generate just 70 tokens (2 steps worth) and show what they are
print('\nGenerating 70 tokens (should be 2 VM steps):')
for i in range(70):
    token = runner.model.generate_next(context)
    context.append(token)

    # Print token with name if special
    token_name = None
    if token == Token.STEP_END:
        token_name = 'STEP_END'
    elif token == Token.THINKING_END:
        token_name = 'THINKING_END'
    elif token == Token.THINKING_START:
        token_name = 'THINKING_START'
    elif token == Token.REG_PC:
        token_name = 'REG_PC'
    elif token == Token.REG_AX:
        token_name = 'REG_AX'
    elif token == Token.REG_SP:
        token_name = 'REG_SP'
    elif token == Token.REG_BP:
        token_name = 'REG_BP'
    elif token == Token.MEM:
        token_name = 'MEM'
    elif token == Token.STACK0:
        token_name = 'STACK0'
    elif token < 256:
        token_name = f'byte {token}'

    if token_name:
        print(f'  Token {i}: {token} ({token_name})')
    else:
        print(f'  Token {i}: {token}')

    # Check if we got a step end
    if token == Token.STEP_END:
        print(f'    ✅ STEP_END at position {i} (should be at 34, 69, etc.)')

print('\nDone')
