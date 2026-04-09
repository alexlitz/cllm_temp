"""Check what tokens are being generated."""

import sys
import os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token

c_code = '''
int main() {
    return 42;
}
'''

print("Compiling...")
code, data = compile_c(c_code)

print("\nCreating runner...")
runner = AutoregressiveVMRunner(conversational_io=True)

tokens_generated = []
original_generate = runner.model.generate_next

def track_tokens(context):
    token = original_generate(context)
    tokens_generated.append(token)
    return token

runner.model.generate_next = track_tokens

print("\nRunning for 2 steps...")
try:
    output, exit_code = runner.run(code, data, [], max_steps=2)
    print(f"\nExit code: {exit_code}")
    print(f"Total tokens: {len(tokens_generated)}")

    # Show first 40 tokens
    print(f"\nFirst 40 tokens generated:")
    for i, t in enumerate(tokens_generated[:40]):
        # Try to get token name
        tok_name = "?"
        for attr in dir(Token):
            if not attr.startswith('_') and getattr(Token, attr) == t:
                tok_name = attr
                break
        if t < 256:
            tok_name = f"byte_{t}"

        print(f"  {i:3d}: {t:3d} {tok_name}")

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
