#!/usr/bin/env python3
"""Test which handlers are being called."""

from neural_vm.run_vm import AutoregressiveVMRunner
from src.compiler import compile_c

# Function call program
code = '''
int helper(int x) { return x * 2; }
int main() { return helper(21); }
'''

print("Compiling...")
bytecode, data = compile_c(code)

print("Creating runner...")
runner = AutoregressiveVMRunner()

# Track handler calls
handler_calls = []

# Monkey-patch handlers to track calls
original_jsr = runner._handler_jsr
original_lev = runner._handler_lev

def tracked_jsr(context, output):
    handler_calls.append('JSR')
    print(f"  JSR handler called (total JSR calls: {handler_calls.count('JSR')})")
    return original_jsr(context, output)

def tracked_lev(context, output):
    handler_calls.append('LEV')
    print(f"  LEV handler called (total LEV calls: {handler_calls.count('LEV')})")
    return original_lev(context, output)

runner._handler_jsr = tracked_jsr
runner._handler_lev = tracked_lev

print("\nRunning program (max 50 steps)...")
result = runner.run(bytecode, data, max_steps=50)

print(f"\nResult: {result}")
print(f"\nHandler calls: {handler_calls}")
print(f"  JSR: {handler_calls.count('JSR')} times")
print(f"  LEV: {handler_calls.count('LEV')} times")

if 'JSR' in handler_calls or 'LEV' in handler_calls:
    print("\n❌ Handlers ARE being used - not 100% neural")
else:
    print("\n✅ No handlers called - 100% neural!")
