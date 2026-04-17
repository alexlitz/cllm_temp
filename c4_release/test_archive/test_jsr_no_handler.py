#!/usr/bin/env python3
"""Test if JSR works without the handler."""

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from src.compiler import compile_c

# Simple JSR test
code = '''
int helper(int x) {
    return x * 2;
}

int main() {
    return helper(21);
}
'''

print("=" * 80)
print("JSR Test Without Handler")
print("=" * 80)
print(f"\nCode: {code}")
print("Expected: 42\n")

print("Compiling...")
bytecode, data = compile_c(code)

print("Creating runner...")
runner = AutoregressiveVMRunner()

# REMOVE JSR handler temporarily
print("\n⚠️  Removing JSR handler to test neural path...")
if Opcode.JSR in runner._func_call_handlers:
    del runner._func_call_handlers[Opcode.JSR]
    print("✓ JSR handler removed")
else:
    print("✗ JSR handler not found!")

print("\nRunning (max 100 steps, will be slow)...")
try:
    result = runner.run(bytecode, data, max_steps=100)
    print(f"\nResult: {result}")
    
    if result == 42:
        print("\n✅ SUCCESS! JSR works neurally!")
    else:
        print(f"\n❌ FAILED - got {result} instead of 42")
        print("   Neural JSR path may be broken")
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
