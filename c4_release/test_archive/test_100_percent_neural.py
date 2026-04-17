#!/usr/bin/env python3
"""Test 100% neural execution - JSR and LEV without handlers."""

from neural_vm.run_vm import AutoregressiveVMRunner
from src.compiler import compile_c
import sys

print("=" * 80)
print("100% NEURAL VM TEST")
print("=" * 80)

# Test 1: Simple function call
code = '''
int helper(int x) {
    return x * 2;
}

int main() {
    return helper(21);
}
'''

print("\nTest: Function call (JSR + LEV)")
print(f"Code: {code}")
print("Expected: 42")

print("\nCompiling...")
bytecode, data = compile_c(code)

print("Creating runner (with NO handlers)...")
runner = AutoregressiveVMRunner()

print(f"Handlers registered: {len(runner._func_call_handlers)}")
if len(runner._func_call_handlers) == 0:
    print("✅ Zero handlers - attempting 100% neural execution!")
else:
    print(f"❌ {len(runner._func_call_handlers)} handlers still active")
    sys.exit(1)

print("\nRunning (this will be slow on CPU, max 200 steps)...")
print("Progress: ", end="", flush=True)

# Patch to show progress
step_count = [0]
original_generate = runner.model.generate_next

def progress_generate(context, **kwargs):
    step_count[0] += 1
    if step_count[0] % 20 == 0:
        print(".", end="", flush=True)
    return original_generate(context, **kwargs)

runner.model.generate_next = progress_generate

try:
    result = runner.run(bytecode, data, max_steps=200)
    print(f"\n\nResult: {result}")
    
    if isinstance(result, tuple):
        output, exit_code = result
        print(f"  Output: '{output}'")
        print(f"  Exit code: {exit_code}")
        final_result = exit_code
    else:
        final_result = result
    
    if final_result == 42:
        print("\n" + "=" * 80)
        print("✅✅✅ SUCCESS! 100% NEURAL EXECUTION ACHIEVED! ✅✅✅")
        print("=" * 80)
        print("Function calls (JSR + LEV) work WITHOUT handlers!")
        print("The C4 Transformer VM is now FULLY NEURAL!")
        sys.exit(0)
    else:
        print(f"\n❌ FAILED - Expected 42, got {final_result}")
        print("   Neural JSR/LEV may still have issues")
        sys.exit(1)
        
except Exception as e:
    print(f"\n\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
