#!/usr/bin/env python3
"""Test function calls with optimized model."""

import sys
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights
from neural_vm.run_vm import AutoregressiveVMRunner
from src.compiler import compile_c

print("=" * 80)
print("Function Call Test (Optimized)")
print("=" * 80)

# Simple function call
code = '''
int helper(int x) {
    return x * 2;
}

int main() {
    return helper(21);
}
'''

print(f"\nCode: {code}")
print("Expected result: 42\n")

print("1. Compiling...")
bytecode, data = compile_c(code)
print(f"   ✓ {len(bytecode)} bytes")

print("2. Creating and configuring model...")
model = AutoregressiveVM(n_layers=17)
set_vm_weights(model)
print("   ✓ Weights set")

print("3. Compacting model (removes unused FFN units)...")
model.compact(block_size=1)
print("   ✓ Model compacted")

print("4. Creating runner with optimized model...")
runner = AutoregressiveVMRunner()
runner.model = model
print("   ✓ Runner ready")

print("\n5. Running program (this may take a few minutes on CPU)...")
print("   Progress: ", end="", flush=True)

# Monkey-patch to show progress
original_generate = model.generate_next
step_count = [0]

def progress_generate(context, **kwargs):
    step_count[0] += 1
    if step_count[0] % 10 == 0:
        print(".", end="", flush=True)
    return original_generate(context, **kwargs)

model.generate_next = progress_generate

try:
    result = runner.run(bytecode, data, max_steps=500)
    print(f"\n\n6. Result: {result}")
    print(f"   Expected: 42")
    
    if result == 42:
        print("\n✅ SUCCESS! Function calls work!")
        sys.exit(0)
    else:
        print(f"\n❌ FAILED - Got {result} instead of 42")
        
        # Check if handlers were used
        print("\nChecking which handlers were called...")
        # The result might be a tuple (output, exit_code)
        if isinstance(result, tuple):
            print(f"   Output: {result[0]}")
            print(f"   Exit code: {result[1]}")
        sys.exit(1)
except KeyboardInterrupt:
    print("\n\n⚠️  Interrupted by user")
    sys.exit(2)
except Exception as e:
    print(f"\n\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
