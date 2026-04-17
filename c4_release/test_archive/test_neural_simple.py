#!/usr/bin/env python3
"""Test neural execution with simple program and token visibility."""

from neural_vm.run_vm import AutoregressiveVMRunner
from src.compiler import compile_c
import sys

print("=" * 80)
print("SIMPLE NEURAL TEST - Token Visibility")
print("=" * 80)

# Very simple program - just return a constant
code = '''
int main() {
    return 42;
}
'''

print(f"\nCode: {code}")
print("Expected: 42")
print("\nCompiling...")
bytecode, data = compile_c(code)

print("Creating runner...")
runner = AutoregressiveVMRunner()

print(f"Handlers: {len(runner._func_call_handlers)}")

# Patch to show actual tokens
step_count = [0]
tokens_generated = []
original_generate = runner.model.generate_next

def tracking_generate(context, **kwargs):
    step_count[0] += 1
    result = original_generate(context, **kwargs)
    tokens_generated.append(result)

    if step_count[0] <= 10:
        print(f"  Step {step_count[0]}: token {result}")
    elif step_count[0] % 10 == 0:
        print(f"  Step {step_count[0]}: token {result}")

    # Stop after 100 steps to avoid infinite loops
    if step_count[0] >= 100:
        print("\n⚠️  Hit 100 step limit - stopping to prevent infinite loop")
        raise RuntimeError("Max steps reached")

    return result

runner.model.generate_next = tracking_generate

print("\nRunning (max 100 steps)...\n")

try:
    result = runner.run(bytecode, data, max_steps=100)
    print(f"\n{'=' * 80}")
    print(f"Result: {result}")

    if isinstance(result, tuple):
        output, exit_code = result
        print(f"  Output: '{output}'")
        print(f"  Exit code: {exit_code}")
        final_result = exit_code
    else:
        final_result = result

    print(f"\nTotal steps: {step_count[0]}")
    print(f"Tokens generated: {len(tokens_generated)}")

    if final_result == 42:
        print("\n✅ SUCCESS! Simple program works neurally!")
        sys.exit(0)
    else:
        print(f"\n❌ FAILED - Expected 42, got {final_result}")
        sys.exit(1)

except RuntimeError as e:
    if "Max steps" in str(e):
        print(f"\n❌ Program didn't complete in 100 steps")
        print(f"   Generated {step_count[0]} tokens")
        print(f"   Last 10 tokens: {tokens_generated[-10:]}")
        sys.exit(1)
    raise

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
