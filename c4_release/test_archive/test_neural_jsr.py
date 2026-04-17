#!/usr/bin/env python3
"""Test neural JSR (function call) without return."""

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.vm_step import Token
import sys

print("=" * 80)
print("NEURAL JSR TEST")
print("=" * 80)

# Test JSR without LEV:
# - JSR to function
# - Function just does: IMM 42, EXIT (no return)
#
# This tests if JSR works (push return addr, jump to target)
# WITHOUT testing LEV

bytecode = [
    # Main code at offset 0:
    Opcode.JSR | (25 << 8),  # Call function at byte offset 25 (instruction 5)
    Opcode.EXIT,              # (Never reached if function exits directly)

    # Padding (to make function start at byte 25 = 5 instructions * 5 bytes)
    Opcode.NOP,               # Offset 10 (2 instructions * 5)
    Opcode.NOP,               # Offset 15 (3 instructions * 5)
    Opcode.NOP,               # Offset 20 (4 instructions * 5)

    # Function at offset 25 (instruction 5):
    Opcode.IMM | (42 << 8),   # Load 42
    Opcode.EXIT,              # Exit (don't return)
]

data = b""

print(f"\nBytecode:")
print(f"  0: JSR 25     (call function)")
print(f"  1: EXIT       (never reached)")
print(f"  2-4: NOP      (padding)")
print(f"  5: IMM 42     (function code)")
print(f"  6: EXIT       (function exits)")
print(f"\nExpected behavior:")
print(f"  - JSR pushes return address to stack")
print(f"  - JSR jumps to offset 25 (instruction 5)")
print(f"  - Function loads 42 into AX")
print(f"  - Function exits with code 42")

print("\nCreating runner...")
runner = AutoregressiveVMRunner()
print(f"Handlers: {len(runner._func_call_handlers)}")

# Track execution
step_count = [0]
tokens = []
original_generate = runner.model.generate_next

def tracking_generate(context, **kwargs):
    step_count[0] += 1
    result = original_generate(context, **kwargs)
    tokens.append(result)

    if step_count[0] <= 150:
        if result == Token.HALT:
            print(f"  Token {step_count[0]:3d}: {result:3d} <- HALT!")
        elif result == Token.STEP_END:
            step_num = (step_count[0] - 1) // 35 + 1
            print(f"  Token {step_count[0]:3d}: {result:3d} <- STEP_END (step {step_num})")
        elif result >= 257 and result <= 268 and step_count[0] % 35 == 1:
            names = {257: "PC", 258: "AX", 259: "SP", 260: "BP", 261: "STACK0", 262: "MEM"}
            step_num = step_count[0] // 35 + 1
            print(f"  Token {step_count[0]:3d}: {result:3d} <- {names.get(result, 'MARKER')} (step {step_num} start)")

    if step_count[0] >= 200:
        print(f"\n⚠️  Hit 200 token limit (step {step_count[0]//35 + 1})")
        raise RuntimeError("Max tokens")

    return result

runner.model.generate_next = tracking_generate

print("\nRunning (max 200 tokens = ~6 steps)...\n")

try:
    result = runner.run(bytecode, data, max_steps=10)
    print(f"\n{'=' * 80}")
    print(f"RESULT: {result}")

    if isinstance(result, tuple):
        output, exit_code = result
        print(f"  Output: '{output}'")
        print(f"  Exit code: {exit_code}")
        final_code = exit_code
    else:
        final_code = result

    print(f"\nTotal tokens: {step_count[0]}")
    print(f"VM steps executed: {step_count[0] // 35}")

    if final_code == 42:
        print("\n✅ SUCCESS! Neural JSR works!")
        print("   - JSR successfully jumped to function")
        print("   - Function executed and exited with correct code")
        sys.exit(0)
    else:
        print(f"\n❌ FAILED - Expected 42, got {final_code}")
        sys.exit(1)

except RuntimeError as e:
    if "Max tokens" in str(e):
        print(f"\n❌ JSR appears stuck - program didn't complete in 200 tokens")
        print(f"   This suggests JSR neural implementation has issues")
        print(f"   Expected: ~3 steps (JSR + IMM + EXIT)")
        print(f"   Actual: {step_count[0] // 35} steps before timeout")
        sys.exit(1)
    raise

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
