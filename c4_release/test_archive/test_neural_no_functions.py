#!/usr/bin/env python3
"""Test neural execution WITHOUT function calls (no JSR/LEV)."""

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.vm_step import Token
import sys

print("=" * 80)
print("NEURAL TEST - NO FUNCTIONS")
print("=" * 80)

# Simple linear program - no function calls
# Just: IMM 42, EXIT
bytecode = [
    Opcode.IMM | (42 << 8),  # Load 42 into AX
    Opcode.EXIT               # Exit with AX as exit code
]
data = b""

print(f"\nBytecode:")
print(f"  0: IMM 42")
print(f"  1: EXIT")
print(f"Expected exit code: 42")

print("\nCreating runner...")
runner = AutoregressiveVMRunner()
print(f"Handlers: {len(runner._func_call_handlers)}")

# Track tokens
step_count = [0]
tokens = []
original_generate = runner.model.generate_next

def tracking_generate(context, **kwargs):
    step_count[0] += 1
    result = original_generate(context, **kwargs)
    tokens.append(result)

    if step_count[0] <= 70:
        if result == Token.HALT:
            print(f"  Token {step_count[0]:3d}: {result:3d} <- HALT!")
        elif result == Token.STEP_END:
            print(f"  Token {step_count[0]:3d}: {result:3d} <- STEP_END (step {(step_count[0]-1)//35 + 1} done)")
        elif result >= 257 and result <= 268:
            names = {257: "PC", 258: "AX", 259: "SP", 260: "BP", 261: "STACK0", 262: "MEM", 263: "HALT", 264: "STEP_END"}
            print(f"  Token {step_count[0]:3d}: {result:3d} <- {names.get(result, 'MARKER')}")
        elif step_count[0] % 5 == 1:
            print(f"  Token {step_count[0]:3d}: {result:3d}")

    if step_count[0] >= 100:
        print(f"\n⚠️  Hit 100 token limit at step {step_count[0]//35 + 1}")
        raise RuntimeError("Max tokens")

    return result

runner.model.generate_next = tracking_generate

print("\nRunning (max 100 tokens)...\n")

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
        print("\n✅ SUCCESS! Neural execution works without JSR/LEV!")
        sys.exit(0)
    else:
        print(f"\n❌ FAILED - Expected 42, got {final_code}")
        sys.exit(1)

except RuntimeError as e:
    if "Max tokens" in str(e):
        print(f"\n❌ Program didn't complete in 100 tokens")
        print(f"   Tokens generated: {len(tokens)}")
        if Token.HALT in tokens:
            halt_pos = tokens.index(Token.HALT)
            print(f"   HALT was at position {halt_pos}")
        else:
            print(f"   No HALT found")
        sys.exit(1)
    raise

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
