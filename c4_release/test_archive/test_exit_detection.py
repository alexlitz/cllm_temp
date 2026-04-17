#!/usr/bin/env python3
"""Test EXIT opcode detection and HALT token generation."""

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.vm_step import Token
from src.compiler import compile_c
import sys

print("=" * 80)
print("EXIT DETECTION TEST")
print("=" * 80)

# Simplest possible program - just EXIT
bytecode = [Opcode.EXIT]
data = b""

print(f"\nBytecode: [EXIT]")
print(f"Expected behavior: Should generate HALT token (263) immediately")

print("\nCreating runner...")
runner = AutoregressiveVMRunner()

# Patch to show tokens
tokens_generated = []
step_count = [0]
original_generate = runner.model.generate_next

def tracking_generate(context, **kwargs):
    step_count[0] += 1
    result = original_generate(context, **kwargs)
    tokens_generated.append(result)

    if step_count[0] <= 50:
        token_name = ""
        if result == Token.HALT:
            token_name = " <- HALT!"
        elif result == Token.STEP_END:
            token_name = " <- STEP_END"
        elif result >= 257 and result <= 268:
            marker_names = {
                257: "PC", 258: "AX", 259: "SP", 260: "BP",
                261: "STACK0", 262: "MEM", 263: "HALT",
                264: "STEP_END", 265: "DATA_END", 266: "TOOL_CALL",
                267: "THINKING_START", 268: "THINKING_END"
            }
            token_name = f" <- {marker_names.get(result, 'MARKER')}"

        print(f"  Token {step_count[0]:3d}: {result:3d}{token_name}")

    if step_count[0] >= 50 and step_count[0] % 10 == 0:
        print(f"  Token {step_count[0]:3d}: {result:3d}")

    if step_count[0] >= 100:
        print("\n⚠️  Hit 100 token limit")
        raise RuntimeError("Max tokens")

    return result

runner.model.generate_next = tracking_generate

print("\nRunning...\n")

try:
    result = runner.run(bytecode, data, max_steps=10)
    print(f"\n{'=' * 80}")
    print(f"Result: {result}")

    if isinstance(result, tuple):
        output, exit_code = result
        print(f"  Output: '{output}'")
        print(f"  Exit code: {exit_code}")

    print(f"\nTotal tokens generated: {step_count[0]}")

    if Token.HALT in tokens_generated:
        halt_pos = tokens_generated.index(Token.HALT)
        print(f"✅ HALT token found at position {halt_pos}")
        print(f"   Tokens before HALT: {tokens_generated[:halt_pos]}")
    else:
        print("❌ HALT token NEVER generated!")
        print(f"   First 35 tokens: {tokens_generated[:35]}")
        print(f"   Looking for token 263 (HALT)...")

except RuntimeError as e:
    if "Max tokens" in str(e):
        print(f"\n❌ No HALT after 100 tokens")
        print(f"   Generated tokens: {tokens_generated[:50]}")
        if Token.HALT in tokens_generated:
            print("   But HALT was in the list somehow??")
        else:
            print(f"   HALT (263) not found in: {set(tokens_generated)}")
    else:
        raise

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
