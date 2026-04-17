#!/usr/bin/env python3
"""Compare IMM vs JSR opcode fetch to understand the difference."""

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode

print("=" * 80)
print("IMM vs JSR COMPARISON")
print("=" * 80)

# Test 1: IMM (works)
print("\n### TEST 1: IMM 42; EXIT ###")
bytecode_imm = [Opcode.IMM | (42 << 8), Opcode.EXIT]
runner_imm = AutoregressiveVMRunner()
context_imm = runner_imm._build_context(bytecode_imm, b"", [], "")

print(f"Context: {context_imm[:15]}")
print(f"Code bytes:")
for i in range(16):
    if i < len(context_imm):
        print(f"  Position {i}: token {context_imm[i]}")

result_imm = runner_imm.run(bytecode_imm, b"", max_steps=3)
if isinstance(result_imm, tuple):
    _, exit_code_imm = result_imm
else:
    exit_code_imm = result_imm

print(f"\nResult: exit code = {exit_code_imm}")
if exit_code_imm == 42:
    print("✅ IMM WORKS")
else:
    print(f"❌ IMM FAILED (expected 42, got {exit_code_imm})")

# Test 2: JSR (broken)
print("\n### TEST 2: JSR 25; EXIT ###")
bytecode_jsr = [Opcode.JSR | (25 << 8), Opcode.EXIT]
runner_jsr = AutoregressiveVMRunner()
context_jsr = runner_jsr._build_context(bytecode_jsr, b"", [], "")

print(f"Context: {context_jsr[:15]}")
print(f"Code bytes:")
for i in range(16):
    if i < len(context_jsr):
        print(f"  Position {i}: token {context_jsr[i]}")

# Just generate first step to see PC
print("\nGenerating first step...")
for i in range(35):
    next_token = runner_jsr.model.generate_next(context_jsr)
    context_jsr.append(next_token)
    
    if i == 1:  # PC byte 0
        pc_byte0 = next_token

print(f"PC byte 0 after first step: {pc_byte0}")
if pc_byte0 == 25:
    print("✅ JSR WORKED - jumped to 25")
elif pc_byte0 == 10:
    print("❌ JSR FAILED - PC=10 (normal advancement, didn't jump)")
else:
    print(f"⚠️  JSR unexpected - PC byte 0 = {pc_byte0}")

print("\n" + "=" * 80)
print("ANALYSIS:")
print("=" * 80)

print("\nIMM bytecode:")
print(f"  Opcode: {Opcode.IMM} (0x{Opcode.IMM:02x})")
print(f"  Nibbles: lo={Opcode.IMM & 0xF}, hi={(Opcode.IMM >> 4) & 0xF}")

print("\nJSR bytecode:")
print(f"  Opcode: {Opcode.JSR} (0x{Opcode.JSR:02x})")
print(f"  Nibbles: lo={Opcode.JSR & 0xF}, hi={(Opcode.JSR >> 4) & 0xF}")

print("\nL5 head 2 fetches from address PC_OFFSET=2:")
imm_ctx_pos_2 = context_imm[2] if len(context_imm) > 2 else None
jsr_ctx_pos_2 = context_jsr[2] if len(context_jsr) > 2 else None

print(f"  IMM program, position 2: {imm_ctx_pos_2}")
print(f"  JSR program, position 2: {jsr_ctx_pos_2}")

if imm_ctx_pos_2 is not None:
    print(f"\n  IMM @ pos 2: nibbles lo={imm_ctx_pos_2 & 0xF}, hi={(imm_ctx_pos_2 >> 4) & 0xF}")
if jsr_ctx_pos_2 is not None:
    print(f"  JSR @ pos 2: nibbles lo={jsr_ctx_pos_2 & 0xF}, hi={(jsr_ctx_pos_2 >> 4) & 0xF}")

print("\nHypothesis: If L5 head 2 fetches from address 2 (PC_OFFSET),")
print("it would get the value at context position 1+2=3 (after CODE_START).")
print("This is immediate byte 1, which is 0 for both programs!")
