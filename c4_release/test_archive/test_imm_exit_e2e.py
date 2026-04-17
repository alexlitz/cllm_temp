#!/usr/bin/env python3
"""End-to-end test of IMM+EXIT program."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

BD = _SetDim

print("Building model...")
model = AutoregressiveVM()
set_vm_weights(model)
model = model.cuda()
model.eval()

runner = AutoregressiveVMRunner()
runner.model = model
bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]

# Use runner's context builder
context = runner._build_context(bytecode, b'', [])
print(f"Initial context: {len(context)} tokens")
print(f"Program: IMM 42, EXIT\n")

# Generate step 0 (IMM)
print("=" * 60)
print("STEP 0: IMM 42")
print("=" * 60)

for i in range(100):
    tok = model.generate_next(context)
    context.append(tok)
    if tok == Token.STEP_END:
        print(f"Step 0 complete ({len(context)} total tokens)\n")
        break

# Check step 0 PC
se0_pos = len(context) - 1
for pos in range(20, se0_pos):
    if context[pos] == 257:  # REG_PC
        pc_bytes = context[pos+1:pos+5]
        pc = sum(b << (i*8) for i, b in enumerate(pc_bytes))
        print(f"Step 0 PC: 0x{pc:08x} (expected 0x0a)")
        if pc == 0x0a:
            print("  ✓ PC correct")
        else:
            print(f"  ✗ PC wrong!")
        break

# Check step 0 AX
for pos in range(20, se0_pos):
    if context[pos] == 260:  # REG_AX
        ax_bytes = context[pos+1:pos+5]
        ax = sum(b << (i*8) for i, b in enumerate(ax_bytes))
        print(f"Step 0 AX: 0x{ax:08x} (expected 0x0000002a)")
        if ax == 0x2a:
            print("  ✓ AX correct")
        else:
            print(f"  ✗ AX wrong!")
        break

# Generate step 1 (EXIT)
print("\n" + "=" * 60)
print("STEP 1: EXIT")
print("=" * 60)

step1_start = len(context)
for i in range(100):
    tok = model.generate_next(context)
    context.append(tok)
    if tok == Token.STEP_END or tok == Token.HALT:
        if tok == Token.HALT:
            print(f"Step 1 complete with HALT ({len(context)} total tokens)\n")
        else:
            print(f"Step 1 complete with STEP_END ({len(context)} total tokens)\n")
        break

# Check step 1 PC
se1_pos = len(context) - 1
for pos in range(step1_start, se1_pos):
    if context[pos] == 257:  # REG_PC
        pc_bytes = context[pos+1:pos+5]
        pc = sum(b << (i*8) for i, b in enumerate(pc_bytes))
        print(f"Step 1 PC: 0x{pc:08x} (expected 0x12)")
        if pc == 0x12:
            print("  ✓ PC correct")
        else:
            print(f"  ✗ PC wrong!")
        break

# Check step 1 AX (should be preserved)
for pos in range(step1_start, se1_pos):
    if context[pos] == 260:  # REG_AX
        ax_bytes = context[pos+1:pos+5]
        ax = sum(b << (i*8) for i, b in enumerate(ax_bytes))
        print(f"Step 1 AX: 0x{ax:08x} (expected 0x0000002a)")
        if ax == 0x2a:
            print("  ✓ AX preserved (exit code 42)")
        else:
            print(f"  ✗ AX wrong!")
        break

# Check if HALT was generated
if context[-1] == Token.HALT:
    print("\n" + "=" * 60)
    print("✓✓✓ SUCCESS! HALT generated")
    print("=" * 60)
    print("Exit code: 42")
elif context[-1] == Token.STEP_END:
    print("\n" + "=" * 60)
    print("✗ HALT not generated (got STEP_END instead)")
    print("=" * 60)

    # Check if we can generate HALT by continuing
    print("\nTrying to generate HALT...")
    for i in range(10):
        tok = model.generate_next(context)
        context.append(tok)
        print(f"  Token {i}: {tok}")
        if tok == Token.HALT:
            print("  ✓ HALT generated!")
            break
else:
    print(f"\n✗ Unexpected final token: {context[-1]}")
