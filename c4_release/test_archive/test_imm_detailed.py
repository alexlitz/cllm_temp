#\!/usr/bin/env python3
"""Detailed check of IMM execution."""
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

context = runner._build_context(bytecode, b'', [])
print(f"Program: IMM 42 (0x{42:02x}), EXIT")
print(f"Bytecode[0]: op=0x{bytecode[0] & 0xFF:02x}, imm=0x{bytecode[0] >> 8:08x}\n")

# Generate step 0 (IMM execution)
print("=" * 60)
print("STEP 0 (IMM execution)")
print("=" * 60)

for i in range(100):
    tok = model.generate_next(context)
    context.append(tok)
    if tok == Token.STEP_END:
        break

print(f"Step 0 generated {len(context) - 20} tokens\n")

# Find and display AX in step 0
for pos in range(20, len(context)):
    if context[pos] == 260:  # REG_AX
        ax_bytes = [context[pos+i] for i in range(1, 5)]
        ax = sum(b << (i*8) for i, b in enumerate(ax_bytes))
        print(f"Step 0 AX register:")
        print(f"  Bytes: {[hex(b) for b in ax_bytes]}")
        print(f"  Value: 0x{ax:08x} = {ax}")
        print(f"  Expected: 0x0000002a = 42")

        if ax == 0x2a:
            print("  ✓ CORRECT")
        else:
            print(f"  ✗ WRONG")
        break

# Check what opcode was fetched for step 1
print("\n" + "=" * 60)
print("STEP 1 (EXIT execution)")
print("=" * 60)

# Generate step 1 up to REG_AX
step1_start = len(context)
for i in range(50):
    tok = model.generate_next(context)
    context.append(tok)
    if tok == 260:  # REG_AX
        print(f"\nREG_AX marker at position {len(context)-1}")
        break

# Check what the model sees at step 1
token_ids = torch.tensor([context], dtype=torch.long, device='cuda')

with torch.no_grad():
    x = model.embed(token_ids)

    # Check after Layer 5 (where immediate fetch happens)
    for i in range(6):
        x = model.blocks[i](x, kv_cache=None)

    reg_ax_pos = len(context) - 1

    print(f"\nAfter Layer 5, at REG_AX position {reg_ax_pos}:")
    print(f"  HAS_SE: {x[0, reg_ax_pos, BD.HAS_SE].item():.3f}")
    print(f"  OP_IMM: {x[0, reg_ax_pos, BD.OP_IMM].item():.3f}")
    print(f"  OP_EXIT: {x[0, reg_ax_pos, BD.OP_EXIT].item():.3f}")

    # Check immediate fetch values (FETCH_LO/HI)
    print(f"\n  FETCH_LO values:")
    for i in range(16):
        val = x[0, reg_ax_pos, BD.FETCH_LO + i].item()
        if abs(val) > 0.1:
            print(f"    FETCH_LO[{i}] = {val:.3f}")

    print(f"\n  FETCH_HI values:")
    for i in range(16):
        val = x[0, reg_ax_pos, BD.FETCH_HI + i].item()
        if abs(val) > 0.1:
            print(f"    FETCH_HI[{i}] = {val:.3f}")

    # Check AX_CARRY values
    print(f"\n  AX_CARRY_LO values:")
    for i in range(16):
        val = x[0, reg_ax_pos, BD.AX_CARRY_LO + i].item()
        if abs(val) > 0.1:
            print(f"    AX_CARRY_LO[{i}] = {val:.3f}")

    print(f"\n  AX_CARRY_HI values:")
    for i in range(16):
        val = x[0, reg_ax_pos, BD.AX_CARRY_HI + i].item()
        if abs(val) > 0.1:
            print(f"    AX_CARRY_HI[{i}] = {val:.3f}")
