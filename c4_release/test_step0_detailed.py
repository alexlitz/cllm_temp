#\!/usr/bin/env python3
"""Detailed check of step 0 (IMM execution)."""
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
print(f"Program: IMM 42, EXIT")
print(f"Initial context: {len(context)} tokens\n")

# Generate step 0
for i in range(100):
    tok = model.generate_next(context)
    context.append(tok)
    if tok == Token.STEP_END:
        break

print(f"Step 0 complete\n")

# Find REG_AX position in step 0
reg_ax_pos = None
for pos in range(20, len(context)):
    if context[pos] == 260:  # REG_AX
        reg_ax_pos = pos
        break

if reg_ax_pos is None:
    print("REG_AX not found in step 0\!")
else:
    print(f"REG_AX at position {reg_ax_pos}\n")

    # Check what the model sees at this position
    token_ids = torch.tensor([context], dtype=torch.long, device='cuda')

    with torch.no_grad():
        x = model.embed(token_ids)

        print("After embedding:")
        print(f"  HAS_SE: {x[0, reg_ax_pos, BD.HAS_SE].item():.3f}")
        print(f"  MARK_PC: {x[0, reg_ax_pos, BD.MARK_PC].item():.3f}")

        # After Layer 1 (HAS_SE detection)
        x = model.blocks[0](x, kv_cache=None)
        print(f"\nAfter Layer 1:")
        print(f"  HAS_SE: {x[0, reg_ax_pos, BD.HAS_SE].item():.3f}")

        # After Layer 5 (immediate fetch)
        for i in range(1, 5):
            x = model.blocks[i](x, kv_cache=None)

        print(f"\nAfter Layer 5:")
        print(f"  HAS_SE: {x[0, reg_ax_pos, BD.HAS_SE].item():.3f}")
        print(f"  OP_IMM: {x[0, reg_ax_pos, BD.OP_IMM].item():.3f}")

        # Check FETCH values
        fetch_lo_vals = []
        for i in range(16):
            val = x[0, reg_ax_pos, BD.FETCH_LO + i].item()
            if abs(val) > 0.1:
                fetch_lo_vals.append((i, val))

        if fetch_lo_vals:
            print(f"\n  FETCH_LO values:")
            for i, val in fetch_lo_vals:
                print(f"    FETCH_LO[{i}] = {val:.3f}")
        else:
            print(f"\n  FETCH_LO: all zero")

        fetch_hi_vals = []
        for i in range(16):
            val = x[0, reg_ax_pos, BD.FETCH_HI + i].item()
            if abs(val) > 0.1:
                fetch_hi_vals.append((i, val))

        if fetch_hi_vals:
            print(f"\n  FETCH_HI values:")
            for i, val in fetch_hi_vals:
                print(f"    FETCH_HI[{i}] = {val:.3f}")
        else:
            print(f"\n  FETCH_HI: all zero")

        # Check what byte FETCH predicts
        if fetch_lo_vals:
            fetch_byte = max(fetch_lo_vals, key=lambda x: x[1])[0]
            print(f"\n  FETCH predicts byte 0: 0x{fetch_byte:02x}")
            if fetch_byte == 0x2a:
                print("    ✓ CORRECT (0x2a = 42)")
            else:
                print(f"    ✗ WRONG (expected 0x2a)")
