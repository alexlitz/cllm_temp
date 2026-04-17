#\!/usr/bin/env python3
"""Debug JSR neural path - check OPCODE_BYTE, TEMP[0], and FETCH after L5."""

import torch
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.vm_step import _SetDim as BD, Token

bytecode = [Opcode.JSR | (26 << 8)]
runner = AutoregressiveVMRunner()
model = runner.model

# Build context
context = runner._build_context(bytecode, b"", [], "")
device = next(model.parameters()).device

print("="*60)
print("DEBUGGING JSR NEURAL PATH")
print("="*60)
print(f"Device: {device}")
print(f"Initial context: {context}")

# Generate first step token by token, capturing intermediate states
for i in range(35):
    next_token = model.generate_next(context)
    context.append(next_token)

    # When PC marker is generated
    if i == 0:
        print(f"\nToken 0: {next_token} (should be PC marker 257)")

        # Get the current state
        x = torch.tensor([context], dtype=torch.long, device=device)
        x = model.embed(x)

        # Check CODE bytes in embedding
        print("\nCODE section in embedding:")
        for pos in range(min(5, len(context)-1)):
            op_jsr = x[0, pos+1, BD.OP_JSR].item()
            addr_key_lo = x[0, pos+1, BD.ADDR_KEY:BD.ADDR_KEY+16].argmax().item() if x[0, pos+1, BD.ADDR_KEY:BD.ADDR_KEY+16].max() > 0.5 else -1
            print(f"  Pos {pos+1}: token={context[pos+1]}, ADDR_KEY_lo={addr_key_lo}, OP_JSR={op_jsr:.2f}")

        # Run through L5 and check state
        for layer_idx in range(6):
            x = model.blocks[layer_idx](x)

        # After L5, check PC marker (last position)
        pc_pos = len(context) - 1
        print(f"\nAfter L5 (position {pc_pos}, token={context[pc_pos]}):")

        # OPCODE_BYTE
        opcode_lo = -1
        opcode_hi = -1
        for k in range(16):
            if x[0, pc_pos, BD.OPCODE_BYTE_LO + k].item() > 0.5:
                opcode_lo = k
            if x[0, pc_pos, BD.OPCODE_BYTE_HI + k].item() > 0.5:
                opcode_hi = k
        print(f"  OPCODE_BYTE: lo={opcode_lo}, hi={opcode_hi} (expected: lo=3, hi=0 for JSR)")

        # TEMP[0]
        temp0 = x[0, pc_pos, BD.TEMP + 0].item()
        print(f"  TEMP[0]: {temp0:.2f} (expected: >=5.0 for JSR decode)")

        # FETCH
        fetch_lo = -1
        for k in range(16):
            if x[0, pc_pos, BD.FETCH_LO + k].item() > 0.5:
                fetch_lo = k
        print(f"  FETCH_LO[0]: {fetch_lo} (expected: 10 for jump target 26 nibble)")

        # MARK_PC, HAS_SE
        mark_pc = x[0, pc_pos, BD.MARK_PC].item()
        has_se = x[0, pc_pos, BD.HAS_SE].item()
        print(f"  MARK_PC: {mark_pc:.2f}, HAS_SE: {has_se:.2f}")

# Check final PC value
pc_bytes = context[13:17]  # PC marker at 12, bytes at 13-16
pc_value = sum([b << (k*8) for k, b in enumerate(pc_bytes)])
print(f"\n{'='*60}")
print(f"FINAL PC VALUE: {pc_value}")
print(f"Expected: 26 if JSR works, 10 if fails")
print(f"JSR Neural Path: {'WORKING' if pc_value == 26 else 'BROKEN'}")
print(f"{'='*60}")
