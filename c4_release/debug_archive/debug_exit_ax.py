#!/usr/bin/env python3
"""Debug EXIT AX corruption - why is AX becoming 154?"""

import torch
from neural_vm.run_vm import AutoregressiveVMRunner, Token
from neural_vm.vm_step import _SetDim as BD
from neural_vm.embedding import Opcode

bytecode = [
    Opcode.IMM | (42 << 8),  # [0] IMM 42
    Opcode.EXIT,             # [1] EXIT
]

runner = AutoregressiveVMRunner()
runner._func_call_handlers = {}
runner._bytecode = bytecode
runner._last_sp = 0x1F800
runner._last_bp = 0x10000

ctx = runner._build_context(bytecode, b"", [])
runner.model.set_active_opcode(bytecode[0] & 0xFF)

# Step 0: IMM 42
print("=== Step 0: IMM 42 ===")
step_tokens = []
for _ in range(35):
    next_token = runner.model.generate_next(ctx, max_context_window=2048)
    ctx.append(next_token)
    step_tokens.append(next_token)
    if next_token == Token.HALT:
        break

def extract_reg(tokens, marker):
    try:
        idx = tokens.index(marker)
        if idx + 4 < len(tokens):
            b = tokens[idx+1:idx+5]
            return b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)
    except ValueError:
        pass
    return None

ax = extract_reg(step_tokens, Token.REG_AX)
print(f"Step 0 AX output: {ax}")

# Set opcode for step 1 (EXIT)
runner.model.set_active_opcode(Opcode.EXIT)

# Step 1: EXIT - generate only REG_AX marker and check state
print("\n=== Step 1: EXIT - debugging AX marker position ===")

# Generate tokens until AX marker
for _ in range(20):
    next_token = runner.model.generate_next(ctx, max_context_window=2048)
    ctx.append(next_token)
    if next_token == Token.REG_AX:
        print(f"Generated REG_AX at position {len(ctx)-1}")
        break

# Now check the model state at this position
with torch.no_grad():
    device = next(runner.model.parameters()).device
    token_ids = torch.tensor([ctx], dtype=torch.long, device=device)

    # Full forward pass
    x = runner.model.embed(token_ids)
    for block in runner.model.blocks:
        x = block(x)

    pos = x[0, -1, :]  # Last position (AX marker)

    print(f"\nAt AX marker position:")
    print(f"  MARK_AX: {pos[BD.MARK_AX].item():.2f}")
    print(f"  HAS_SE: {pos[BD.HAS_SE].item():.2f}")  # Critical for L3 head 5
    print(f"  OP_EXIT: {pos[BD.OP_EXIT].item():.2f}")
    print(f"  OP_IMM: {pos[BD.OP_IMM].item():.2f}")
    print(f"  IS_BYTE: {pos[BD.IS_BYTE].item():.2f}")

    # Check AX_FULL (should have 42)
    ax_full_lo = [pos[BD.AX_FULL_LO + k].item() for k in range(16)]
    ax_full_hi = [pos[BD.AX_FULL_HI + k].item() for k in range(16)]
    print(f"\n  AX_FULL_LO: {ax_full_lo}")
    print(f"  AX_FULL_HI: {ax_full_hi}")
    max_ax_full_lo = ax_full_lo.index(max(ax_full_lo)) if max(ax_full_lo) > 0.1 else -1
    max_ax_full_hi = ax_full_hi.index(max(ax_full_hi)) if max(ax_full_hi) > 0.1 else -1
    print(f"  AX_FULL max: lo={max_ax_full_lo}, hi={max_ax_full_hi}")

    if max_ax_full_lo >= 0 and max_ax_full_hi >= 0:
        ax_full_val = max_ax_full_lo | (max_ax_full_hi << 4)
        print(f"  → AX_FULL byte 0 = 0x{ax_full_val:02x} = {ax_full_val}")
    else:
        print(f"  → AX_FULL not populated!")

    # Check OUTPUT (what gets written)
    out_lo = [pos[BD.OUTPUT_LO + k].item() for k in range(16)]
    out_hi = [pos[BD.OUTPUT_HI + k].item() for k in range(16)]
    max_out_lo = out_lo.index(max(out_lo))
    max_out_hi = out_hi.index(max(out_hi))
    print(f"\n  OUTPUT_LO max: idx={max_out_lo}, val={max(out_lo):.2f}")
    print(f"  OUTPUT_HI max: idx={max_out_hi}, val={max(out_hi):.2f}")
    out_byte = max_out_lo | (max_out_hi << 4)
    print(f"  → OUTPUT byte 0 = 0x{out_byte:02x} = {out_byte}")

    # Check what would produce 154
    print(f"\n=== 154 = 0x9A = lo=10, hi=9 ===")
    print(f"  OUTPUT_LO[10]: {out_lo[10]:.2f}")
    print(f"  OUTPUT_HI[9]: {out_hi[9]:.2f}")

    # Check L3 head 5 state at AX marker
    print(f"\n=== L3 head 5 state (AX_FULL relay) ===")
    # Re-run forward pass up to L3 to check intermediate state
    x_test = runner.model.embed(token_ids)
    for i in range(4):  # Run through blocks 0-3 (L1-L4)
        x_test = runner.model.blocks[i](x_test)

    pos_l4 = x_test[0, -1, :]
    ax_full_lo_l4 = [pos_l4[BD.AX_FULL_LO + k].item() for k in range(16)]
    ax_full_hi_l4 = [pos_l4[BD.AX_FULL_HI + k].item() for k in range(16)]
    print(f"  After L4 - AX_FULL_LO: {ax_full_lo_l4}")
    print(f"  After L4 - AX_FULL_HI max: {max(ax_full_hi_l4):.4f}")
    print(f"  After L4 - HAS_SE: {pos_l4[BD.HAS_SE].item():.2f}")
