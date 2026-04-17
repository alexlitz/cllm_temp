#!/usr/bin/env python3
"""Debug JMP output routing - why isn't PC jumping to target?"""

import torch
from neural_vm.run_vm import AutoregressiveVMRunner, Token
from neural_vm.vm_step import _SetDim as BD
from neural_vm.embedding import Opcode

# Simple JMP 4 test
bytecode = [
    Opcode.JMP | (4 << 8),    # [0] JMP to instruction 4
    Opcode.IMM | (99 << 8),   # [1] IMM 99 (should be skipped)
    Opcode.EXIT,              # [2] EXIT 99 (should be skipped)
    Opcode.NOP,               # [3] NOP padding
    Opcode.IMM | (42 << 8),   # [4] IMM 42 (target)
    Opcode.EXIT,              # [5] EXIT with 42
]
data = b""
target = (bytecode[0] >> 8) & 0xFFFFFF
print(f"Bytecode[0] = 0x{bytecode[0]:08x}")
print(f"Opcode: {bytecode[0] & 0xFF} (JMP={Opcode.JMP})")
print(f"Target INDEX: {target}")
print(f"Expected PC: {target} * 8 + 2 = {target * 8 + 2} = 0x{target * 8 + 2:02x}")

runner = AutoregressiveVMRunner()
runner._func_call_handlers = {}
runner._bytecode = bytecode
runner._last_sp = 0x1F800
runner._last_bp = 0x10000

ctx = runner._build_context(bytecode, data, [])

# Set active opcode to JMP
runner.model.set_active_opcode(Opcode.JMP)

# Generate REG_PC marker with proper context window
next_token = runner.model.generate_next(ctx, max_context_window=2048)
ctx.append(next_token)
print(f"\nFirst token: {next_token} (REG_PC={Token.REG_PC})")

# Now let's look at the model output dimensions at the last position
with torch.no_grad():
    device = next(runner.model.parameters()).device
    token_ids = torch.tensor([ctx], dtype=torch.long, device=device)

    # Full forward pass
    x = runner.model.embed(token_ids)
    for i, block in enumerate(runner.model.blocks):
        x = block(x)

    # Check OUTPUT dimensions at last position
    last_pos = x[0, -1, :]

    print(f"\n=== OUTPUT Dimensions (for PC byte generation) ===")
    out_lo = [last_pos[BD.OUTPUT_LO + k].item() for k in range(16)]
    out_hi = [last_pos[BD.OUTPUT_HI + k].item() for k in range(16)]
    print(f"OUTPUT_LO[0:16]: {out_lo}")
    print(f"OUTPUT_HI[0:16]: {out_hi}")

    # Check which nibble has max value
    max_lo = out_lo.index(max(out_lo))
    max_hi = out_hi.index(max(out_hi))
    print(f"\nMax OUTPUT: lo={max_lo}, hi={max_hi} → byte 0x{max_lo + (max_hi << 4):02x}")

    # Check CMP dimensions (critical for JMP detection)
    print(f"\n=== CMP Dimensions (JMP detection) ===")
    cmp_vals = [last_pos[BD.CMP + k].item() for k in range(8)]
    print(f"CMP[0:8]: {cmp_vals}")

    # Check FETCH dimensions
    print(f"\n=== FETCH Dimensions (immediate byte) ===")
    fetch_lo = [last_pos[BD.FETCH_LO + k].item() for k in range(16)]
    fetch_hi = [last_pos[BD.FETCH_HI + k].item() for k in range(16)]
    print(f"FETCH_LO[0:16]: {fetch_lo}")
    print(f"FETCH_HI[0:16]: {fetch_hi}")
    max_fetch_lo = fetch_lo.index(max(fetch_lo))
    max_fetch_hi = fetch_hi.index(max(fetch_hi))
    print(f"Max FETCH: lo={max_fetch_lo}, hi={max_fetch_hi} → byte 0x{max_fetch_lo + (max_fetch_hi << 4):02x}")

    # Check AX_CARRY dimensions
    print(f"\n=== AX_CARRY Dimensions (JMP target relayed here) ===")
    ax_carry_lo = [last_pos[BD.AX_CARRY_LO + k].item() for k in range(16)]
    ax_carry_hi = [last_pos[BD.AX_CARRY_HI + k].item() for k in range(16)]
    print(f"AX_CARRY_LO: {ax_carry_lo}")
    print(f"AX_CARRY_HI: {ax_carry_hi}")
    max_axc_lo = ax_carry_lo.index(max(ax_carry_lo)) if max(ax_carry_lo) > 0.1 else -1
    max_axc_hi = ax_carry_hi.index(max(ax_carry_hi)) if max(ax_carry_hi) > 0.1 else -1
    print(f"Max AX_CARRY: lo={max_axc_lo}, hi={max_axc_hi}")

    # Check EMBED dimensions (initial PC value for increment)
    print(f"\n=== EMBED Dimensions (initial value) ===")
    embed_lo = [last_pos[BD.EMBED_LO + k].item() for k in range(16)]
    embed_hi = [last_pos[BD.EMBED_HI + k].item() for k in range(16)]
    print(f"EMBED_LO: {embed_lo}")
    print(f"EMBED_HI: {embed_hi}")

    # Check opcode flags
    print(f"\n=== Opcode Flags ===")
    print(f"OP_JMP: {last_pos[BD.OP_JMP].item()}")
    print(f"HAS_SE: {last_pos[BD.HAS_SE].item()}")
    print(f"MARK_PC: {last_pos[BD.MARK_PC].item()}")
    print(f"MARK_AX: {last_pos[BD.MARK_AX].item()}")
    print(f"IS_BYTE: {last_pos[BD.IS_BYTE].item()}")

    # Check intermediate values before L6 FFN
    print(f"\n=== Layer-by-layer OUTPUT values ===")
    token_ids = torch.tensor([ctx], dtype=torch.long, device=device)
    x = runner.model.embed(token_ids)

    for i, block in enumerate(runner.model.blocks):
        x = block(x)
        pos = x[0, -1, :]
        out_lo = [pos[BD.OUTPUT_LO + k].item() for k in range(16)]
        out_hi = [pos[BD.OUTPUT_HI + k].item() for k in range(16)]
        max_lo_val = max(out_lo)
        max_hi_val = max(out_hi)
        max_lo = out_lo.index(max_lo_val) if max_lo_val > 0.1 else -1
        max_hi = out_hi.index(max_hi_val) if max_hi_val > 0.1 else -1
        if i in [5, 6, 7, 8, 12, 13, 14, 15, 16]:  # Key layers
            print(f"  L{i+1}: OUTPUT max_lo={max_lo} ({max_lo_val:.2f}), max_hi={max_hi} ({max_hi_val:.2f})")
            if i == 5:  # L6 - check AX_CARRY too
                axc_lo = [pos[BD.AX_CARRY_LO + k].item() for k in range(16)]
                axc_hi = [pos[BD.AX_CARRY_HI + k].item() for k in range(16)]
                max_axc_lo = axc_lo.index(max(axc_lo)) if max(axc_lo) > 0.1 else -1
                print(f"    L{i+1} AX_CARRY_LO max={max_axc_lo} ({max(axc_lo):.2f})")

    # Expected: JMP target INDEX=4
    # PC = 4 * 8 + 2 = 34 = 0x22 (lo=2, hi=2)
    print(f"\n=== Expected ===")
    print(f"FETCH byte 0: 0x04 (lo=4, hi=0) - the JMP INDEX")
    print(f"OUTPUT byte 0: 0x22 (lo=2, hi=2) - the converted PC")
