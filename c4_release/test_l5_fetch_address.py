#!/usr/bin/env python3
"""Check what address L5 is fetching from."""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import _SetDim as BD, Token
import inspect

# Simple program: IMM 42, EXIT
bytecode = [1 | (42 << 8), 34 | (0 << 8)]

print("Bytecode:")
print(f"  [0] = opcode {bytecode[0]&0xFF} (IMM), imm {bytecode[0]>>8} (42)")
print(f"  [1] = opcode {bytecode[1]&0xFF} (EXIT), imm {bytecode[1]>>8} (0)")

sig = inspect.signature(AutoregressiveVMRunner.__init__)
if 'conversational_io' in sig.parameters:
    runner = AutoregressiveVMRunner(conversational_io=False)
else:
    runner = AutoregressiveVMRunner()

# Hook L5 attention to see what it fetches
model = runner.model
original_l5_attn = model.blocks[5].attn.forward
step_count = [0]

def track_l5(x, kv_cache=None):
    # Only check first 2 steps
    if step_count[0] < 2:
        B, S_len, D = x.shape
        print(f"\n=== Step {step_count[0]}, L5 input, context len {S_len} ===")

        # Find PC and AX markers
        pc_marker_pos = None
        ax_marker_pos = None

        for pos in range(S_len):
            is_mark = x[0, pos, BD.IS_MARK].item()
            mark_pc = x[0, pos, BD.MARK_PC].item() if hasattr(BD, 'MARK_PC') else 0
            mark_ax = x[0, pos, BD.MARK_AX].item() if hasattr(BD, 'MARK_AX') else 0

            if is_mark > 0.5:
                if mark_pc > 0.5:
                    pc_marker_pos = pos
                elif mark_ax > 0.5:
                    ax_marker_pos = pos

        if pc_marker_pos is not None:
            print(f"  PC marker at position {pc_marker_pos}")
            embed_lo = [x[0, pc_marker_pos, BD.EMBED_LO + k].item() for k in range(16)]
            embed_hi = [x[0, pc_marker_pos, BD.EMBED_HI + k].item() for k in range(16)]
            print(f"    EMBED_LO: {[f'{k}:{v:.2f}' for k,v in enumerate(embed_lo) if abs(v) > 0.1]}")
            print(f"    EMBED_HI: {[f'{k}:{v:.2f}' for k,v in enumerate(embed_hi) if abs(v) > 0.1]}")

        if ax_marker_pos is not None:
            print(f"  AX marker at position {ax_marker_pos}")
            embed_lo = [x[0, ax_marker_pos, BD.EMBED_LO + k].item() for k in range(16)]
            embed_hi = [x[0, ax_marker_pos, BD.EMBED_HI + k].item() for k in range(16)]
            print(f"    EMBED_LO: {[f'{k}:{v:.2f}' for k,v in enumerate(embed_lo) if abs(v) > 0.1]}")
            print(f"    EMBED_HI: {[f'{k}:{v:.2f}' for k,v in enumerate(embed_hi) if abs(v) > 0.1]}")

            # Decode PC value from EMBED at AX marker
            lo_nibble = max(range(16), key=lambda k: embed_lo[k])
            hi_nibble = max(range(16), key=lambda k: embed_hi[k])
            pc_value = (hi_nibble << 4) | lo_nibble
            print(f"    → Fetch address (PC): {pc_value} (0x{pc_value:02x})")

            # Convert to instruction index
            from neural_vm.constants import PC_OFFSET, INSTR_WIDTH
            if pc_value >= PC_OFFSET:
                idx = (pc_value - PC_OFFSET) // INSTR_WIDTH
                print(f"    → Instruction index: {idx}")
                if idx < len(bytecode):
                    expected_opcode = bytecode[idx] & 0xFF
                    print(f"    → Expected opcode: {expected_opcode}")

            step_count[0] += 1

    return original_l5_attn(x, kv_cache)

model.blocks[5].attn.forward = track_l5

print("\nRunning...")
output, exit_code = runner.run(bytecode, b'', [], max_steps=3)
print(f"\nFinal exit code: {exit_code}")
