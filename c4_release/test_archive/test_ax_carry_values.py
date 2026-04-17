#!/usr/bin/env python3
"""Check what values are in AX_CARRY vs OUTPUT."""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import _SetDim as BD, Token
import inspect

# Simple program: IMM 42, EXIT
bytecode = [1 | (42 << 8), 34 | (0 << 8)]

sig = inspect.signature(AutoregressiveVMRunner.__init__)
if 'conversational_io' in sig.parameters:
    runner = AutoregressiveVMRunner(conversational_io=False)
else:
    runner = AutoregressiveVMRunner()

# Hook after L6 AND L15 to see values
model = runner.model
original_l6_ffn_forward = model.blocks[6].ffn.forward
original_l15_ffn_forward = model.blocks[15].ffn.forward

printed = [False, False]

def track_after_l6(x):
    result = original_l6_ffn_forward(x)

    # Only print once
    if not printed[0]:
        B, S_len, D = result.shape
        # Find AX marker
        for pos in range(S_len):
            is_mark = result[0, pos, BD.IS_MARK].item()
            mark_ax = result[0, pos, BD.MARK_AX].item() if hasattr(BD, 'MARK_AX') else 0
            if is_mark > 0.5 and mark_ax > 0.5:
                print(f"\n=== After L6: Position {pos} (AX marker, context len {S_len}) ===")
                output_lo = [result[0, pos, BD.OUTPUT_LO + k].item() for k in range(16)]
                output_hi = [result[0, pos, BD.OUTPUT_HI + k].item() for k in range(16)]
                print(f"OUTPUT_LO: {[f'{k}:{v:.2f}' for k,v in enumerate(output_lo) if abs(v) > 0.1]}")
                print(f"OUTPUT_HI: {[f'{k}:{v:.2f}' for k,v in enumerate(output_hi) if abs(v) > 0.1]}")
                printed[0] = True
                break
    return result

def track_after_l15(x):
    result = original_l15_ffn_forward(x)

    # Only print once
    if not printed[1]:
        B, S_len, D = result.shape
        # Find AX marker
        for pos in range(S_len):
            is_mark = result[0, pos, BD.IS_MARK].item()
            mark_ax = result[0, pos, BD.MARK_AX].item() if hasattr(BD, 'MARK_AX') else 0
            if is_mark > 0.5 and mark_ax > 0.5:
                print(f"\n=== After L15: Position {pos} (AX marker, context len {S_len}) ===")
                output_lo = [result[0, pos, BD.OUTPUT_LO + k].item() for k in range(16)]
                output_hi = [result[0, pos, BD.OUTPUT_HI + k].item() for k in range(16)]
                print(f"OUTPUT_LO: {[f'{k}:{v:.2f}' for k,v in enumerate(output_lo)]}")
                print(f"OUTPUT_HI: {[f'{k}:{v:.2f}' for k,v in enumerate(output_hi)]}")
                print(f"Max LO nibble: {max(range(16), key=lambda k: output_lo[k])} = {max(output_lo):.2f}")
                print(f"Max HI nibble: {max(range(16), key=lambda k: output_hi[k])} = {max(output_hi):.2f}")
                printed[1] = True
                break
    return result

model.blocks[6].ffn.forward = track_after_l6
model.blocks[15].ffn.forward = track_after_l15

print("Running IMM 42, EXIT...")
try:
    output, exit_code = runner.run(bytecode, b'', [], max_steps=3)
    print(f"\nFinal exit code: {exit_code} (0x{exit_code:08x})")
    if exit_code == 42:
        print("✓ Correct!")
    else:
        print(f"✗ Expected 42")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
