#!/usr/bin/env python3
"""Check AX_CARRY values in step 1 vs step 0."""

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

# Hook L6 FFN to check AX_CARRY
model = runner.model
original_l6_ffn = model.blocks[6].ffn.forward
step_count = [0]

def track_l6(x):
    result = original_l6_ffn(x)

    # Check first two steps only
    if step_count[0] < 2:
        B, S_len, D = result.shape

        # Find AX marker
        for pos in range(S_len):
            is_mark = result[0, pos, BD.IS_MARK].item()
            mark_ax = result[0, pos, BD.MARK_AX].item() if hasattr(BD, 'MARK_AX') else 0

            if is_mark > 0.5 and mark_ax > 0.5:
                print(f"\n=== Step {step_count[0]}, AX marker at pos {pos} ===")

                # Check what's in AX_CARRY (input to this layer)
                ax_carry_lo_in = [x[0, pos, BD.AX_CARRY_LO + k].item() for k in range(16)]
                ax_carry_hi_in = [x[0, pos, BD.AX_CARRY_HI + k].item() for k in range(16)]
                print(f"AX_CARRY input (before L6):")
                print(f"  LO: {[f'{k}:{v:.2f}' for k,v in enumerate(ax_carry_lo_in) if abs(v) > 0.1]}")
                print(f"  HI: {[f'{k}:{v:.2f}' for k,v in enumerate(ax_carry_hi_in) if abs(v) > 0.1]}")

                # Check what's in OUTPUT (output from this layer)
                output_lo = [result[0, pos, BD.OUTPUT_LO + k].item() for k in range(16)]
                output_hi = [result[0, pos, BD.OUTPUT_HI + k].item() for k in range(16)]
                print(f"OUTPUT (after L6):")
                print(f"  LO: {[f'{k}:{v:.2f}' for k,v in enumerate(output_lo) if abs(v) > 0.1]}")
                print(f"  HI: {[f'{k}:{v:.2f}' for k,v in enumerate(output_hi) if abs(v) > 0.1]}")

                step_count[0] += 1
                break

    return result

model.blocks[6].ffn.forward = track_l6

print("Running IMM 42, EXIT...")
output, exit_code = runner.run(bytecode, b'', [], max_steps=3)
print(f"\nFinal exit code: {exit_code} (expected 42)")
