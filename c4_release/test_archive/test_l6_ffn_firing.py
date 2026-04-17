#!/usr/bin/env python3
"""Check if L6 FFN IMM units are actually firing."""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import _SetDim as BD, Token
import inspect

print("=" * 70)
print("L6 FFN IMM UNIT ACTIVATION CHECK")
print("=" * 70)

# Simple program: IMM 42
bytecode = [1 | (42 << 8)]

sig = inspect.signature(AutoregressiveVMRunner.__init__)
if 'conversational_io' in sig.parameters:
    runner = AutoregressiveVMRunner(conversational_io=False)
else:
    runner = AutoregressiveVMRunner()

if torch.cuda.is_available():
    try:
        runner.model = runner.model.cuda()
    except:
        pass

# Hook the L6 FFN to see activations
model = runner.model
original_l6_ffn_forward = model.blocks[6].ffn.forward

def track_l6_ffn(x):
    # Check input at AX marker position (should be around position 5 in step)
    print(f"\nL6 FFN input shape: {x.shape}")

    # Find AX marker (IS_MARK and position ~5)
    B, S_len, D = x.shape
    for pos in range(min(10, S_len)):
        is_mark = x[0, pos, BD.IS_MARK].item()
        mark_ax = x[0, pos, BD.MARK_AX].item() if hasattr(BD, 'MARK_AX') else 0
        if is_mark > 0.5:
            print(f"\n[Position {pos}] Marker found (IS_MARK={is_mark:.2f})")
            if mark_ax > 0.5:
                print(f"  → AX marker!")
                # Check IMM opcode flag
                op_imm = x[0, pos, BD.OP_IMM].item()
                mark_pc = x[0, pos, BD.MARK_PC].item() if hasattr(BD, 'MARK_PC') else 0
                is_byte = x[0, pos, BD.IS_BYTE].item()
                print(f"  OP_IMM: {op_imm:.2f}")
                print(f"  MARK_PC: {mark_pc:.2f}")
                print(f"  IS_BYTE: {is_byte:.2f}")

                # Check FETCH values (all nibbles)
                if hasattr(BD, 'FETCH_LO'):
                    fetch_lo_nibbles = [x[0, pos, BD.FETCH_LO + k].item() for k in range(16)]
                    fetch_hi_nibbles = [x[0, pos, BD.FETCH_HI + k].item() for k in range(16)]
                    print(f"  FETCH_LO nibbles: {[f'{v:.1f}' for k,v in enumerate(fetch_lo_nibbles) if v > 0.5]}")
                    print(f"  FETCH_HI nibbles: {[f'{k}:{v:.1f}' for k,v in enumerate(fetch_hi_nibbles) if v > 0.5]}")
                    # Decode value
                    lo_nibble = next((k for k,v in enumerate(fetch_lo_nibbles) if v > 0.5), None)
                    hi_nibble = next((k for k,v in enumerate(fetch_hi_nibbles) if v > 0.5), None)
                    if lo_nibble is not None and hi_nibble is not None:
                        value = (hi_nibble << 4) | lo_nibble
                        print(f"  → FETCH value: 0x{value:02x} ({value})")
                else:
                    print(f"  FETCH dims not available")

                # Calculate expected activation for unit 0 (IMM LO nibble 0)
                # W_up: OP_IMM=S, MARK_AX=S, MARK_PC=-6S, IS_BYTE=-S
                # b_up: -S*T where T=0.5
                # activation = op_imm*S + mark_ax*S - mark_pc*6S - is_byte*S - S*0.5
                S = 100.0
                T = 0.5
                expected_up = op_imm*S + mark_ax*S - mark_pc*6*S - is_byte*S - S*T
                print(f"  Expected W_up activation: {expected_up:.2f}")
                if expected_up > 0:
                    print(f"  → Should FIRE ✓")
                else:
                    print(f"  → Should NOT fire ✗")

    # Run original
    result = original_l6_ffn_forward(x)

    # Check output at AX marker
    for pos in range(min(10, S_len)):
        is_mark = x[0, pos, BD.IS_MARK].item()
        mark_ax = x[0, pos, BD.MARK_AX].item() if hasattr(BD, 'MARK_AX') else 0
        if is_mark > 0.5 and mark_ax > 0.5:
            output_lo_0 = result[0, pos, BD.OUTPUT_LO].item()
            output_hi_0 = result[0, pos, BD.OUTPUT_HI].item()
            print(f"\n  After L6 FFN:")
            print(f"    OUTPUT_LO[0]: {output_lo_0:.2f}")
            print(f"    OUTPUT_HI[0]: {output_hi_0:.2f}")

    return result

model.blocks[6].ffn.forward = track_l6_ffn

# Generate just the first step
print("\nGenerating first VM step...\n")
try:
    # Initialize context with CODE_START and REG_PC
    ctx = [Token.CODE_START, Token.REG_PC]

    # Generate rest of first step (should be 33 more tokens to STEP_END)
    for i in range(35):
        tok = runner.model.generate_next(ctx)
        ctx.append(tok)
        if tok == Token.STEP_END:
            break

    print(f"\nFirst step completed ({len(ctx)} tokens)")
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
