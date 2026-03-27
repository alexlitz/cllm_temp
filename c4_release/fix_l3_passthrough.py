#!/usr/bin/env python3
"""Rewrite L3 FFN to use passthrough (no PC increment)."""
import sys
sys.path.insert(0, '.')

with open('neural_vm/vm_step.py', 'r') as f:
    lines = f.readlines()

# Find L3 FFN function
start_line = None
end_line = None
for i, line in enumerate(lines):
    if 'def _set_layer3_ffn(ffn, S, BD):' in line:
        start_line = i
    if start_line is not None and i > start_line and line.startswith('def '):
        end_line = i
        break

print(f"Found L3 FFN at lines {start_line+1} to {end_line}")

# New L3 FFN with PC=0 default and passthrough
new_ffn = '''def _set_layer3_ffn(ffn, S, BD):
    """Layer 3 FFN: PC first-step default + passthrough.

    First step (HAS_SE=0): PC = 0 (DraftVM formula: idx * 8 = 0 * 8 = 0)
    Subsequent steps (HAS_SE=1): PC passthrough from EMBED (no increment here)

    NOTE: PC increment happens in L6 FFN for ALL steps (after instruction execution).
    L3 only handles the first-step default and subsequent-step passthrough.
    """
    unit = 0

    # PC FIRST-STEP DEFAULT: when MARK_PC AND NOT HAS_SE, set PC=0
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.b_up[unit] = -S * 0.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_LO + 0, unit] = 2.0 / S  # PC=0 (lo nibble)
    unit += 1
    # Unit B: undo when HAS_SE (subsequent steps use passthrough from EMBED)
    ffn.W_up[unit, BD.HAS_SE] = S
    ffn.b_up[unit] = -S * 0.5
    ffn.W_gate[unit, BD.MARK_PC] = 1.0
    ffn.W_down[BD.OUTPUT_LO + 0, unit] = -2.0 / S
    unit += 1

    # Same for OUTPUT_HI[0]
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.b_up[unit] = -S * 0.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_HI + 0, unit] = 2.0 / S  # PC=0 (hi nibble=0)
    unit += 1
    ffn.W_up[unit, BD.HAS_SE] = S
    ffn.b_up[unit] = -S * 0.5
    ffn.W_gate[unit, BD.MARK_PC] = 1.0
    ffn.W_down[BD.OUTPUT_HI + 0, unit] = -2.0 / S
    unit += 1

    # PC PASSTHROUGH for subsequent steps: when MARK_PC AND HAS_SE, copy EMBED to OUTPUT
    # (no increment - that happens in L6 after instruction execution)
    for k in range(16):
        ffn.W_up[unit, BD.HAS_SE] = S
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.b_up[unit] = -S * 1.5
        ffn.W_gate[unit, BD.EMBED_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1

    # Hi nibble passthrough (only at MARK_PC AND HAS_SE)
    for k in range(16):
        ffn.W_up[unit, BD.HAS_SE] = S
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.b_up[unit] = -S * 1.5
        ffn.W_gate[unit, BD.EMBED_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1


'''

# Replace
new_lines = lines[:start_line] + [new_ffn] + lines[end_line:]

with open('neural_vm/vm_step.py', 'w') as f:
    f.writelines(new_lines)

print("Rewrote L3 FFN: PC=0 default, passthrough for subsequent steps")
