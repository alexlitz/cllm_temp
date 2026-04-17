#!/usr/bin/env python3
"""Rewrite L4 FFN to compute PC for opcode and PC+1 for immediate."""
import sys
sys.path.insert(0, '.')

with open('neural_vm/vm_step.py', 'r') as f:
    lines = f.readlines()

# Find L4 FFN function
start_line = None
end_line = None
for i, line in enumerate(lines):
    if 'def _set_layer4_ffn(ffn, S, BD):' in line:
        start_line = i
    if start_line is not None and i > start_line and line.startswith('def '):
        end_line = i
        break

print(f"Found L4 FFN at lines {start_line+1} to {end_line}")

# New L4 FFN implementation
new_ffn = '''def _set_layer4_ffn(ffn, S, BD):
    """Layer 4 FFN: Compute PC+1 for immediate fetch.

    PC value is in EMBED_LO/HI at AX marker (from L4 attention relay).
    L5 fetch uses:
      - opcode query at address PC (EMBED_LO/HI) — opcode is at instruction start
      - immediate query at address PC+1 (TEMP[0..31]) — immediate follows opcode

    Strategy:
      EMBED stays as PC (for opcode fetch)
      TEMP gets PC+1 (for immediate fetch)
    """
    unit = 0

    # === Compute PC+1 and write to TEMP ===
    # Increment low nibble: k → (k + 1) % 16
    for k in range(16):
        new_k = (k + 1) % 16
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.b_up[unit] = -S * 0.5
        ffn.W_gate[unit, BD.EMBED_LO + k] = 1.0
        ffn.W_down[BD.TEMP + new_k, unit] = 2.0 / S
        unit += 1

    # High nibble: default copy (no carry)
    for k in range(16):
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.b_up[unit] = -S * 0.5
        ffn.W_gate[unit, BD.EMBED_HI + k] = 1.0
        ffn.W_down[BD.TEMP + 16 + k, unit] = 2.0 / S
        unit += 1

    # Carry: when low nibble was 15 (wraps to 0 after +1), increment high nibble
    for k in range(16):
        new_k = (k + 1) % 16
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.EMBED_LO + 15] = S  # Check if low nibble wrapped
        ffn.b_up[unit] = -S * 1.5
        ffn.W_gate[unit, BD.EMBED_HI + k] = 1.0
        # Cancel the default copy
        ffn.W_down[BD.TEMP + 16 + k, unit] = -2.0 / S
        # Write incremented value
        ffn.W_down[BD.TEMP + 16 + new_k, unit] = 2.0 / S
        unit += 1


'''

# Replace
new_lines = lines[:start_line] + [new_ffn] + lines[end_line:]

with open('neural_vm/vm_step.py', 'w') as f:
    f.writelines(new_lines)

print("Rewrote L4 FFN: PC for opcode (EMBED), PC+1 for immediate (TEMP)")
