#!/usr/bin/env python3
"""Fix L4 FFN to compute PC-1 for opcode and PC for immediate."""
import sys
sys.path.insert(0, '.')

# Read the file
with open('neural_vm/vm_step.py', 'r') as f:
    lines = f.readlines()

# Find the L4 FFN function (starts around line 1696)
start_line = None
end_line = None
for i, line in enumerate(lines):
    if 'def _set_layer4_ffn(ffn, S, BD):' in line:
        start_line = i
    if start_line is not None and i > start_line and line.startswith('def '):
        end_line = i
        break

print(f"Found L4 FFN function at lines {start_line+1} to {end_line}")

# New L4 FFN implementation
new_ffn = '''def _set_layer4_ffn(ffn, S, BD):
    """Layer 4 FFN: Compute PC-1 for opcode fetch and PC for immediate fetch.

    PC value is in EMBED_LO/HI at AX marker (from L4 attention relay).
    L5 fetch uses:
      - opcode query at address PC-1 (EMBED_LO/HI) — opcode is one byte before imm
      - immediate query at address PC (TEMP[0..31])

    Strategy:
      1. Copy EMBED → TEMP (PC → PC for immediate address)
      2. Decrement EMBED by 1 (PC → PC-1 for opcode address)
    """
    unit = 0

    # === Step 1: Copy EMBED to TEMP (PC → TEMP for immediate fetch) ===
    # TEMP_LO = EMBED_LO
    for k in range(16):
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.b_up[unit] = -S * 0.5
        ffn.W_gate[unit, BD.EMBED_LO + k] = 1.0
        ffn.W_down[BD.TEMP + k, unit] = 2.0 / S
        unit += 1

    # TEMP_HI = EMBED_HI
    for k in range(16):
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.b_up[unit] = -S * 0.5
        ffn.W_gate[unit, BD.EMBED_HI + k] = 1.0
        ffn.W_down[BD.TEMP + 16 + k, unit] = 2.0 / S
        unit += 1

    # === Step 2: Decrement EMBED by 1 (PC → PC-1 for opcode fetch) ===
    # Decrement low nibble: k → (k - 1) % 16
    # When EMBED_LO[0]=1, we need to borrow from high nibble
    for k in range(16):
        new_k = (k - 1) % 16
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.b_up[unit] = -S * 0.5
        ffn.W_gate[unit, BD.EMBED_LO + k] = 1.0
        ffn.W_down[BD.EMBED_LO + new_k, unit] = 2.0 / S
        ffn.W_down[BD.EMBED_LO + k, unit] = -2.0 / S  # cancel original
        unit += 1

    # Decrement high nibble when low nibble was 0 (borrow condition)
    # Default: copy high nibble (no borrow)
    for k in range(16):
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.b_up[unit] = -S * 0.5
        ffn.W_gate[unit, BD.EMBED_HI + k] = 1.0
        ffn.W_down[BD.EMBED_HI + k, unit] = 2.0 / S
        unit += 1

    # Borrow: when EMBED_LO[0]=1 (original was 0), decrement high nibble
    for k in range(16):
        new_k = (k - 1) % 16
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.EMBED_LO + 15] = S  # Check if low nibble wrapped (0 → 15)
        ffn.b_up[unit] = -S * 1.5
        ffn.W_gate[unit, BD.EMBED_HI + k] = 1.0
        # Cancel the default copy
        ffn.W_down[BD.EMBED_HI + k, unit] = -2.0 / S
        # Write decremented value
        ffn.W_down[BD.EMBED_HI + new_k, unit] = 2.0 / S
        unit += 1


'''

# Replace the function
new_lines = lines[:start_line] + [new_ffn] + lines[end_line:]

# Write back
with open('neural_vm/vm_step.py', 'w') as f:
    f.writelines(new_lines)

print("Fixed L4 FFN - now computes PC-1 for opcode and PC for immediate")
