#!/usr/bin/env python3
"""Add PC increment to L6 FFN."""

with open('neural_vm/vm_step.py', 'r') as f:
    lines = f.readlines()

# Find the line with "# === IMM: FETCH → OUTPUT ===" in _set_layer6_routing_ffn
insert_line = None
for i, line in enumerate(lines):
    if '# === IMM: FETCH → OUTPUT ===' in line:
        # Make sure we're in the right function
        if any('def _set_layer6_routing_ffn' in lines[j] for j in range(max(0, i-30), i)):
            insert_line = i
            break

if insert_line is None:
    print("Could not find insertion point!")
    sys.exit(1)

print(f"Inserting PC increment before line {insert_line+1}")

# PC increment code
pc_inc_code = '''    # === PC INCREMENT FOR ALL STEPS: add 8 to OUTPUT ===
    # PC increment happens AFTER instruction execution, so it applies to all steps.
    # L3 handles first-step default (PC=0) and passthrough for subsequent steps.
    # L6 increments PC by 8 for the next instruction (8 bytes per instruction).
    T_pc_inc = 0.5

    # Low nibble increment with wrap: (k+8)%16
    for k in range(16):
        new_k = (k + 8) % 16
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.b_up[unit] = -S * T_pc_inc
        ffn.W_gate[unit, BD.OUTPUT_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + new_k, unit] = 2.0 / S
        ffn.W_down[BD.OUTPUT_LO + k, unit] = -2.0 / S  # cancel original
        unit += 1

    # High nibble carry: increment if old_lo >= 8 (when k+8 >= 16)
    T_pc_inc_hi = 1.5
    for k in range(16):
        new_k = (k + 1) % 16
        ffn.W_up[unit, BD.MARK_PC] = S
        # Fire only if OUTPUT_LO in [8, 9, 10, 11, 12, 13, 14, 15]
        for lo_bit in range(8, 16):
            ffn.W_up[unit, BD.OUTPUT_LO + lo_bit] = S
        ffn.b_up[unit] = -S * T_pc_inc_hi
        # Read current high nibble
        ffn.W_gate[unit, BD.OUTPUT_HI + k] = 1.0
        # Write to incremented position
        ffn.W_down[BD.OUTPUT_HI + new_k, unit] = 2.0 / S
        ffn.W_down[BD.OUTPUT_HI + k, unit] = -2.0 / S  # cancel original
        unit += 1

'''

# Insert the code
lines.insert(insert_line, pc_inc_code)

with open('neural_vm/vm_step.py', 'w') as f:
    f.writelines(lines)

print("Added PC increment (+8) to L6 FFN")
