#!/usr/bin/env python3
"""
Apply 4-unit inclusion-exclusion pattern to DIV, MOD, SHL, SHR operations.
"""

import re

def apply_4unit_pattern_div_mod(content, func_name, has_carry):
    """Apply 4-unit pattern to DIV or MOD."""
    # Find the function
    pattern = rf'(def {func_name}\(self, position: int\):.*?)'
    pattern += r'(# Lookup table with.*?self\.unit_offset \+= \d+)'

    match = re.search(pattern, content, re.DOTALL)
    if not match:
        print(f"Could not find {func_name}")
        return content

    # Build replacement based on whether it has carry output
    if has_carry:
        # DIV has result_slot and carry_slot
        replacement = '''# Lookup table with 4-unit inclusion-exclusion: 1024 units (16×16 × 4)
        for a in range(base):
            for b in range(base):
                if b == 0:
                    quotient = 0
                    remainder = a
                else:
                    quotient = a // b
                    remainder = a % b

                op_idx = self.reg_map.opcode_index(self.opcode)

                # Unit 1: +result for region [a,∞) × [b,∞)
                u1 = self.unit_offset
                self.W_up[u1, a_slot] = S
                self.b_up[u1] = -S * (a - 0.5)
                self.W_gate[u1, b_slot] = S
                self.b_gate[u1] = -S * (b - 0.5)
                if self.W_gate[u1, op_idx] == 0:
                    self.W_gate[u1, op_idx] = 1.0
                self.W_down[result_slot, u1] = quotient / S
                self.W_down[carry_slot, u1] = remainder / S

                # Unit 2: -result for region [a+1,∞) × [b,∞)
                u2 = self.unit_offset + 1
                self.W_up[u2, a_slot] = S
                self.b_up[u2] = -S * (a + 0.5)
                self.W_gate[u2, b_slot] = S
                self.b_gate[u2] = -S * (b - 0.5)
                if self.W_gate[u2, op_idx] == 0:
                    self.W_gate[u2, op_idx] = 1.0
                self.W_down[result_slot, u2] = -quotient / S
                self.W_down[carry_slot, u2] = -remainder / S

                # Unit 3: -result for region [a,∞) × [b+1,∞)
                u3 = self.unit_offset + 2
                self.W_up[u3, a_slot] = S
                self.b_up[u3] = -S * (a - 0.5)
                self.W_gate[u3, b_slot] = S
                self.b_gate[u3] = -S * (b + 0.5)
                if self.W_gate[u3, op_idx] == 0:
                    self.W_gate[u3, op_idx] = 1.0
                self.W_down[result_slot, u3] = -quotient / S
                self.W_down[carry_slot, u3] = -remainder / S

                # Unit 4: +result for region [a+1,∞) × [b+1,∞)
                u4 = self.unit_offset + 3
                self.W_up[u4, a_slot] = S
                self.b_up[u4] = -S * (a + 0.5)
                self.W_gate[u4, b_slot] = S
                self.b_gate[u4] = -S * (b + 0.5)
                if self.W_gate[u4, op_idx] == 0:
                    self.W_gate[u4, op_idx] = 1.0
                self.W_down[result_slot, u4] = quotient / S
                self.W_down[carry_slot, u4] = remainder / S

                self.unit_offset += 4'''
    else:
        # MOD has only result_slot
        replacement = '''# Lookup table with 4-unit inclusion-exclusion: 1024 units (16×16 × 4)
        for a in range(base):
            for b in range(base):
                if b == 0:
                    result = 0
                else:
                    result = a % b

                op_idx = self.reg_map.opcode_index(self.opcode)

                # Unit 1: +result for region [a,∞) × [b,∞)
                u1 = self.unit_offset
                self.W_up[u1, a_slot] = S
                self.b_up[u1] = -S * (a - 0.5)
                self.W_gate[u1, b_slot] = S
                self.b_gate[u1] = -S * (b - 0.5)
                if self.W_gate[u1, op_idx] == 0:
                    self.W_gate[u1, op_idx] = 1.0
                self.W_down[result_slot, u1] = result / S

                # Unit 2: -result for region [a+1,∞) × [b,∞)
                u2 = self.unit_offset + 1
                self.W_up[u2, a_slot] = S
                self.b_up[u2] = -S * (a + 0.5)
                self.W_gate[u2, b_slot] = S
                self.b_gate[u2] = -S * (b - 0.5)
                if self.W_gate[u2, op_idx] == 0:
                    self.W_gate[u2, op_idx] = 1.0
                self.W_down[result_slot, u2] = -result / S

                # Unit 3: -result for region [a,∞) × [b+1,∞)
                u3 = self.unit_offset + 2
                self.W_up[u3, a_slot] = S
                self.b_up[u3] = -S * (a - 0.5)
                self.W_gate[u3, b_slot] = S
                self.b_gate[u3] = -S * (b + 0.5)
                if self.W_gate[u3, op_idx] == 0:
                    self.W_gate[u3, op_idx] = 1.0
                self.W_down[result_slot, u3] = -result / S

                # Unit 4: +result for region [a+1,∞) × [b+1,∞)
                u4 = self.unit_offset + 3
                self.W_up[u4, a_slot] = S
                self.b_up[u4] = -S * (a + 0.5)
                self.W_gate[u4, b_slot] = S
                self.b_gate[u4] = -S * (b + 0.5)
                if self.W_gate[u4, op_idx] == 0:
                    self.W_gate[u4, op_idx] = 1.0
                self.W_down[result_slot, u4] = result / S

                self.unit_offset += 4'''

    # Find the for loop and replace
    func_start = match.start()
    func_body = match.group(0)

    # Find where the for loop starts
    for_match = re.search(r'# Lookup table.*?\n(        for a in range)', func_body, re.DOTALL)
    if not for_match:
        print(f"Could not find for loop in {func_name}")
        return content

    # Replace from "# Lookup table" to "self.unit_offset += X"
    before = content[:func_start + for_match.start()]
    after_match = re.search(r'self\.unit_offset \+= \d+', func_body)
    if after_match:
        after = content[func_start + after_match.end():]
        return before + replacement + after

    return content


# Read file
with open('neural_vm/nibble_weight_compiler.py', 'r') as f:
    content = f.read()

# Apply fixes
print("Fixing emit_div_nibble...")
content = apply_4unit_pattern_div_mod(content, 'emit_div_nibble', has_carry=True)

print("Fixing emit_mod_nibble...")
content = apply_4unit_pattern_div_mod(content, 'emit_mod_nibble', has_carry=False)

# For SHL and SHR, they're similar to MUL
print("Fixing emit_shl_nibble and emit_shr_nibble...")
# These need manual fixing similar to MUL

# Write back
with open('neural_vm/nibble_weight_compiler.py', 'w') as f:
    f.write(content)

print("Fixed DIV and MOD. SHL/SHR need manual fixing.")
