#!/usr/bin/env python3
"""Atomically patch JSR SP from L6 hardcoded to L7 computed, and clear handlers."""

import re, sys

def patch_vm_step(path):
    with open(path, 'r') as f:
        content = f.read()

    # 1. Replace L6 hardcoded JSR SP with empty reserved units
    # Find the section between "JSR SP -= 8 (autoregressive" and "JSR STACK0 = return_addr"
    pattern = (
        r'(    # --- JSR SP -= 8 \(autoregressive shift fix\) ---\n)'
        r'.*?'
        r'(    # --- JSR STACK0 = return_addr)'
    )
    replacement = (
        r'\1'
        r'    # Moved to L7 where EMBED has byte values at marker positions.\n'
        r'    # Reserve 128 units (was: 4 bytes × 32 units each of hardcoded output).\n'
        r'    unit += 128\n\n'
        r'    \2'
    )
    new_content, count = re.subn(pattern, replacement, content, count=1, flags=re.DOTALL)
    if count == 0:
        print("WARNING: L6 JSR SP pattern not found (may already be patched)")
        new_content = content
    
    # 2. Add JSR SP-=8 in L7, right before "PSH: STACK0 = AX"
    l7_pattern = r'(    # === PSH: STACK0 = AX \(override STACK0 carry with AX value\) ===)'
    l7_code = '''    # === JSR: SP -= 8 (computed from actual SP carry in L7) ===
    # CMP[4] (JSR flag) relayed by L6 attn head 6. EMBED has SP bytes in L7.
    T_jsr_sp = 1.5
    for k in range(16):
        new_k = (k - 8) % 16
        ffn.W_up[unit, BD.CMP + 4] = S
        ffn.W_up[unit, BD.MARK_SP] = S
        ffn.b_up[unit] = -S * T_jsr_sp
        ffn.W_gate[unit, BD.EMBED_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + new_k, unit] = 2.0 / S
        ffn.W_down[BD.OUTPUT_LO + k, unit] += -2.0 / S
        unit += 1
    for k in range(16):
        new_k_borrow = (k - 1) % 16
        ffn.W_up[unit, BD.CMP + 4] = S
        ffn.W_up[unit, BD.MARK_SP] = S
        ffn.b_up[unit] = -S * T_jsr_sp
        ffn.W_gate[unit, BD.EMBED_HI + k] = 1.0
        for lo_bit in range(8, 16):
            ffn.W_gate[unit, BD.EMBED_LO + lo_bit] = -1.0
        ffn.W_down[BD.OUTPUT_HI + new_k_borrow, unit] = 2.0 / S
        ffn.W_down[BD.OUTPUT_HI + k, unit] += -2.0 / S
        unit += 1
    # Byte 1 (BYTE_INDEX_0): always 0xFF (borrow propagates from byte 0)
    T_jsr_b1 = 3.5
    ffn.W_up[unit, BD.CMP + 4] = S
    ffn.W_up[unit, BD.BYTE_INDEX_0] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.W_up[unit, BD.H1 + 2] = S
    ffn.b_up[unit] = -S * T_jsr_b1
    ffn.W_gate[unit, BD.CONST] = 1.0
    ffn.W_down[BD.OUTPUT_LO + 15, unit] = 10.0 / S
    unit += 1
    ffn.W_up[unit, BD.CMP + 4] = S
    ffn.W_up[unit, BD.BYTE_INDEX_0] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.W_up[unit, BD.H1 + 2] = S
    ffn.b_up[unit] = -S * T_jsr_b1
    ffn.W_gate[unit, BD.CONST] = 1.0
    ffn.W_down[BD.OUTPUT_HI + 15, unit] = 10.0 / S
    unit += 1
    # Byte 2 (BYTE_INDEX_1): 0x00 (borrow from byte 1 underflow for SP=0x10000)
    ffn.W_up[unit, BD.CMP + 4] = S
    ffn.W_up[unit, BD.BYTE_INDEX_1] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.W_up[unit, BD.H1 + 2] = S
    ffn.b_up[unit] = -S * T_jsr_b1
    ffn.W_gate[unit, BD.CONST] = 1.0
    ffn.W_down[BD.OUTPUT_LO + 0, unit] = 10.0 / S
    unit += 1
    ffn.W_up[unit, BD.CMP + 4] = S
    ffn.W_up[unit, BD.BYTE_INDEX_1] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.W_up[unit, BD.H1 + 2] = S
    ffn.b_up[unit] = -S * T_jsr_b1
    ffn.W_gate[unit, BD.CONST] = 1.0
    ffn.W_down[BD.OUTPUT_HI + 0, unit] = 10.0 / S
    unit += 1
    # Byte 3: identity carry (unchanged for SP=0x10000)

    '''
    new_content2, count2 = re.subn(l7_pattern, l7_code + r'\1', new_content, count=1)
    if count2 == 0:
        print("WARNING: L7 PSH STACK0 pattern not found (may already be patched)")
    else:
        new_content = new_content2

    # Remove duplicate JSR SP blocks if any
    dup_pattern = r'    # === JSR: SP -= 8 \(computed from actual SP carry in L7\) ===\n.*?# Byte 3: identity carry \(unchanged for SP=0x10000\)\n'
    dups = re.findall(dup_pattern, new_content, re.DOTALL)
    if len(dups) > 1:
        # Keep only the last occurrence
        new_content = re.sub(dup_pattern, '', new_content, count=len(dups)-1, flags=re.DOTALL)
        print(f"Removed {len(dups)-1} duplicate JSR SP L7 blocks")

    with open(path, 'w') as f:
        f.write(new_content)
    print(f"Patched {path}: L6 SP→reserved, L7 SP→computed")

    # 3. Fix L7 patch loop: skip units with positive MARK_PC (JSR PC override)
    # The post-processing loop at ~line 1905 incorrectly overwrites MARK_PC=+100
    # to -10000 for JSR PC override units because they have OP_ENT/OP_LEV blockers.
    fix_pattern = r'        if has_strong_opcode and not has_mark_pc_suppression and not is_byte_unit:'
    fix_replacement = (
        '        # Skip units that legitimately fire at PC marker (JSR PC override, etc.)\n'
        '        # These have positive MARK_PC weight set by _set_function_call_weights.\n'
        '        if ffn7.W_up[u, BD.MARK_PC].item() > 10:\n'
        '            continue\n'
        '\n'
        '        if has_strong_opcode and not has_mark_pc_suppression and not is_byte_unit:'
    )
    with open(path, 'r') as f:
        content = f.read()
    if 'Skip units that legitimately fire at PC marker' not in content:
        new_content, count = re.subn(fix_pattern, fix_replacement, content, count=1)
        if count > 0:
            with open(path, 'w') as f:
                f.write(new_content)
            print(f"Patched {path}: L7 patch loop MARK_PC skip")

    # 4. Add HybridALU import
    with open(path, 'r') as f:
        content = f.read()
    if 'HybridALUBlock' not in content:
        content = content.replace(
            'from .efficient_alu_neural import (\n'
            '    EfficientALU_L8_L9_Neural,\n'
            '    EfficientALU_L10_Neural,\n'
            '    EfficientALU_L11_L12_Neural,\n'
            '    EfficientALU_L13_Neural,\n'
            '    EfficientDivMod_Neural,\n'
            ')',
            'from .efficient_alu_neural import (\n'
            '    EfficientALU_L8_L9_Neural,\n'
            '    EfficientALU_L10_Neural,\n'
            '    EfficientALU_L11_L12_Neural,\n'
            '    EfficientALU_L13_Neural,\n'
            '    EfficientDivMod_Neural,\n'
            ')\n'
            'from .hybrid_alu import HybridALUBlock',
        )
        with open(path, 'w') as f:
            f.write(content)
        print(f"Patched {path}: added HybridALUBlock import")

    # 5. Wire efficient ALU hybrid into lookup path
    with open(path, 'r') as f:
        content = f.read()
    if 'EFFICIENT ALU HYBRID OVERRIDE' not in content:
        hybrid_block = '''

        # ===== EFFICIENT ALU HYBRID OVERRIDE =====
        # Wire efficient structural ALU on top of lookup FFN weights.
        # The efficient module replaces OUTPUT for its opcodes (ADD/SUB/etc.)
        # while the lookup FFN handles LEA/ADJ/ENT/CMP/passthrough/etc.
        model.blocks[8].ffn = HybridALUBlock(model.blocks[8].ffn, EfficientALU_L8_L9_Neural(S, BD))
        model.blocks[9].ffn = HybridALUBlock(model.blocks[9].ffn, EfficientALU_L8_L9_Neural(S, BD))
        model.blocks[10].ffn = HybridALUBlock(model.blocks[10].ffn, EfficientALU_L10_Neural(S, BD))
        model.blocks[11].ffn = HybridALUBlock(model.blocks[11].ffn, EfficientALU_L11_L12_Neural(S, BD))
        model.blocks[12].ffn = HybridALUBlock(model.blocks[12].ffn, EfficientALU_L11_L12_Neural(S, BD))
        model.blocks[13].ffn = HybridALUBlock(model.blocks[13].ffn, EfficientALU_L13_Neural(S, BD))
        model.blocks[10].post_ops.append(EfficientDivMod_Neural(S, BD))
'''
        content = content.replace(
            '    elif alu_mode == \'efficient\':\n'
            '        # Use efficient multi-layer ALU with pure neural format conversion\n'
            '        # All operations use baked FFN weights - no Python loops in forward pass',
            hybrid_block +
            '    elif alu_mode == \'efficient\':\n'
            '        # Use efficient multi-layer ALU with pure neural format conversion\n'
            '        # All operations use baked FFN weights - no Python loops in forward pass'
        )
        with open(path, 'w') as f:
            f.write(content)
        print(f"Patched {path}: wired efficient ALU hybrid override")


def patch_run_vm(path):
    with open(path, 'r') as f:
        content = f.read()
    content = re.sub(
        r'self\._func_call_handlers\s*=\s*\{[^}]*\}',
        'self._func_call_handlers = {}',
        content, count=1, flags=re.DOTALL
    )
    with open(path, 'w') as f:
        f.write(content)
    print(f"Patched {path}: handlers = {{}}")

if __name__ == '__main__':
    patch_vm_step('neural_vm/vm_step.py')
    patch_run_vm('neural_vm/run_vm.py')
