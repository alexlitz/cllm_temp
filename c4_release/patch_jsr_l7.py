#!/usr/bin/env python3
"""Atomically patch JSR SP from L6 hardcoded to L7 computed, and clear handlers.

Designed to be idempotent: running multiple times produces the same result.
Run: chmod u+w neural_vm/vm_step.py neural_vm/run_vm.py && python3 patch_jsr_l7.py
"""

import os
import re
import sys


def ensure_writable(path):
    if not os.access(path, os.W_OK):
        os.chmod(path, os.stat(path).st_mode | 0o200)


def patch_vm_step(path):
    ensure_writable(path)
    with open(path, 'r') as f:
        content = f.read()
    original = content

    # 1. Replace L6 hardcoded JSR SP with empty reserved units
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
    content, count = re.subn(pattern, replacement, content, count=1, flags=re.DOTALL)
    if count == 0:
        print("  [1/5] L6 JSR SP: already patched or pattern not found")
    else:
        print("  [1/5] L6 JSR SP: replaced with reserved units")

    # 2. Add JSR SP-=8 in L7, right before "PSH: STACK0 = AX"
    l7_marker = '    # === JSR: SP -= 8 (computed from actual SP carry in L7) ==='
    if l7_marker in content:
        # Remove duplicates, keep first
        dup_pattern = r'    # === JSR: SP -= 8 \(computed from actual SP carry in L7\) ===\n.*?# Byte 3: identity carry \(unchanged for SP=0x10000\)\n'
        dups = re.findall(dup_pattern, content, re.DOTALL)
        if len(dups) > 1:
            content = re.sub(dup_pattern, '', content, count=len(dups) - 1, flags=re.DOTALL)
            print(f"  [2/5] L7 JSR SP: removed {len(dups)-1} duplicates")
        else:
            print("  [2/5] L7 JSR SP: already present")
    else:
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
        content, count2 = re.subn(l7_pattern, l7_code + r'\1', content, count=1)
        if count2 > 0:
            print("  [2/5] L7 JSR SP: inserted computed SP-=8 code")
        else:
            print("  [2/5] L7 JSR SP: insertion point not found")

    # 3. Fix L7 patch loop: skip units with positive MARK_PC
    fix_marker = 'Skip units that legitimately fire at PC marker'
    if fix_marker in content:
        print("  [3/5] L7 MARK_PC skip: already present")
    else:
        fix_pattern = r'        if has_strong_opcode and not has_mark_pc_suppression and not is_byte_unit:'
        fix_replacement = (
            '        # Skip units that legitimately fire at PC marker (JSR PC override, etc.)\n'
            '        # These have positive MARK_PC weight set by _set_function_call_weights.\n'
            '        if ffn7.W_up[u, BD.MARK_PC].item() > 10:\n'
            '            continue\n'
            '\n'
            '        if has_strong_opcode and not has_mark_pc_suppression and not is_byte_unit:'
        )
        content, count = re.subn(fix_pattern, fix_replacement, content, count=1)
        if count > 0:
            print("  [3/5] L7 MARK_PC skip: inserted")
        else:
            print("  [3/5] L7 MARK_PC skip: pattern not found")

    # 4. Add HybridALUBlock import
    if 'from .hybrid_alu import HybridALUBlock' in content:
        print("  [4/5] HybridALUBlock import: already present")
    else:
        content = content.replace(
            'from .efficient_alu_neural import (\n'
            '    ALUAndOrXor,\n'
            '    ALUMul,\n'
            '    ALUDivMod,\n'
            ')',
            'from .efficient_alu_neural import (\n'
            '    ALUAddSub,\n'
            '    ALUAndOrXor,\n'
            '    ALUMul,\n'
            '    ALUShift,\n'
            '    ALUDivMod,\n'
            ')\n'
            'from .hybrid_alu import HybridALUBlock',
        )
        if 'from .hybrid_alu import HybridALUBlock' in content:
            print("  [4/5] HybridALUBlock import: added")
        else:
            print("  [4/5] HybridALUBlock import: insertion point not found")

    # 5. Wire efficient ALU hybrid into lookup path
    if 'EFFICIENT ALU HYBRID OVERRIDE' in content:
        print("  [5/5] Efficient ALU wiring: already present")
    else:
        hybrid_block = '''

        # ===== EFFICIENT ALU HYBRID OVERRIDE =====
        # Wire efficient structural ALU on top of lookup FFN weights.
        # The efficient module replaces OUTPUT for its opcodes (ADD/SUB/etc.)
        # while the lookup FFN handles LEA/ADJ/ENT/CMP/passthrough/etc.
        model.blocks[8].ffn = HybridALUBlock(model.blocks[8].ffn, ALUAddSub(S, BD))
        model.blocks[9].ffn = HybridALUBlock(model.blocks[9].ffn, ALUAddSub(S, BD))
        model.blocks[10].ffn = HybridALUBlock(model.blocks[10].ffn, ALUAndOrXor(S, BD))
        model.blocks[11].ffn = HybridALUBlock(model.blocks[11].ffn, ALUMul(S, BD))
        model.blocks[12].ffn = HybridALUBlock(model.blocks[12].ffn, ALUMul(S, BD))
        model.blocks[13].ffn = HybridALUBlock(model.blocks[13].ffn, ALUShift(S, BD))
        model.blocks[10].post_ops.append(ALUDivMod(S, BD))
'''
        anchor = (
            "    elif alu_mode == 'efficient':\n"
            "        # Use efficient multi-layer ALU with pure neural format conversion\n"
            "        # All operations use baked FFN weights - no Python loops in forward pass"
        )
        if anchor in content:
            content = content.replace(anchor, hybrid_block + anchor)
            print("  [5/5] Efficient ALU wiring: inserted")
        else:
            print("  [5/5] Efficient ALU wiring: anchor not found")

    if content != original:
        with open(path, 'w') as f:
            f.write(content)
    else:
        print("  (no changes needed)")


def patch_run_vm(path):
    ensure_writable(path)
    with open(path, 'r') as f:
        content = f.read()
    original = content

    content = re.sub(
        r'self\._func_call_handlers\s*=\s*\{[^}]*\}',
        'self._func_call_handlers = {}',
        content, count=1, flags=re.DOTALL,
    )

    if content != original:
        with open(path, 'w') as f:
            f.write(content)
        print("  run_vm.py: cleared func_call_handlers")
    else:
        print("  run_vm.py: handlers already empty")


def verify():
    errors = []
    with open('neural_vm/vm_step.py') as f:
        vm = f.read()
    if 'EFFICIENT ALU HYBRID OVERRIDE' not in vm:
        errors.append("vm_step.py: missing hybrid override")
    if 'from .hybrid_alu import HybridALUBlock' not in vm:
        errors.append("vm_step.py: missing HybridALUBlock import")
    if 'Skip units that legitimately fire at PC marker' not in vm:
        errors.append("vm_step.py: missing MARK_PC skip fix")

    with open('neural_vm/run_vm.py') as f:
        run = f.read()
    handlers_match = re.search(r'self\._func_call_handlers\s*=\s*(\{[^}]*\})', run)
    if handlers_match and handlers_match.group(1).strip() != '{}':
        errors.append(f"run_vm.py: handlers not empty: {handlers_match.group(1)[:80]}")

    if errors:
        print("\n  VERIFICATION FAILED:")
        for e in errors:
            print(f"    - {e}")
        return False
    print("\n  Verification: OK")
    return True


if __name__ == '__main__':
    print("Patching neural VM files...")
    patch_vm_step('neural_vm/vm_step.py')
    patch_run_vm('neural_vm/run_vm.py')
    ok = verify()
    sys.exit(0 if ok else 1)
