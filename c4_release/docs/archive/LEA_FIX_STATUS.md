# LEA Fix Status - Final Investigation

## Current State

**Test Results**: 9/11 passing (81.8%)
- ✅ JMP 8, 16, 32: ALL PASS
- ✅ IMM 0, 1, 42, 255: ALL PASS
- ❌ LEA 8, 16: FAIL (predicting 257 instead of expected values)

---

## Root Cause Found

### L9 OUTPUT_HI Corruption

**The Bug**: L9 FFN writes the value **10257** to ALL OUTPUT_HI dimensions (making them equal).

**Evidence**:
```
L0-L8: OUTPUT_HI ≈ 0 (all dims equal, normal initialization)
L9:    OUTPUT_HI ≈ 10257 (all dims equal) ← CORRUPTION HERE!
L10:   OUTPUT_HI ≈ 10257 (propagated)
```

**Impact**:
- When all OUTPUT_HI dims are equal, `argmax` picks dimension 0 by default
- This makes the predicted token wrong (257 instead of 8)
- The final head projection gets distorted logits

---

## Investigation Timeline

### Phase 1: L10 Carry Override (FIXED ✅)
- **Issue**: Carry override activating at marker (changing OUTPUT 8→1)
- **Cause**: CARRY[1]=17394 (unnormalized) making `up` positive even when IS_BYTE=0
- **Fix**: Reduced W_up weight for CARRY[1] from S to 0.001, reduced bias to -S*1.9
- **Result**: Carry override no longer activates at markers

### Phase 2: L10 AX Passthrough (NOT THE ISSUE)
- Checked if AX passthrough was clearing OUTPUT
- Found: up=-444.2 (correctly suppressed by OP_LEA)
- AX passthrough NOT activating

### Phase 3: L10 ALU Clearing (ISSUE IDENTIFIED)
- **Discovery**: OP_LEA was added to `non_alu_opcodes` list (line 3679)
- This clears ALU_LO/HI for LEA
- **BUT**: ALU clearing only affects ALU dims, NOT OUTPUT dims
- So this doesn't explain OUTPUT_HI=10257

### Phase 4: OUTPUT_HI Corruption (ROOT CAUSE)
- **Finding**: L9 sets ALL OUTPUT_HI dimensions to 10257
- This is the real bug causing LEA failure
- Need to investigate L9 FFN (_set_layer9_alu) to find which units are doing this

---

## Code Changes Made

### vm_step.py Line 4044-4053: L10 Carry Override Fix
```python
# Before:
ffn.W_up[unit, BD.CARRY + 1] = S
ffn.W_up[unit, BD.IS_BYTE] = S
ffn.W_up[unit, BD.BYTE_INDEX_0] = S
ffn.b_up[unit] = -S * 2.5

# After:
ffn.W_up[unit, BD.CARRY + 1] = 0.001  # Reduced to prevent spurious activation
ffn.W_up[unit, BD.IS_BYTE] = S
ffn.W_up[unit, BD.BYTE_INDEX_0] = S
ffn.b_up[unit] = -S * 1.9  # Reduced bias
```

### External Changes (Not by me):
- Line 3679: OP_LEA added to non_alu_opcodes
- Lines 1940-1956: "FIRST-STEP AX BYTE 0" logic added

---

## Next Steps

### Priority 1: Fix L9 OUTPUT_HI Corruption
**Task**: Find which L9 FFN units write 10257 to all OUTPUT_HI dims

**Approach**:
1. Check _set_layer9_alu function
2. Look for logic that writes to OUTPUT_HI with large weights
3. Identify why all 16 dims get the same value
4. Fix the weights or logic to write correct values

**Possible Causes**:
- SUB borrow detection writing to all OUTPUT_HI dims
- Comparison logic incorrectly activating for LEA
- Some unit with W_down weights that affect all OUTPUT_HI dims

### Priority 2: Verify OP_LEA in non_alu_opcodes
**Question**: Should OP_LEA be in the ALU clearing list?

**Analysis**:
- LEA computes: AX = imm + BP
- This IS an ALU operation (uses ADD circuit in L8/L9)
- So LEA SHOULD have ALU operands gathered
- Adding it to non_alu_opcodes would prevent operand gather
- **Recommendation**: REMOVE OP_LEA from non_alu_opcodes (line 3679)

---

## Debug Scripts Created

1. `debug_lea8_l10_ffn.py` - L10 FFN unit activations
2. `debug_lea8_all_layers.py` - OUTPUT trace through layers
3. `debug_lea8_carry_override.py` - Carry override check
4. `debug_lea8_l9_carry.py` - L9 CARRY[1] debug
5. `debug_lea8_l8_carry.py` - L8 CARRY[0] debug
6. `debug_lea8_onehot.py` - One-hot encoding check
7. `debug_lea8_l9_output.py` - Layer OUTPUT trace
8. `debug_lea8_l6_units.py` - L6 FFN unit analysis
9. `debug_lea8_l6_attn.py` - L6 attention vs FFN
10. `debug_lea8_l10_ax_passthrough.py` - AX passthrough check
11. `debug_lea8_l10_negative.py` - L10 OUTPUT contributions
12. `debug_lea8_output_hi_trace.py` - OUTPUT_HI corruption trace
13. `test_lea_fix.py` - Quick LEA test

---

## Token Budget

~100k tokens used so far.
Major progress on understanding the issue.
Root cause identified: L9 OUTPUT_HI corruption.

---

## Summary

✅ **Fixed**: L10 carry override no longer activates spuriously
✅ **Identified**: L9 writes 10257 to all OUTPUT_HI dimensions
⏳ **Next**: Investigate L9 FFN to fix OUTPUT_HI corruption
❓ **Question**: Should OP_LEA be removed from non_alu_opcodes?
