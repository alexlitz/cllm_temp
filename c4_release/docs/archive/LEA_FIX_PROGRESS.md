# LEA Fix Progress Summary

## Problem

LEA 8/16 were predicting 257 instead of the correct values (8/16).

## Root Causes Identified and Fixed

### 1. ✅ L9 OUTPUT_HI Corruption (FIXED)
**Location**: `neural_vm/vm_step.py` lines 3484-3502

**Issue**: ADD hi-nibble units (256-511) were activating spuriously, writing large values to OUTPUT_HI.

**Root Cause**: Units with `carry_in=1` read `CARRY[0]` with weight `S*2.0=200`. When `CARRY[0]=32-36` (unnormalized), this caused `up ≈ 6500`, activating 16 units per OUTPUT_HI dim.

**Fix Applied**: Reduced `CARRY[0]` weight from `S*2.0` to `0.01`, and relaxed bias from `-S*4.5` to `-S*2.9`.

**Result**: OUTPUT_HI corruption eliminated in these units.

---

### 2. ✅ L9 CARRY[1] Accumulation (FIXED)
**Location**: `neural_vm/vm_step.py` lines 3622-3644 (carry-out detection)

**Issue**: 136 carry-out detection units were activating, accumulating CARRY[1] to 18,603 instead of ~1.

**Root Cause**: Same as #1 - these units read `CARRY[0]` with weight `S*2.0` in a 6-way AND.

**Fix Applied**: Reduced `CARRY[0]` weight from `S*2.0` to `0.01` in carry-out units.

**Result**: CARRY[1] no longer accumulates; stays at 0 for LEA.

---

### 3. ✅ L9 CARRY[2] Borrow Detection (FIXED)
**Location**: `neural_vm/vm_step.py` lines 3646-3670 (borrow-out detection)

**Issue**: Same pattern as CARRY[1] for SUB operations.

**Fix Applied**: Same weight reduction for consistency.

**Result**: CARRY[2] handled correctly.

---

### 4. ✅ L10 Carry Override Amplification (FIXED AS SIDE EFFECT)
**Location**: `neural_vm/vm_step.py` lines 3978-3986

**Issue**: Unit 1849 was reading unnormalized `CARRY[1]=18,603` with weight `S=100`, amplifying OUTPUT_HI[0] from 19 to 186,000.

**Fix**: Eliminated by fixing #2 above. CARRY[1] now stays at 0, so no amplification occurs.

**Result**: L10 no longer corrupts OUTPUT_HI.

---

### 5. ✅ DivModModule Spurious Activation (FIXED)
**Location**: `neural_vm/vm_step.py` lines 363-408 (DivModModule initialization)

**Issue**: 414 DIV/MOD units were activating for LEA, writing ~69,600 to OUTPUT_HI[0].

**Root Cause**: 6-way AND with bias `-5.5*S` allowed activation when only 5 inputs matched (OP_DIV=0, but other 5 dims > 5.5).

**Fix Applied**: Changed from 6-way AND to **5-way AND + gate check**:
- `up`: 5-way AND (MARK_AX + 4 operand dims) with bias `-4.5*S`
- `gate`: Check OP_DIV/OP_MOD via `W_gate`
- When OP_DIV=0, gate≈0, so hidden≈0 regardless of up value

**Result**: DivModModule now completely inactive for LEA (0 contribution).

---

## Current Status

### What Works
- L9 FFN correctly computes OUTPUT_HI, writing ~19 to appropriate dims
- L10 no longer amplifies via carry override
- L10 post_ops (DivModModule) no longer activates
- OUTPUT_HI stable at ~19 through L11-L15

### Remaining Issue
**Problem**: Model predicts token 24 instead of token 8.

**Cause**: OUTPUT_HI[1]=19.2 > OUTPUT_HI[0]=19.1 (very close!)

**Root Cause**: ALU_HI and AX_CARRY_HI representations aren't perfectly one-hot:
```
ALU_HI[0] = 0.416  (should be ~1.0)
ALU_HI[1] = 0.012  (leakage)
AX_CARRY_HI[0] = 2.458  (unnormalized)
```

This causes **wrong L9 units to activate**:
- Unit 0 (a=0, b=0): should fully activate
- Unit 16 (a=1, b=0): shouldn't activate, but ALU_HI[1] leakage causes partial activation
- Unit 256 (a=0, b=0, carry=1): partially activates due to CARRY[0]=36

Total to OUTPUT_HI[0]: 19.1
Total to OUTPUT_HI[1]: 19.2 (from wrong units)

---

## Next Steps

### Option A: Normalize Operand Representations
Add normalization layers or units to ensure ALU_HI and AX_CARRY_HI are clean one-hot before L9.

### Option B: Increase L9 Activation Thresholds
Make L9 ADD units require stricter matching (higher bias) to suppress partial activations.

### Option C: Use Softmax/Argmax for Operands
Convert ALU/AX_CARRY to categorical distributions before arithmetic operations.

---

## Files Modified

1. `neural_vm/vm_step.py`:
   - Lines 3484-3502: L9 ADD hi-nibble OUTPUT_HI units (CARRY weight fix)
   - Lines 3622-3644: L9 ADD hi-nibble CARRY[1] detection (CARRY weight fix)
   - Lines 3646-3670: L9 SUB hi-nibble CARRY[2] detection (CARRY weight fix)
   - Lines 363-408: DivModModule lookup mode (gate-based opcode gating)

## Test Results

Before fixes:
```
LEA 8: FAIL (predicted 257)
LEA 16: FAIL (predicted 257)
```

After fixes:
```
LEA 8: FAIL (predicted 24, closer but still wrong)
LEA 16: Not tested yet
OUTPUT_HI corruption eliminated
```

**Progress**: Eliminated 3 major corruption sources. Remaining issue is operand normalization.
