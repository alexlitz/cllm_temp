# Session Status - LEA Fix Investigation

## Summary

**Progress**: 9/11 tests passing (81.8%)
- ✅ All JMP tests fixed and passing (3/3)
- ✅ All IMM tests passing (4/4)
- ❌ Both LEA tests still failing (0/2)

---

## Fixed Issues

### 1. JMP Regression (FIXED ✅)
**Issue**: JMP 16 predicting 0 instead of 16
**Root Cause**: L5 Head 3 Q[0] weights caused softmax spreading
**Fix**: Removed lines 2325-2326 in vm_step.py (Q[0] gate logic)
**Result**: Both JMP 8 and JMP 16 now pass perfectly

### 2. L10 Carry Override Investigation (IDENTIFIED, PARTIALLY FIXED)
**Issue**: L10 carry override activating at marker positions
**Root Cause**: CARRY[1] = 17394.5 (should be ~1) causing spurious activation
**Investigation**:
- CARRY[1] abnormally high due to accumulated activations in L9
- CARRY[0] also abnormally high (34.1 instead of ~1) from L8
- AX_CARRY_LO not perfectly one-hot (has residual values)
- Multiple carry detection units activating when they shouldn't

**Attempted Fix**: Reduced W_up weight for CARRY[1] from S to 0.001, reduced bias from -S*2.5 to -S*1.9
**Result**: Carry override no longer activates at markers (✅), but LEA still broken for different reason

---

## Current Issue: LEA Predicting 257

**Test Results**:
- LEA 8: Expected 8, got 257
- LEA 16: Expected 16, got 257

**Investigation**:
Layer-by-layer OUTPUT trace for LEA 8 at marker position:
```
L0-L2: OUTPUT = 0
L3:    OUTPUT = 16  (lo=0, hi=1) ← First change
L6:    OUTPUT = 105 (lo=9, hi=6) ← WRONG! Should be 8
L8:    OUTPUT = 104 (lo=8, hi=6) ← lo corrected, hi still wrong
L9:    OUTPUT = 104 (propagated)
L10:   OUTPUT = 0   (cleared by something)
```

**Key Finding**: L6 produces OUTPUT=105 (lo=9, hi=6) when it should be 8 (lo=8, hi=0).

**Possible Causes**:
1. L6 "AX passthrough" logic for JSR/ENT/LEV activating spuriously
2. TEMP dimensions have residual values triggering wrong units
3. L6 immediate write logic interfering with LEA

---

## Investigation Needed

### Priority 1: L6 OUTPUT Corruption
- **Question**: Why does L6 write OUTPUT=105 for LEA 8?
- **Location**: `_set_layer6_routing_ffn` in vm_step.py
- **Suspects**:
  - Lines 5060-5074: "Write AX_CARRY_LO/HI (jump target from L5 Head 3 first-step fetch)"
  - Lines 5076-5090: "JSR AX passthrough"
  - Lines 5137-5151: "ENT AX passthrough"
  - Lines 5153-5167: "LEV AX passthrough"

### Priority 2: CARRY Normalization
- L8/L9 carry detection writing huge unnormalized values
- Should investigate why carry units activate for (0, 8) when sum < 16
- May need better one-hot enforcement for ALU_LO/AX_CARRY_LO

---

## Files Modified

- `neural_vm/vm_step.py`:
  - Lines 2320-2328: Removed L5 Q[0] gate logic (JMP fix) ✅
  - Lines 2074-2078: AX byte defaults previously removed (earlier session)

## Debug Scripts Created

- `debug_lea8_l10_ffn.py` - L10 FFN unit activations
- `debug_lea8_all_layers.py` - OUTPUT trace through all layers
- `debug_lea8_l10_detailed.py` - L10 attention analysis
- `debug_lea8_opcodes_l10.py` - Opcode flags at L10
- `debug_lea8_l10_precise.py` - Precise L10 INPUT/OUTPUT trace
- `debug_lea8_carry_override.py` - Carry override activation check
- `debug_lea8_l9_carry.py` - L9 CARRY[1] write debug
- `debug_lea8_l8_carry.py` - L8 CARRY[0] debug
- `debug_lea8_onehot.py` - Check ALU_LO/AX_CARRY_LO one-hot encoding
- `debug_lea8_l9_output.py` - OUTPUT trace L0-L10
- `test_lea_fix.py` - Quick LEA test

---

## Next Steps

1. Debug why L6 writes OUTPUT=105 instead of 8
2. Check which L6 FFN units are activating for LEA
3. Verify TEMP dimensions are clear for LEA (no JSR/ENT/LEV flags)
4. Fix L6 logic to handle LEA correctly
5. Retest comprehensive suite

---

## Token Budget

~84k tokens used so far in this session.
Focus on efficient debugging and targeted fixes.
