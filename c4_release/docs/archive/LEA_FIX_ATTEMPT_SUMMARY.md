# LEA Fix Attempt - Session Summary

## Objective
Fix LEA (Load Effective Address) operations which were showing 6/10 tokens correct on first-step execution.

## Attempted Solutions

### Attempt 1: Layer 9 FFN Units for Upper Bytes
**Goal:** Initialize AX bytes 1-3 to BP values (0x00, 0x01, 0x00) on first step
**Method:** Added 6 FFN units in Layer 9 with:
- Conditioning: `H1[AX] + BYTE_INDEX_K + OP_LEA - HAS_SE`
- Strong signals (50.0) to override Layer 10 passthrough (2.0)
- Cancellation of Layer 3 DEFAULT (- 10.0)

**Result:** ✗ FAILED
- LEA: Improved to 8/10 tokens
- **IMM: BROKE - Token 8 predicted 1 instead of 0**
- LEA units were firing for IMM operations

### Attempt 2: Gate-Path IMM Suppression
**Goal:** Suppress LEA units when OP_IMM is active
**Method:** Added `ffn.W_gate[unit, BD.OP_IMM] = -5.0` to all 6 units

**Result:** ✗ FAILED
- LEA: 0/10 (completely broken)
- IMM: Still broken
- Gate became too restrictive, preventing LEA from working

### Attempt 3: Up-Path IMM Suppression
**Goal:** Suppress LEA units via up path instead of gate path
**Method:** Added `ffn.W_up[unit, BD.OP_IMM] = -S` to all 6 units

**Result:** ✗ FAILED
- LEA: 0/10 (completely broken)
- IMM: Still broken
- Up-path suppression also prevented LEA from activating

### Attempt 4: BP Marker Initialization Only
**Goal:** Initialize BP OUTPUT at marker for Layer 7 to copy
**Method:** Added 4 units in Layer 3 FFN to set OUTPUT_LO[0] and OUTPUT_HI[0] at MARK_BP on first step

**Result:** ✗ FAILED
- LEA: Still broken
- IMM: Still broken
- BP marker init alone wasn't sufficient and caused side effects

## Root Cause Analysis

### Issue 1: Opcode Flag Discrimination
The LEA units couldn't reliably distinguish between OP_LEA and OP_IMM:
- Both flags are ~5.0 when active, ~0.0 when inactive
- Simple thresholding (OP_LEA > 3.0) doesn't exclude IMM
- Negative weights in gate or up path broke LEA entirely
- May indicate crosstalk or overlapping flag values

### Issue 2: BP Architecture for First Step
LEA requires BP values in ALU registers via:
1. Layer 6 identity carry: BP EMBED → BP OUTPUT (fails on first step, no EMBED)
2. Layer 7 head 1: BP OUTPUT → ALU at AX marker
3. Layer 8/9 ADD: AX_CARRY + ALU → AX OUTPUT

On first step, BP has no EMBED, so Layer 6 identity carry doesn't work.
Adding BP marker init in Layer 3 wasn't sufficient to fix the data flow.

### Issue 3: Token 6 (AX Byte 0) Failure
Even with BP properly initialized, the ADD circuit at MARK_AX wasn't producing correct results.
This suggests either:
- ALU_LO/HI registers don't contain BP byte 0 after Layer 7 copy
- AX_CARRY doesn't contain FETCH value after Layer 6 write
- ADD units don't fire correctly with OP_LEA gate on first step

## Lessons Learned

### 1. Opcode Selectivity is Critical
Simple threshold-based gating isn't sufficient when multiple opcodes have similar flag magnitudes.
Need more sophisticated discrimination:
- Multi-way AND gates (OP_LEA AND NOT OP_IMM AND NOT OP_ADD...)
- Hierarchical opcode groups
- Exclusive flags computed in earlier layers

### 2. First-Step Initialization is Complex
Operations that rely on prev-step data (via EMBED or identity carry) require special handling on first step.
Can't simply add marker init without understanding full data flow through multiple layers.

### 3. Side Effects are Easy to Introduce
Changes to weight logic can have unexpected effects on other operations:
- Strong signals can interfere with other prediction paths
- Cancellation can remove needed defaults
- Suppression in one dimension can affect unrelated operations

### 4. Testing Must Be Comprehensive
Single-operation testing isn't enough - must verify:
- All variations of target operation (LEA 0, LEA 8, LEA 16)
- Related operations (IMM, ADD, arithmetic)
- Different execution steps (first vs subsequent)

## Current Status

**REVERTED:** All LEA fix attempts have been reverted to avoid breaking IMM operations.

**LEA Status:** Still 6/10 tokens correct (baseline)
**IMM Status:** Working correctly (10/10 tokens)

## Recommendations for Future Work

### Option 1: Diagnostic-First Approach
Before adding more weight units:
1. Trace layer outputs for LEA 8 to see where data flow breaks
2. Check OP_LEA flag values in Layer 9/10
3. Verify BP OUTPUT values at MARK_BP after Layer 3
4. Check ALU_LO/HI values after Layer 7 copy
5. Verify AX_CARRY values after Layer 6 write

### Option 2: Alternative Architecture
Consider different approaches:
- Use TEMP dimensions for BP routing (like ENT/JSR)
- Add dedicated LEA_BP dimensions written in Layer 3
- Use Layer 6 attention instead of Layer 7 for BP copy
- Implement LEA as special case in existing ADD circuit

### Option 3: Accept Limitation
If LEA is rarely used on first step:
- Document limitation in opcode status
- Verify LEA works correctly on subsequent steps (after BP EMBED available)
- Prioritize other opcode fixes

### Option 4: Neural Approach
Train small MLP to handle LEA case correctly:
- Input: OP_LEA flag, HAS_SE, BYTE_INDEX, BP values
- Output: Correct AX byte predictions
- Integrate as specialized Layer 9 or 10 module

## Files Modified (Reverted)

- `neural_vm/vm_step.py`
  - Layer 3 FFN: BP marker initialization (reverted)
  - Layer 9 FFN: LEA upper bytes units (reverted)

## Test Results (Before Revert)

```
LEA 0:   FAIL ✗ (Token 1 mismatch)
LEA 8:   PASS ✓ (after fix, broke IMM)
LEA 16:  PASS ✓ (after fix, broke IMM)
IMM 1:   FAIL ✗ (Token 8: pred=1 exp=0)
IMM 8:   FAIL ✗ (Token 8: pred=1 exp=0)
```

## Conclusion

LEA operations require deeper architectural changes than simple weight additions can provide. The interaction between opcode flags, first-step initialization, and multi-layer data flow is complex enough that diagnostic investigation should precede any further fix attempts.

The attempted fixes successfully improved LEA accuracy but at the cost of breaking IMM operations, demonstrating the difficulty of adding selective logic without proper opcode discrimination mechanisms.

---

**Status:** REVERTED
**Date:** 2026-03-29
**Reason:** Breaking IMM operations (Token 8 prediction incorrect)
**Next Steps:** Diagnostic investigation or alternative architecture design
