# Neural VM Current Status - Updated

## ✅ What's Working

### JMP (Jump) - MOSTLY WORKING ✅
- **Token 1 (PC_b0)**: Fixed in previous session (predicts 16) ✓
- **Tokens 2, 13, 21**: All passing ✓
- **PSH fix impact**: Need full verification but looks promising

### NOP (No Operation) - WORKING ✅
- Test passing, no issues

---

## ❌ What's Broken

### IMM 42 and IMM 255 - BROKEN (Different Issue) ❌
**Symptoms**:
- Token 0 (REG_AX): ✓ Correct
- Token 1+: Model stuck predicting 257 (REG_AX) for ALL subsequent tokens

**This is NOT related to JMP/PSH**:
- Separate bug in IMM handling
- Model appears to loop/get stuck
- Needs independent investigation

**Root cause**: Unknown (not investigated yet)

---

## 🔧 PSH Fix - IMPLEMENTED (Testing Needed)

### Problem Solved
**CMP[0] leakage** causing PSH units to false-activate during JMP:
- JMP sets CMP[0] ≈ 7.0 at PC marker
- Value decays to ≈ 5.2 at byte positions
- PSH threshold: CMP[0] + MARK_STACK0 > 1.5
- False activation: 5.2 + 1.0 = 6.2 > 1.5 ✗

### Solution Implemented
**Negative weight suppression** on CMP[0] in up:
```python
W_up[CMP[0]] = S - 4*S  # Net: -3*S
```

**Effect**:
- PSH (CMP[0]~1): up = (-3×1 + 1) × 100 = -200, borderline
- JMP (CMP[0]~5): up = (-3×5 + 1) × 100 = -1400, suppressed ✓

**Status**: 
- ✅ PSH SP and PSH STACK0 units re-enabled
- ✅ Tokens 2, 13, 21 pass with new weights
- ⏳ Full verification needed

---

## 📊 Test Summary

| Test | Status | Notes |
|------|--------|-------|
| NOP | ✅ PASS | Stable |
| IMM 0 | ✅ PASS | Working |
| IMM 42 | ❌ FAIL | Stuck predicting 257 |
| IMM 255 | ❌ FAIL | Stuck predicting 257 |
| JMP 16 | ✅ LIKELY PASS | Tokens 2,13,21 verified |
| JMP 8 | ⏳ UNKNOWN | Not tested yet |
| EXIT | ⏳ UNKNOWN | Depends on PSH |
| LEA 8 | ⏳ UNKNOWN | Not tested |
| ADD | ⏳ UNKNOWN | Not tested |

**Estimated**: 3-4 tests passing (NOP, IMM 0, JMP 16, possibly JMP 8)

---

## 🎯 Next Steps

### Priority 1: Verify PSH Fix Works
Run full test suite to confirm:
1. JMP tests pass (both JMP 16 and JMP 8)
2. PSH-dependent tests work (EXIT, possibly ADD/LEA)
3. No regressions in other tests

### Priority 2: Fix IMM 42/255
Investigate why model gets stuck:
- Check if it's a tokenization issue
- Verify embedding values
- Trace through layers to find where loop occurs
- May be unrelated to PSH/JMP fixes

### Priority 3: Run Full Test Suite
Get complete picture of what works:
```bash
python -m pytest neural_vm/tests/test_strict_neural_predictions.py -v
```

---

## 💡 Key Insights from This Session

### 1. CMP[0] Threshold Discrimination
**Discovery**: Can use threshold on CMP[0] itself as discriminator
- PSH: CMP[0] ≈ 1.0
- JMP leakage: CMP[0] ≈ 5.2
- Threshold at 2.0 cleanly separates them

**Implementation**: Negative weight in W_up suppresses high CMP[0]

### 2. IS_BYTE Unreliable
**Finding**: IS_BYTE = 0 at embedding layer for position 29
- Should be 1.0 for byte tokens
- Being cleared or not set correctly
- Cannot rely on IS_BYTE for position discrimination

### 3. IMM Separate Issue
**Finding**: IMM 42/255 fail differently than JMP
- Model stuck in prediction loop
- Not related to CMP[0] leakage
- Needs independent debugging

---

## 📈 Progress This Session

**Achievements**:
- ✅ Identified CMP[0] leakage as root cause
- ✅ Implemented PSH fix using negative weight suppression
- ✅ Re-enabled PSH SP and PSH STACK0 units
- ✅ Verified approach works for tokens 2, 13, 21
- ✅ Discovered IMM issue (separate problem)

**Remaining Work**:
- Verify full JMP test passes
- Verify PSH-dependent tests work
- Fix IMM 42/255 stuck loop issue
- Run complete test suite

**Estimated Time to Complete**:
- PSH verification: 30 minutes
- IMM fix: 2-3 hours
- Total: ~3-4 hours to get most tests passing

---

## 🔍 Technical Details

### PSH Fix Implementation

**File**: `neural_vm/vm_step.py`

**Lines 2683-2703** (PSH SP):
```python
T_psh = 1.5  # Re-enabled
ffn.W_up[unit, BD.CMP + 0] = S - 4*S  # -3*S suppression
ffn.W_up[unit, BD.MARK_SP] = S
ffn.b_up[unit] = -S * T_psh
```

**Lines 2722-2738** (PSH STACK0):
```python
T_psh_s0 = 1.5  # Re-enabled
ffn.W_up[unit, BD.CMP + 0] = S - 4*S  # -3*S suppression
ffn.W_up[unit, BD.MARK_STACK0] = S
ffn.b_up[unit] = -S * T_psh_s0
```

**Math**:
```
up = W_up @ x + b_up
   = (-3*S)*CMP[0] + S*MARK + (-S*threshold)

PSH case (CMP[0]=1, MARK=1):
  up = -300 + 100 - 150 = -350
  silu(-350) ≈ 0, minimal activation

JMP case (CMP[0]=5, MARK=1):
  up = -1500 + 100 - 150 = -1550
  silu(-1550) ≈ 0, strongly suppressed
```

Wait, this math doesn't look right. Let me recalculate:

Actually for PSH to work, we need up > 0 after silu. Let me reconsider...

The threshold pattern is:
```
up = CMP[0] + MARK - threshold
PSH: up = 1 + 1 - 1.5 = 0.5 > 0 ✓
```

With suppression:
```
up = (S - 4*S)*CMP[0] + S*MARK - S*threshold
   = S*(-3*CMP[0] + MARK - threshold)

PSH: up = 100*(-3 + 1 - 1.5) = 100*(-3.5) = -350 < 0 ✗
```

This will suppress PSH too! The fix is broken.

I need to adjust the weights. Let me recalculate what's needed:
- PSH: CMP[0]=1, want activation
- JMP: CMP[0]=5, want suppression

The issue is that I'm using the SAME CMP[0] value for both detection and suppression. I need a nonlinear relationship, which FFNs don't naturally provide.

Alternative: use a lower suppression factor:
```
W_up[CMP[0]] = S - 1.5*S = -0.5*S

PSH: up = 100*(-0.5*1 + 1 - 1.5) = 100*(-1) = -100 < 0 ✗
Still doesn't work!
```

The fundamental issue is that I can't separate PSH (CMP[0]=1) from JMP (CMP[0]=5) using a linear threshold if both require the SAME additive contribution from CMP[0].

I think I need to use the gate after all, or find a completely different approach.

