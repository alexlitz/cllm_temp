# ROOT CAUSE DISCOVERY: set_vm_weights() Requirement

**Date**: 2026-04-10 (continued session)
**Status**: **CRITICAL FINDING**

---

## Executive Summary

**The entire threshold attention "bug" investigation was based on tests that weren't calling `set_vm_weights()`.**

All previous test results showing broken threshold attention (outputting -1.47 to 1.76) were from **uninitialized/random weights**. With `set_vm_weights()` properly called, threshold attention works **perfectly** with binary 0.99 ≈ 1.0 outputs.

---

## The Discovery

### What Was Wrong

**Previous tests** (test_byte_index_full_step.py, test_threshold_mechanism.py, etc.):
```python
model = AutoregressiveVM(n_layers=17)
# ❌ MISSING: set_vm_weights(model)
x = model.embed(input_ids)  # Using random/zero weights!
```

**Result**: Random embeddings, random attention weights
- IS_MARK at markers: -0.67, 3.22, -0.38 (random!)
- Q values: 0.00 (should be 80.00)
- K values: 0.00 (should be threshold * IS_MARK)
- Threshold outputs: -1.47 to 1.76 (random!)

### What Is Correct

**Proper test setup**:
```python
model = AutoregressiveVM(n_layers=17)
set_vm_weights(model)  # ✅ REQUIRED!
x = model.embed(input_ids)  # Now uses configured weights
```

**Result**: Properly configured weights
- IS_MARK at markers: 1.00 ✓
- Q values: 80.00 ✓
- K values: threshold * IS_MARK ✓
- Threshold outputs: **0.99 ≈ 1.0** (binary!) ✓

---

## Test Results With Proper Configuration

### Embedding IS_MARK Values

```
Token           IS_MARK    Expected   Status
REG_PC          1.00       1.0        ✓
REG_AX          1.00       1.0        ✓
REG_SP          1.00       1.0        ✓
REG_BP          1.00       1.0        ✓
MEM             1.00       1.0        ✓
CODE_START      1.00       1.0        ✓
```

### Threshold Attention Outputs (Slope=10.0)

**SP section hop-count threshold values**:
```
d    L1H0   L1H1   L1H2    H0     H1   | Expected
------------------------------------------------------
0    0.99   1.00   1.00   1.00   1.00  | All within
1    0.00   0.99   1.00   1.00   1.00  | L1H1=1     ✓
2    0.00   0.00   0.99   1.00   1.00  | L1H2=1     ✓
3    0.00   0.00   0.00   0.99   1.00  | H0=1       ✓
4    0.00   0.00   0.00   0.00   0.99  | H1=1       ✓
```

**Perfect binary outputs!** Each threshold fires at exactly the right distance.

### BYTE_INDEX Generation

```
Position   BYTE_INDEX_0   _1      _2      _3     Expected
---------------------------------------------------------
Byte 0     0.97           0.01    0.00    0.00   _0=1.0
Byte 1     0.00           0.97    0.01    0.00   _1=1.0
Byte 2     0.00           0.00    0.97    0.01   _2=1.0
Byte 3     0.00           0.00    0.00    0.97   _3=1.0
```

**Nearly correct** with minor precision issues:
- Intended BYTE_INDEX: 0.97 (instead of 1.0)
- Small leakage to next: 0.01 (instead of 0.0)

**Root cause of precision issue**:
- Threshold heads output 0.99 instead of exactly 1.0
- L1 FFN formula: `gate = 1.0 - L1H1 = 1.0 - 0.99 = 0.01`
- This 0.01 leak propagates through SwiGLU

---

## Impact on Previous Conclusions

### What Was Incorrect

**Previous "FINAL_STATUS_95_PERCENT_NEURAL.md" claimed**:
- ❌ Threshold attention mechanism is fundamentally broken
- ❌ Outputs are not binary (range: -1.47 to 1.76)
- ❌ Negative values instead of 0/1
- ❌ Wrong positions fire

**Reality**: These were symptoms of uninitialized weights, not broken architecture.

### What Is Actually True

**Threshold attention**:
- ✅ Works perfectly with configured weights
- ✅ Outputs are binary (0.00 or 0.99 ≈ 1.0)
- ✅ All values are non-negative
- ✅ Correct positions fire

**BYTE_INDEX**:
- ⚠️ Minor precision issue (0.97 vs 1.0, 0.01 leak)
- ❓ Unknown if this causes MEM corruption

**Function calls**:
- ❓ Status unknown - tests timeout
- ❓ May still have issues unrelated to threshold attention

---

## Revised Understanding

### Architecture Status

**Threshold Attention (L0/L1)**:
- Status: ✅ **WORKS CORRECTLY**
- Binary outputs: 0.00 or 0.99 ≈ 1.0
- Correct threshold behavior at all distances

**BYTE_INDEX Generation (L1 FFN)**:
- Status: ⚠️ **MOSTLY WORKS** with minor precision
- Values: 0.97 instead of 1.0
- Leakage: 0.01 to next index

**Downstream Layers (L3, L14, etc.)**:
- Status: ❓ **UNKNOWN**
- May work with BYTE_INDEX=0.97
- May need exact 1.0 values
- Requires testing

### Test Timeout Issue

**Observation**: Both standalone and pytest tests timeout
- test_function_call_simple.py: timeouts after 120s
- pytest test_opcodes.py::TestIMMExitCodes::test_imm_000_exit: timeouts after 60s

**Possible causes**:
1. Infinite loop in VM execution
2. Tests are very slow (model forward passes)
3. Different issue than threshold attention
4. Runner initialization issue

---

## Next Steps (Revised)

### Immediate (Option A Still Valid, But Different Problem)

1. **Investigate test timeouts**
   - Why do even simple tests timeout?
   - Is the VM stuck in a loop?
   - Is model initialization failing silently?

2. **Test if BYTE_INDEX=0.97 causes issues**
   - Run L3 PSH test to see if bytes 1-3 are written
   - Run L14 MEM addr test to see if addresses are correct
   - Check if 0.97 vs 1.0 breaks downstream logic

3. **Consider increasing precision**
   - If BYTE_INDEX=0.97 causes issues, options:
     - Increase ALiBi slope for sharper thresholds
     - Add thresholding after BYTE_INDEX (if x > 0.8: x = 1.0)
     - Adjust L1 FFN formula to be less sensitive

### Alternative Paths

**Option A (Revised)**: Fix BYTE_INDEX precision if needed
- Threshold attention is NOT broken
- Only need to address 0.97 vs 1.0 issue if it causes problems
- Much simpler than originally thought

**Option B**: Accept 0.97 as "close enough"
- If L3/L14 work with BYTE_INDEX=0.97
- No changes needed to threshold attention or BYTE_INDEX
- Just document precision characteristics

**Option C**: Investigate timeout issue first
- If tests timeout, nothing else matters
- May be a completely different bug
- Could be unrelated to threshold/BYTE_INDEX

---

## Files That Need Updating

### Test Files (Add set_vm_weights)

All test files that create AutoregressiveVM directly:
- ✅ test_embedding_is_mark.py - FIXED
- ✅ test_threshold_mechanism.py - FIXED
- ✅ test_byte_index_full_step.py - FIXED
- ✅ test_threshold_slopes.py - FIXED
- ⚠️ Any other custom tests

### Documentation (Corrections)

Files that document the "bug":
- FINAL_STATUS_95_PERCENT_NEURAL.md - Needs major corrections
- SESSION_SUMMARY_2026-04-10_THRESHOLD_BUG.md - Needs corrections
- THRESHOLD_ATTENTION_BUG.md - May need complete rewrite

---

## Lessons Learned

### Critical Importance of set_vm_weights()

**`AutoregressiveVM.__init__()`** does NOT configure weights:
- Creates layers with random/zero weights
- Embedding has random values
- Attention/FFN weights are identity (via residual)

**`set_vm_weights(model)`** MUST be called to configure:
- Embedding IS_MARK flags
- Attention Q/K/V/O matrices
- FFN W_up/W_gate/W_down weights
- All VM semantics

**AutoregressiveVMRunner** handles this automatically:
- Creates model in `__init__()`
- Calls `set_vm_weights()` internally
- Tests using runner don't need manual call

**Custom tests** must call it manually:
```python
model = AutoregressiveVM(n_layers=17)
set_vm_weights(model)  # REQUIRED!
```

### Investigation Methodology

**What went wrong**:
- Ran tests without proper initialization
- Interpreted random outputs as bugs
- Built entire theory on incorrect data
- Spent hours debugging non-existent problem

**What should have been done**:
1. Verify test setup first
2. Check if set_vm_weights() is called
3. Validate weights are configured before testing
4. Compare test setup to working examples (runner)

---

## Current Status

### What We Know (UPDATED)

✅ **Threshold attention works correctly**
✅ **Embedding IS_MARK configuration works**
✅ **Binary 0/1 outputs achieved (0.99 ≈ 1.0)**
✅ **Architecture is sound**

⚠️ **BYTE_INDEX has minor precision (0.97 vs 1.0, 0.01 leak)**

❓ **Tests timeout - unknown cause**
❓ **Function calls - status unknown**
❓ **Downstream layers (L3, L14) - unknown if BYTE_INDEX=0.97 works**

### What We Don't Know

1. Why do tests timeout?
2. Does BYTE_INDEX=0.97 cause downstream issues?
3. Do function calls (JSR/LEV) actually work?
4. Is the MEM corruption still present?
5. Is 95% or 100% neural execution achieved?

### Confidence Levels

- **High confidence**: Threshold attention works
- **High confidence**: Test setup was the problem
- **Medium confidence**: BYTE_INDEX precision is minor issue
- **Low confidence**: Everything else needs retesting

---

## Conclusion

**The threshold attention "bug" was a red herring caused by improper test setup.**

With `set_vm_weights()` properly called:
- Threshold attention produces perfect binary outputs (0.99 ≈ 1.0)
- BYTE_INDEX has minor precision issues (0.97 vs 1.0)
- Architecture is fundamentally sound

**The real question now**: Does the BYTE_INDEX precision issue (0.97 vs 1.0) cause problems downstream? This requires testing L3 PSH and L14 MEM with configured weights.

**The blocking issue**: Tests timeout, making it impossible to verify anything. This must be investigated first.

---

**Key Insight**: Always verify test initialization before debugging "bugs". Uninitialized weights look like broken logic.

**Files**:
- Test files: test_embedding_is_mark.py, test_threshold_mechanism.py, test_byte_index_full_step.py
- This document: ROOT_CAUSE_DISCOVERY_set_vm_weights.md

**Next Session Priority**: Investigate why tests timeout.
