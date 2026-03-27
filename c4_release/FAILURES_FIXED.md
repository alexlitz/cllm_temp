# Test Failures - Found and Fixed

## Summary

This document tracks test failures discovered after removing the broken weight scaling code and the fixes applied.

## Failures Fixed

### 1. IndexError in UltraBatchRunner ✅ FIXED

**File**: `neural_vm/batch_runner_v2.py` line 87
**Error**: `IndexError: list index out of range`

**Symptom**:
```
self._build_context(bc, (data_list or [b''])[i], (argv_list or [[]])[i])
IndexError: list index out of range
```

**Root Cause**:
When running 256 programs in a batch, the code attempted to use default values:
- `data_list or [b'']` creates a list of length 1
- `argv_list or [[]]` creates a list of length 1
- But the code tries to access indices 0-255

**Fix**:
```python
# Before (BROKEN):
self.contexts = [
    self._build_context(bc, (data_list or [b''])[i], (argv_list or [[]])[i])
    for i, bc in enumerate(bytecodes)
]

# After (FIXED):
if data_list is None:
    data_list = [b''] * n_programs
if argv_list is None:
    argv_list = [[]] * n_programs
self.contexts = [
    self._build_context(bc, data_list[i], argv_list[i])
    for i, bc in enumerate(bytecodes)
]
```

**Impact**: This was blocking ALL tests from running.

**Tests Affected**: All tests in `test_opcodes_fast.py`

---

### 2. KeyError in Multiplication Cache ✅ FIXED

**File**: `neural_vm/tests/test_opcodes_fast.py` line 144
**Error**: `KeyError: (0, 255)`

**Symptom**:
```python
def test_mul_0_255(self):
    self.assertEqual(self.cache[(0, 255)], 0)
    # KeyError: (0, 255)
```

**Root Cause**:
The multiplication cache was generated for values 0-15 only:
```python
pairs = [(a, b) for a in range(16) for b in range(16)]
```

But the test `test_mul_0_255` expected the boundary value (0, 255) to be in the cache.

**Fix**:
```python
# Before (BROKEN):
pairs = [(a, b) for a in range(16) for b in range(16)]

# After (FIXED):
values = list(range(16)) + [255]
pairs = [(a, b) for a in values for b in values]
```

**Impact**: One multiplication test was failing.

**Tests Affected**: `TestMultiplication::test_mul_0_255`

---

### 3. KeyError in Binary Operations Cache ✅ FIXED

**File**: `neural_vm/tests/test_opcodes_fast.py` line 85
**Error**: `KeyError: (3, 5)`, `KeyError: (5, 3)`, etc.

**Symptom**:
```
def test_div_3_5(self):
    cache = _get_binop_cache(Opcode.DIV)
    self.assertEqual(cache[(3, 5)], 0)
    # KeyError: (3, 5)
```

**Root Cause**:
The binary operations cache was generated for a limited set of values:
```python
values = [0, 1, 5, 10, 15, 100, 127, 128, 255]
```

But tests used additional values like 3, 4, 7, 16, 17, 42, 50, 200, 240 that weren't in the cache.

**Fix**:
```python
# Before (BROKEN):
values = [0, 1, 5, 10, 15, 100, 127, 128, 255]

# After (FIXED):
values = [0, 1, 3, 4, 5, 7, 10, 15, 16, 17, 42, 50, 100, 127, 128, 200, 240, 255]
```

**Impact**: 27 tests were failing (binary ops, div/mod, comparisons)

**Tests Affected**:
- All TestDivMod tests
- Most TestBinaryOps tests
- Most TestComparisons tests

**Note**: This fix was applied by user/linter before re-running tests.

---

## Previously Fixed (Earlier in Session)

### 4. Weight Scaling Breaking Model ✅ FIXED

**File**: `neural_vm/vm_step.py` lines 1560-1619 (removed)

**Root Cause**:
Global scaling of Q/K weights (×1.15) and slopes (×1.2, ×1.3) for alternative configurations broke the model because slopes are hand-crafted for specific purposes:
- 0.1-0.5: Gentle recency for content matching
- 5.0: Steep slopes for relay heads
- 10.0: Threshold attention for structural detection

Global scaling destroyed the careful balance.

**Fix**:
- Removed weight scaling code
- Added documentation explaining why scaling doesn't work
- Rely on mechanical adaptations instead (null key, RoPE rotation, recency bias)

**Impact**: Model was producing wrong predictions even for simple programs like `IMM 42; EXIT`

---

## Test Status After Fixes

### Currently Passing (Verified):
- ✅ All IMM exit code tests (7 tests)
- ✅ Most multiplication tests (8/8 tests)
- ✅ Tests no longer blocked by IndexError

### Running:
- 🔄 Full test suite running in background to identify any remaining failures

## Files Modified

1. **neural_vm/batch_runner_v2.py**
   - Lines 84-94: Fixed IndexError by creating proper-length default lists

2. **neural_vm/tests/test_opcodes_fast.py**
   - Lines 68-76: Fixed multiplication cache to include boundary value 255

3. **neural_vm/vm_step.py** (earlier)
   - Lines 1560-1619: Removed broken weight scaling code
   - Lines 1507-1554: Added documentation

4. **CONFIGURATION_FIX_SUMMARY.md** (new)
   - Comprehensive documentation of all fixes

## Next Steps

1. ✅ Wait for full test suite to complete
2. ⏳ Identify any remaining failures
3. ⏳ Fix additional issues if found
4. ⏳ Verify all 59 tests pass like they did at 19:43

## Notes

- The working test run at 19:43 had 59 tests passing
- After these fixes, we're restoring that functionality
- The IndexError fix was critical - it was blocking all tests
- The KeyError fix was for a test added after the 19:43 working state
