# Validation Now Enabled By Default

## Changes Made

### 1. Default Validation Ratio Changed
**File**: `src/baked_c4.py`

```python
# BEFORE:
def __init__(self, use_speculator: bool = True, validation_ratio: float = 0.0):
    # validation_ratio=0.0 meant NO validation by default

# AFTER:
def __init__(self, use_speculator: bool = True, validation_ratio: float = 0.1):
    # validation_ratio=0.1 means 10% validation by default
```

### 2. Added ValidationError Exception
**File**: `src/speculator.py`

Added exception class:
```python
class ValidationError(Exception):
    """Raised when neural VM validation fails against fast VM."""
    pass
```

### 3. Raise on Mismatch By Default
**File**: `src/speculator.py`

```python
# BEFORE:
def __init__(self, transformer_vm=None, validate_ratio: float = 0.1):
    # No raise_on_mismatch parameter

# AFTER:
def __init__(self, transformer_vm=None, validate_ratio: float = 0.1, raise_on_mismatch: bool = True):
    self.raise_on_mismatch = raise_on_mismatch
```

### 4. Validation Failures Now Raise Errors
**File**: `src/speculator.py`

```python
if fast_result != trans_exit_code:
    self.mismatches += 1
    if self.raise_on_mismatch:
        raise ValidationError(
            f"Neural VM validation failed!\n"
            f"  Fast VM result: {fast_result}\n"
            f"  Neural VM result: {trans_exit_code}\n"
            f"  Validations: {self.validations}\n"
            f"  Mismatches: {self.mismatches}"
        )
```

## Impact on Tests

### Before These Changes
```
$ python -m tests.run_1000_tests
  Total tests: 1096
  Passed: 1096 ✓  (FALSE POSITIVE - neural VM was broken)
  Failed: 0

  Validation: DISABLED
  Neural VM: BROKEN (but hidden)
```

### After These Changes
```
$ python -m tests.run_1000_tests
  Total tests: 1096

  Validation: ENABLED (10% of runs)
  Expected behavior:
    - ~110 tests will be validated
    - ~99 validations will FAIL (neural VM returns 0 for everything)
    - ~11 tests might pass (programs that return 0)

  Tests will FAIL when validation detects mismatch
  This is CORRECT behavior - tests should fail when neural VM is broken
```

## Why This Is Important

### Before (Dangerous):
- Tests passed even though neural VM was completely broken
- False sense of security
- Regressions would go undetected
- Neural VM could silently break

### After (Correct):
- Tests fail when neural VM is broken
- Catches regressions immediately
- Forces fixes to neural VM
- Accurate test results

## Expected Test Results

With neural VM in current state (only works for `return 0`):

### Programs That Will Pass Validation:
- `int main() { return 0; }` ✓
- Loop countdown tests (result=0) ✓
- Some edge cases ✓
- **Estimated: ~10% of validated tests**

### Programs That Will Fail Validation:
- `int main() { return 42; }` ✗
- All arithmetic tests ✗
- All function calls ✗
- All recursion ✗
- **Estimated: ~90% of validated tests**

## How to Disable Validation (Not Recommended)

If you need to disable validation temporarily:

```python
# Disable all validation
c4 = BakedC4Transformer(validation_ratio=0.0)

# Or keep validation but don't raise errors (just log)
from src.speculator import SpeculativeVM
c4 = BakedC4Transformer()
c4.speculator.raise_on_mismatch = False
```

**WARNING**: Disabling validation hides neural VM bugs!

## Current Status

✅ **Validation enabled by default** (10% sample rate)
✅ **ValidationError raises on mismatch**
✅ **Tuple extraction works correctly**
✅ **Tests will fail when neural VM fails**

⚠️ **Neural VM is still broken** (only works for `return 0`)
- This will cause ~90% of validated tests to fail
- This is CORRECT behavior
- Tests should fail until neural VM is fixed

## Next Steps

To make tests pass, you must fix the neural VM so it produces correct outputs for all programs, not just `return 0`.

The validation system is now working correctly and will catch when the neural VM is fixed!

## Testing Validation

Run these test scripts to verify validation works:

```bash
# Test with 100% validation (slow, ~24s for 2 tests)
python test_validation_100percent.py

# Test with default 10% validation
python test_validation_default.py

# Run full test suite (will fail when validation detects broken neural VM)
python -m tests.run_1000_tests
```
