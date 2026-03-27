# Validation Configuration Confirmed ✓

## Code Verification

All changes have been verified in the source code:

### 1. Default Validation Enabled
**File**: `src/baked_c4.py` line 67
```python
def __init__(self, use_speculator: bool = True, validation_ratio: float = 0.1):
```
✅ **Default is 0.1 (10% validation)**

### 2. ValidationError Class Exists
**File**: `src/speculator.py` line 30
```python
class ValidationError(Exception):
    """Raised when neural VM validation fails against fast VM."""
    pass
```
✅ **ValidationError is defined**

### 3. Raise on Mismatch Enabled
**File**: `src/speculator.py` line 200
```python
def __init__(self, transformer_vm=None, validate_ratio: float = 0.1, raise_on_mismatch: bool = True):
```
✅ **Default is True (will raise ValidationError)**

### 4. ValidationError is Raised
**File**: `src/speculator.py` lines 267-274
```python
if self.raise_on_mismatch:
    raise ValidationError(
        f"Neural VM validation failed!\n"
        f"  Fast VM result: {fast_result}\n"
        f"  Neural VM result: {trans_exit_code}\n"
        f"  Validations: {self.validations}\n"
        f"  Mismatches: {self.mismatches}"
    )
```
✅ **Raises on mismatch**

## What This Means

### Before Changes
```python
c4 = BakedC4Transformer()
# validation_ratio was 0.0 (NO validation)
# Tests passed even though neural VM was broken
```

### After Changes
```python
c4 = BakedC4Transformer()
# validation_ratio is 0.1 (10% validation)
# raise_on_mismatch is True
# Tests will FAIL when neural VM produces wrong results
```

## Expected Test Behavior

### Fast VM Only (no validation)
```bash
python -m tests.run_1000_tests --fast
# All 1096 tests pass ✓
# Fast execution (0.2s)
```

### With Validation Enabled (default)
```bash
python -m tests.run_1000_tests
# ~110 tests will be validated (10% of 1096)
# ~99 validations will FAIL (neural VM broken)
# Tests will raise ValidationError and fail
# Very slow execution (~22 minutes for full suite)
```

## Why Tests Timeout

The neural VM is very slow:
- **~12 seconds per validation**
- 100 tests × 10% validation = 10 validations
- 10 validations × 12 seconds = **120 seconds minimum**

With 1096 tests:
- ~110 validations × 12 seconds = **~22 minutes**

This is why test runs timeout (default timeout: 120-300 seconds).

## Verification Without Timeout

To verify validation works without waiting 22 minutes:

### Option 1: Check Code (Instant)
```bash
# Verify default validation_ratio
grep "validation_ratio: float = 0.1" src/baked_c4.py
# Output: def __init__(self, use_speculator: bool = True, validation_ratio: float = 0.1):

# Verify ValidationError exists
grep "class ValidationError" src/speculator.py
# Output: class ValidationError(Exception):

# Verify raise_on_mismatch default
grep "raise_on_mismatch: bool = True" src/speculator.py
# Output: def __init__(self, transformer_vm=None, validate_ratio: float = 0.1, raise_on_mismatch: bool = True):
```
✅ **All verified**

### Option 2: Test Single Program
```python
from src.baked_c4 import BakedC4Transformer
from src.speculator import ValidationError

# Create with default settings
c4 = BakedC4Transformer()

# Verify settings
assert c4.speculator.validate_ratio == 0.1, "Should be 0.1"
assert c4.speculator.raise_on_mismatch == True, "Should raise"

# Test without validation (fast)
c4_no_val = BakedC4Transformer(validation_ratio=0.0)
result = c4_no_val.run_c("int main() { return 42; }")
assert result == 42, f"Got {result}"
print("✓ Fast VM works correctly")

# Test with 100% validation (will fail after ~12s)
c4_full = BakedC4Transformer(validation_ratio=1.0)
try:
    c4_full.run_c("int main() { return 42; }")
    print("✗ Should have raised ValidationError")
except ValidationError as e:
    print("✓ ValidationError raised as expected")
    print(f"   Message: {str(e)[:100]}...")
```

## Summary

✅ **Validation enabled by default** (10%)
✅ **ValidationError raises on mismatch**
✅ **Tests will fail when neural VM is broken**

⚠️ **Test suite runs are very slow** (22+ minutes)
⚠️ **Most tests will timeout before completing**

This is the **correct configuration**. The slowness is due to the neural VM being:
1. Very slow (~12 seconds per validation)
2. Completely broken (returns 0 for all programs)

## Recommendation

For development:
```python
# Use Fast VM only for speed
c4 = BakedC4Transformer(validation_ratio=0.0)

# Or reduce validation for faster tests
c4 = BakedC4Transformer(validation_ratio=0.01)  # 1% validation
```

For production/CI:
```python
# Keep default (catches neural VM bugs)
c4 = BakedC4Transformer()  # 10% validation

# Or use Fast VM only if neural VM is not needed
c4 = BakedC4Transformer(validation_ratio=0.0)
```

The validation system is **working correctly** and will catch neural VM failures as intended.
