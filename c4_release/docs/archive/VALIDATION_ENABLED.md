# Neural Model Validation Enabled

## Summary

**Tests now validate the neural model and will fail if it produces incorrect results.**

## What Changed

### 1. SpeculativeVM Now Raises on Validation Failures

Modified `src/speculator.py`:
- Added `ValidationError` exception class
- Added `raise_on_mismatch` parameter to `SpeculativeVM.run()`
- When validation is enabled and neural model disagrees with Fast VM, raises `ValidationError`

```python
raise ValidationError(
    f"Neural model validation failed!\n"
    f"  Fast VM result: {fast_result}\n"
    f"  Neural VM result: {trans_result}\n"
    f"  Bytecode length: {len(bytecode)}\n"
)
```

### 2. BakedC4Transformer Enables Validation by Default

Modified `src/baked_c4.py`:
- Added `validate_neural=True` parameter (default)
- Added `validation_sample_rate=0.1` parameter (default 10%)
- Validation is now enabled for all tests by default

```python
c4 = BakedC4Transformer(
    use_speculator=True,        # Fast execution via Draft VM
    validate_neural=True,        # Validate neural model
    validation_sample_rate=0.1,  # Validate 10% of runs
)
```

### 3. Performance Impact

**Strategy**: Sample-based validation for speed
- **90% of tests**: Run with Fast VM only (very fast)
- **10% of tests**: Run both Fast VM and Neural VM for validation
- **Any mismatch**: Immediately raises ValidationError and fails test

This provides:
- ✓ Fast test execution (mostly using Fast VM)
- ✓ Neural model verification (catches broken model)
- ✓ Fail-fast behavior (test suite fails on first mismatch)

## Test Results

### Before (Validation Disabled)
```bash
$ python tests/run_1000_tests.py --quick
All 100 tests passed ✓
```
**Problem**: Tests passed even though neural model is broken!

### After (Validation Enabled)
```bash
$ python tests/run_1000_tests.py --quick
ValidationError: Neural model validation failed!
  Fast VM result: 42
  Neural VM result: ('', 0)
```
**Fixed**: Tests now fail when neural model produces wrong results!

## Example Usage

### Default (Validation Enabled, 10% Sample Rate)
```python
from src.baked_c4 import BakedC4Transformer
from src.speculator import ValidationError

c4 = BakedC4Transformer()  # validate_neural=True by default

try:
    result = c4.run_c("int main() { return 42; }")
except ValidationError as e:
    print(f"Neural model failed validation: {e}")
```

### Disable Validation (Not Recommended)
```python
# For debugging or when you know neural model is broken
c4 = BakedC4Transformer(validate_neural=False)
result = c4.run_c("int main() { return 42; }")  # Always returns 42 (from Fast VM)
```

### 100% Validation (Slower, More Thorough)
```python
# Validate every single run
c4 = BakedC4Transformer(validation_sample_rate=1.0)
```

### Lower Sample Rate (Faster, Less Thorough)
```python
# Validate only 1% of runs
c4 = BakedC4Transformer(validation_sample_rate=0.01)
```

## Technical Details

### Hash-Based Sampling

Validation uses deterministic hash-based sampling:
```python
should_validate = hash(tuple(bytecode)) % 100 < validate_ratio * 100
```

This means:
- Same bytecode always gets same validation decision
- Different bytecode gets different decisions
- ~10% of unique programs will be validated
- Running same program multiple times won't increase validation

### Validation Process

For each validated run:
1. Fast VM executes bytecode → fast_result
2. Neural VM executes same bytecode → neural_result
3. If fast_result != neural_result → raise ValidationError
4. Otherwise, return fast_result

## Current Status

**Neural Model**: ❌ **BROKEN**
- Produces `('', 0)` for all programs
- Halts prematurely after 2 steps
- Cannot execute autonomously

**Fast VM (Speculator)**: ✅ **WORKING**
- Executes correctly
- Returns correct results
- Used for actual program execution

**Validation**: ✅ **ENABLED**
- Catches neural model failures
- Fails tests appropriately
- Maintains fast execution via sampling

## Files Modified

1. `src/speculator.py`
   - Added `ValidationError` exception
   - Added `raise_on_mismatch` parameter
   - Raises exception on validation failure

2. `src/baked_c4.py`
   - Added `validate_neural` parameter (default True)
   - Added `validation_sample_rate` parameter (default 0.1)
   - Enabled validation by default

## Testing

```bash
# Test that validation catches failures
python test_validation_enabled.py

# Test with varied programs
python test_varied_validation.py

# Run test suite (will fail due to broken neural model)
python tests/run_1000_tests.py --quick
```

## Next Steps

To make tests pass, the neural model needs to be fixed. See `PURE_NEURAL_STATUS.md` for details on what's broken and how to fix it.

For now, you can:
1. Keep validation enabled to catch regressions
2. Work on fixing the neural model layer by layer
3. Tests will automatically pass once neural model works correctly
