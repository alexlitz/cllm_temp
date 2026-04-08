# Testing the Layer 6 Head Allocation Fix

**Date**: 2026-04-08
**Fix**: Disabled `_set_layer6_relay_heads()` to prevent head allocation conflicts

## Quick Verification (< 30 seconds)

### Weight Configuration Test

This test verifies the fix is correct by checking the weight configurations:

```bash
cd tests
python test_layer6_head_allocation.py
```

**Expected output:**
```
✓ All weight configuration tests passed!
The Layer 6 head allocation fix is verified to be correct.
```

**What it checks:**
1. ✓ No Layer 6 heads write to AX_CARRY at the AX marker (preserves carry-forward)
2. ✓ Layer 3 Head 1 correctly sets AX_CARRY at AX marker
3. ✓ Head 6 has no configuration conflicts

**Status**: ✅ PASSING (as of 2026-04-08)

## Full Execution Tests (2-5 minutes)

### Arithmetic Operations Without Handlers

This test runs actual C programs to verify arithmetic operations work:

```bash
cd tests
python test_arithmetic_no_handlers.py
```

**Expected output:**
```
  [ 1] ✓ PASS: Literal constant
  [ 2] ✓ PASS: Addition
  [ 3] ✓ PASS: Subtraction
  ...
  Passed: 14/14
  Success rate: 100.0%
```

**What it tests:**
- ADD, SUB, MUL, DIV, MOD operations
- Chained operations (e.g., `20 + 20 + 2`)
- Mixed operations (e.g., `(10 + 5) * 2 + 12`)
- Edge cases (addition with zero, division by one, etc.)

**Note**: This test loads the full neural model, which takes 1-2 minutes.

**Status**: ⏳ PENDING (model loading is slow, test created but not yet run)

## Regression Test Suite (30-60 minutes)

### Full 1000+ Test Suite

Run the comprehensive test suite to ensure no regressions:

```bash
cd tests
python run_1000_tests.py --quick  # First 100 tests
python run_1000_tests.py          # All 1000+ tests
```

**What it tests:**
- 200 basic arithmetic tests
- 50 modulo tests
- 100 variable assignment tests
- 100 comparison tests
- 150 loop tests
- 200 function tests
- And more...

**Status**: ⏸️ NOT YET RUN (full suite takes significant time)

## Manual Verification Scripts

These scripts were created during the investigation and can be run individually:

### 1. AX_CARRY Preservation Check

```bash
python verify_ax_carry_at_ax_marker.py
```

Checks that no Layer 6 heads corrupt AX_CARRY at the AX marker.

### 2. Head 6 Conflict Check

```bash
python check_head6_conflict.py
```

Checks if Head 6 has configuration conflicts between multiple functions.

### 3. Quick Arithmetic Test

```bash
python quick_arithmetic_test.py
```

Simple 3-test verification (literal, ADD, SUB).

## Test Files Created

| File | Purpose | Speed | Status |
|------|---------|-------|--------|
| `tests/test_layer6_head_allocation.py` | Weight config verification | Fast (< 30s) | ✅ Passing |
| `tests/test_arithmetic_no_handlers.py` | Execution test | Medium (2-5min) | ⏳ Pending |
| `verify_ax_carry_at_ax_marker.py` | Manual verification | Fast | ✅ Passing |
| `check_head6_conflict.py` | Manual verification | Fast | ✅ Passing |
| `quick_arithmetic_test.py` | Quick execution test | Medium | ⏳ Pending |

## Continuous Integration

### Recommended CI Tests

For fast CI pipelines, run only the weight configuration test:

```bash
# Fast test (< 30 seconds)
python tests/test_layer6_head_allocation.py
```

For comprehensive CI, include execution tests:

```bash
# Medium tests (2-5 minutes)
python tests/test_layer6_head_allocation.py
python tests/test_arithmetic_no_handlers.py

# Full regression (30-60 minutes)
python tests/run_1000_tests.py --quick
```

## Expected Behavior

### Before the Fix

**Symptoms:**
- ❌ Arithmetic operations fail without Python handlers
- ❌ JMP operations fail on first step
- ❌ JSR (function call) operations fail
- ❌ Layer 6 heads 2-3 have conflicting configurations

**Root Cause:**
- `_set_layer6_relay_heads()` overwrote heads 2-3 configured by `_set_layer6_attn()`

### After the Fix

**Expected:**
- ✅ Arithmetic operations work via Layer 8/9 FFN (neural implementation)
- ✅ JMP operations work (head 2 preserved)
- ✅ JSR operations work (head 3 preserved)
- ✅ No head allocation conflicts
- ✅ AX_CARRY path preserved from Layer 3 to Layer 8

## Debugging Failed Tests

If tests fail, check:

1. **Weight configuration issue:**
   ```bash
   python tests/test_layer6_head_allocation.py
   ```
   If this fails, the fix wasn't applied correctly or was reverted.

2. **Check git status:**
   ```bash
   git diff neural_vm/vm_step.py | grep _set_layer6_relay_heads
   ```
   Should show the function call is commented out.

3. **Check specific head:**
   ```python
   from neural_vm.vm_step import AutoregressiveVM, set_vm_weights
   model = AutoregressiveVM()
   set_vm_weights(model)

   # Check Layer 6 Head 2 has JMP relay config
   attn6 = model.blocks[6].attn
   HD = attn6.head_dim
   base = 2 * HD

   # Should have query weights for MARK_PC and HAS_SE
   w_q = attn6.W_q[base:base+HD, :]
   print(f"Head 2 queries PC: {w_q[:, BD.MARK_PC].abs().max().item()}")
   print(f"Head 2 queries HAS_SE: {w_q[:, BD.HAS_SE].abs().max().item()}")
   ```

4. **Verify AX_CARRY path manually:**
   See `verify_ax_carry_at_ax_marker.py` for detailed checking code.

## Performance Notes

- **Model loading**: 1-2 minutes (one-time cost)
- **Single test execution**: 0.1-0.5 seconds
- **Weight configuration checks**: < 30 seconds
- **100-test suite**: 5-10 minutes
- **1000-test suite**: 30-60 minutes

## References

- **Fix documentation**: `docs/ACTUAL_FIX_AX_CARRY.md`
- **Summary**: `FIX_SUMMARY.md`
- **Original (incorrect) diagnosis**: `docs/FIX_AX_CARRY_ISSUE.md`
