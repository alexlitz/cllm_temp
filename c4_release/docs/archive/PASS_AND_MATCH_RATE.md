# Pass and Match Rate Report

## Summary

**Fast VM (Speculator) Pass Rate**: **100%** ✓
**Neural Model Match Rate**: **0%** ✗

## Detailed Analysis

### Fast VM (Speculator) Performance

The Fast VM executes all programs correctly:

```
Fast VM Pass Rate: 100%
- Correctly computes arithmetic
- Handles all opcodes
- Returns accurate results
```

**Example Results**:
```
int main() { return 42; }        → 42  ✓
int main() { return 100 + 200; } → 300 ✓
int main() { return 5 + 3; }     → 8   ✓
int main() { return 10 * 2; }    → 20  ✓
int main() { return 20 / 4; }    → 5   ✓
```

### Neural Model Match Rate

The Neural Model **never** matches the Fast VM:

```
Neural Model Match Rate: 0%
- Returns ('', 0) for ALL programs
- Halts after 2 steps regardless of program
- Cannot execute autonomously
```

**Example Results**:
```
int main() { return 42; }        → ('', 0)  ✗ (expected 42)
int main() { return 100 + 200; } → ('', 0)  ✗ (expected 300)
int main() { return 5 + 3; }     → ('', 0)  ✗ (expected 8)
int main() { return 10 * 2; }    → ('', 0)  ✗ (expected 20)
int main() { return 20 / 4; }    → ('', 0)  ✗ (expected 5)
```

## Validation Evidence

### From test_varied_validation.py

```
Running 10 tests with 10% validation:

  [ 1] VALIDATION ERROR: int main() { return 0; }
       Neural model validation failed!
       Fast VM result: 0
       Neural VM result: ('', 0)

  [ 2] PASS: int main() { return 1; } → 1
  [ 3] PASS: int main() { return 42; } → 42
  [ 4] PASS: int main() { return 100; } → 100
  [ 5] PASS: int main() { return 5 + 3; } → 8
  [ 6] PASS: int main() { return 10 * 2; } → 20

  [ 7] VALIDATION ERROR: int main() { return 15 - 5; }
       Neural model validation failed!
       Fast VM result: 10
       Neural VM result: ('', 0)

  [ 8] PASS: int main() { return 20 / 4; } → 5
  [ 9] PASS: int main() { return 17 % 5; } → 2
  [10] PASS: int main() { return 2 + 3 * 4; } → 14

Results:
  Passed: 8/10 (validation skipped on these)
  Failed validation: 2/10 (validation happened, both failed)
  Errors: 0

Match rate: 0/2 = 0%
```

### From test_validation_enabled.py

```
Running: int main() { return 42; }

ValidationError: Neural model validation failed!
  Fast VM result: 42
  Neural VM result: ('', 0)
  Bytecode length: 6
  Total validations: 1
  Total mismatches: 1

Match rate: 0/1 = 0%
```

### From test_detailed_trace.py

```
Generated 105 tokens:

PC = 0x00001000 (4096)     ← Wrong (should be 0)
AX = 0x00001000 (4096)     ← Wrong (should be 0)
SP = 0x00001000 (4096)     ← Wrong (should be 0x10000)
BP = 0x00001000 (4096)     ← Wrong (should be 0x10000)

--- STEP_END (step 1) ---

PC = 0x00000010 (16)
AX = 0x00000000 (0)        ← Never set to 42!
SP = 0x000000f8 (248)
BP = 0x00000000 (0)

--- STEP_END (step 2) ---

PC = 0x00000000 (0)
AX = 0x00000000 (0)        ← Still 0!

*** HALT ***

Exit code: 0               ← Wrong (should be 42)
```

## Why This Matters

### Before Validation Was Enabled

**Tests reported**: ✓ 100% pass rate
**Reality**:
- Fast VM: 100% working
- Neural VM: 0% working
- **False positive** - tests didn't validate neural model

### After Validation Enabled

**Tests report**: ✗ Validation failures
**Reality**:
- Fast VM: 100% working ✓
- Neural VM: 0% working (correctly detected) ✓
- **Accurate testing** - failures caught

## Test Suite Behavior

With current configuration (`validate_neural=True`, `validation_sample_rate=0.1`):

```
┌─────────────────────────────────────────────────┐
│ Test Suite (100 tests)                          │
├─────────────────────────────────────────────────┤
│ ~90 tests: Fast VM only                         │
│   Result: All pass (Fast VM works)              │
│                                                  │
│ ~10 tests: Validated against Neural VM          │
│   Result: ALL FAIL (0% match rate)              │
│   Action: ValidationError raised                │
│   Status: TEST SUITE FAILS ✗                    │
└─────────────────────────────────────────────────┘
```

## Statistics

| Metric | Value | Status |
|--------|-------|--------|
| **Fast VM Pass Rate** | 100% | ✓ Working |
| **Neural Model Match Rate** | 0% | ✗ Broken |
| **Validation Sample Rate** | 10% | ✓ Configured |
| **Expected Failure Rate** | 100% of validated | ✗ All fail |

## What This Means

1. **Fast VM is reliable**: Use speculator for actual execution ✓
2. **Neural model is broken**: Cannot execute programs autonomously ✗
3. **Validation works**: Correctly catches the broken neural model ✓
4. **Tests are accurate**: Fail when they should (no false positives) ✓

## Next Steps

To achieve **100% match rate**:

1. Fix neural model's initial state generation
2. Fix instruction execution logic (JSR, IMM, LEV, EXIT, etc.)
3. Fix HALT generation (currently triggers prematurely)
4. Fix multi-byte arithmetic (currently only byte 0 works)

Once fixed:
- Neural model will return correct results
- Match rate will increase to 100%
- Tests will pass (both Fast VM and Neural VM return correct results)
- Validation becomes a correctness guarantee

## Current Recommendation

**Keep validation enabled** with current settings:
- `validate_neural=True`
- `validation_sample_rate=0.1`

This ensures:
- Fast execution (90% Fast VM only)
- Neural model checking (10% validated)
- Accurate test results (failures detected)
- Foundation for fixing neural model (validation provides ground truth)
