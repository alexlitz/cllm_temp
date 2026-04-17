# Strict Neural Validation Results

## Summary

Created `neural_vm/tests/test_strict_neural_predictions.py` - tests that **actually validate transformer predictions**, not DraftVM execution.

**Result: 3/9 tests pass** (33% neural accuracy)

## Test Results

###  ✅ PASSING (3/9)
1. **NOP** - Correctly increments PC from 0 to 8
2. **IMM 0** - Correctly sets AX=0
3. **JMP 8** - Correctly sets PC=8

### ❌ FAILING (6/9)
1. **IMM 42** - Predicts AX=0 instead of 42
   - Token 6 (AX_b0): expected=42, predicted=0
2. **IMM 255** - Predicts AX=0 instead of 255
   - Token 6 (AX_b0): expected=255, predicted=0
3. **JMP 16** - Predicts PC=8 instead of 16
   - (Sequential increment instead of jump)
4. **EXIT** - Wrong END token
   - (Likely emits STEP_END instead of HALT)
5. **LEA 8** - Doesn't compute AX correctly
   - (Should set AX=BP+8)
6. **ADD (5+3)** - Addition fails
   - (Multi-step operation involving PSH)

## Key Findings

### Pattern 1: IMM Works for 0, Fails for Non-Zero
- IMM 0: ✓ Works
- IMM 42: ✗ Predicts 0
- IMM 255: ✗ Predicts 0

**Hypothesis**: The neural weights set AX=0 as default, but don't load the immediate value.

### Pattern 2: JMP Works for 8, Fails for 16
- JMP 8: ✓ Works (sequential PC)
- JMP 16: ✗ Predicts 8 (sequential)

**Hypothesis**: JMP 8 accidentally works because it matches the sequential PC increment. JMP to any other address fails.

### Pattern 3: Multi-Step Operations Fail
- NOP (single step): ✓ Works
- ADD (multi-step): ✗ Fails

**Hypothesis**: Later steps depend on earlier state that wasn't set correctly.

## Why Previous Tests Passed

The existing test suite (`test_opcodes_fast.py`) uses:
```python
return [vm.ax for vm in self.draft_vms]  # ← DraftVM result!
```

So tests pass because:
1. DraftVM executes correctly (Python interpreter)
2. Transformer validates tokens (but failures are ignored)
3. Final result comes from DraftVM
4. Tests check DraftVM result → always pass

## Path Forward

### Immediate Priorities
1. **Fix IMM** - Load immediate value into AX
2. **Fix JMP** - Jump to target PC
3. **Fix EXIT** - Emit correct HALT token

### Medium Term
4. Fix LEA (load effective address)
5. Fix multi-step operations (ADD, etc.)

### Success Criteria
All tests in `test_strict_neural_predictions.py` must pass.

## Running the Strict Tests

```bash
# Run all strict tests
python -m pytest neural_vm/tests/test_strict_neural_predictions.py -v

# Run specific test
python -m pytest neural_vm/tests/test_strict_neural_predictions.py::TestIMM::test_imm_42 -xvs

# Expected: Tests FAIL until neural weights are fixed
```

## Comparison: Before vs After

### Before Strict Validation
```
test_opcodes_fast.py: 59/59 passed ✓
(But transformer predictions were wrong)
```

### After Strict Validation
```
test_opcodes_fast.py: 59/59 passed ✓
(DraftVM execution)

test_strict_neural_predictions.py: 3/9 passed ✗
(Actual transformer predictions)
```

Now we can see the real bugs!

## Next Steps

1. ✅ Created strict validation tests
2. ✅ Identified failing operations
3. ⏭️ Fix IMM neural weights (highest priority)
4. ⏭️ Fix JMP neural weights
5. ⏭️ Fix EXIT neural weights
6. ⏭️ Continue until all strict tests pass

Once all strict tests pass, the Neural VM will be truly neural - transformer predictions will match DraftVM execution.
