# IMM Instruction Bug - Investigation Report

## Problem Statement

The neural VM fails basic `IMM 42; EXIT` test:
- **Expected**: PC=0x0a (10), AX=0x2a (42), exit_code=42
- **Actual**: PC=0x2a (42), AX=0x2a (42), exit_code=16843009 (byte_01 loop)

## Root Cause Analysis

### Layer-by-Layer Trace

**OUTPUT_HI[2] contamination at PC position:**
```
After embed:   OUTPUT_HI[2] = 0.000
After Layer 3: OUTPUT_HI[2] = 0.000  # PC=0x0a written correctly here
After Layer 5: OUTPUT_HI[2] = 0.000
After Layer 6: OUTPUT_HI[2] = 3.000  # BUG: Value appears here!
After Layer 15: OUTPUT_HI[2] = 3.000
```

### Layer 6 FFN Bug Identified

**File**: `neural_vm/vm_step.py:3063`

**Bug**: Threshold T=4.0 was too low for 2-way AND gate

Units activate when: `OP_IMM + MARK_AX > T`

| Position | OP_IMM | MARK_AX | Sum | T=4.0 | T=5.5 |
|----------|--------|---------|-----|-------|-------|
| PC       | 5.0    | 0.0     | 5.0 | ✓ fires (BUG!) | ✗ blocked |
| AX       | 5.0    | 1.0     | 6.0 | ✓ fires | ✓ fires |

**Fix Applied**: Changed `T = 4.0` → `T = 5.5`

```python
# BEFORE:
T = 4.0  # opcode threshold: correct OP ≈ 5 + MARK_AX 1 = 6 > T

# AFTER:
T = 5.5  # opcode threshold: OP(5) + MARK_AX(1) = 6 > 5.5, but OP alone = 5 < 5.5
```

### Result of Fix

✅ **Partial success**: OUTPUT_HI[2] reduced from 8.4 → 3.0
❌ **Still broken**: OUTPUT_HI[2] still contaminating PC (value = 3.0)

## Remaining Issue

Layer 6 FFN still outputs OUTPUT_HI[2]=3.0 at PC position, but:
- No individual FFN units show as activated at PC
- FFN output at PC shows OUTPUT_HI[2]=3.0

**Hypothesis**: There's another set of units in Layer 6 FFN that write to OUTPUT_HI[2] at PC position with different activation conditions.

## Tests Masked the Bug

The 3000+ opcode tests all pass because `AutoregressiveVMRunner` has Python handlers that override neural predictions:
- Handler for IMM: Sets AX to immediate value regardless of neural output
- Handler for EXIT: Extracts exit code from context

**Pure neural execution** (without handlers) has been broken for months.

## Memory Tests Status

Memory tests (SI/LI) cannot work until basic IMM works:
- Memory operations need IMM to load addresses
- Current status: All memory tests fail with exit_code=0x01010101 (byte_01 loop)

## Next Steps

1. **Debug Layer 6 FFN units**: Find which units write OUTPUT_HI at PC
   - Check all 4096 units for activation at PC position
   - Identify activation conditions causing spurious OUTPUT writes
   
2. **Fix activation conditions**: Ensure units only fire at intended positions
   - Add MARK_AX requirement to all OUTPUT routing units
   - Increase thresholds to prevent single-flag activation

3. **Test incrementally**:
   - Verify PC byte prediction = 0x0a
   - Verify AX byte prediction = 0x2a  
   - Verify full IMM+EXIT produces exit_code=42

4. **Run memory tests** after IMM fix

## Files Modified

- `neural_vm/vm_step.py:3063` - Fixed threshold T=4.0 → T=5.5

## Commands for Testing

```bash
# Test basic IMM+EXIT
python3 rebuild_and_test.py

# Test without compaction
python3 test_si_li_direct.py

# Layer-by-layer trace
python3 debug_imm_layers.py
```
