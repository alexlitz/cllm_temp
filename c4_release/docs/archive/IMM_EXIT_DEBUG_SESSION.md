# IMM+EXIT Debugging Session - 2026-04-08

## Summary

Debugged the Neural VM's IMM+EXIT program execution and discovered both fixes and a fundamental pre-existing bug.

## Fixes Applied

### 1. Layer 5 Head 3 HAS_SE Gate (WORKING)

**File**: `neural_vm/vm_step.py` lines 2728-2737

**Issue**: Layer 5 Head 3 lacked HAS_SE blocking, causing it to fire on ALL steps instead of just the first step. This contaminated AX_CARRY_LO with unwanted values during non-first steps.

**Fix**: Added HAS_SE gate with -500.0 weight to block firing when HAS_SE=1:
```python
HAS_SE_GATE = 34
attn.W_q[base + HAS_SE_GATE, BD.HAS_SE] = -500.0
attn.W_k[base + HAS_SE_GATE, BD.CONST] = 5.0
```

**Result**: EXIT now correctly preserves AX during step 1 execution. No more AX_CARRY contamination.

## Verified Working

✅ **PC Update**: PC correctly advances to 0x0a (10) after step 0 and 0x12 (18) after step 1
✅ **HALT Generation**: EXIT instruction successfully produces HALT token
✅ **Program Completion**: Both steps execute without errors or infinite loops
✅ **HAS_SE Flag**: Correctly 0 during step 0, 1 at step 1 STEP_END marker

## Pre-Existing Bug Discovered

### IMM Instruction Produces Wrong AX Value

**Symptom**: IMM 42 produces AX=0x00010000 (65536) instead of AX=0x2a (42)

**Verification**:
- Tested committed version (HEAD): Exit code = 65536 ❌
- Tested with my fixes: Exit code = 65536 ❌
- Both versions have the same bug

**Evidence**:
```
Step 0 AX bytes: [0x00, 0x00, 0x01, 0x00] = 0x00010000 (65536)
Expected:        [0x2a, 0x00, 0x00, 0x00] = 0x0000002a (42)
```

**Value Analysis**:
- 0x00010000 = 2^16 = 65536
- Only byte 2 (zero-indexed) is set to 0x01
- Suggests a byte position error or addressing issue

### Root Cause Investigation

**Addressing Convention Discovery**:

The model was trained with `PC_OFFSET=2` addressing (documented in `constants.py`):
- PC points to immediate byte, NOT opcode
- Opcode is at address `PC - PC_OFFSET`
- First instruction: opcode at address 0, but PC=2

**Layer 5 Fetch Addresses** (first-step):
- Head 2: Fetches opcode from address `PC_OFFSET=2` (line 2665-2676)
  - **Wrong**: Should fetch from address 0 where opcode actually is
  - Fetches byte at address 2 (immediate byte 1) instead
- Head 3: Fetches immediate from address `PC_OFFSET+1=3` (line 2700-2711)
  - **Wrong**: Should fetch from address 1 where immediate byte 0 is
  - Fetches byte at address 3 (immediate byte 2) instead

**Attempted Fix**:
Changed Head 2 to fetch from address 0 and Head 3 from address 1.

**Result**: Model completely broke (infinite loop of 0x01 tokens)

**Conclusion**: The trained weights EXPECT the PC_OFFSET addressing. Correcting the addresses breaks the model because the weights were learned with this "bug" baked in.

### Code Layout

For `IMM 42, EXIT` program:
```
Address | Byte | Meaning
--------|------|------------------
   0    | 0x01 | IMM opcode
   1    | 0x2a | Immediate byte 0 (42)
   2    | 0x00 | Immediate byte 1
   3    | 0x00 | Immediate byte 2
   4    | 0x00 | Immediate byte 3
   5    | 0x00 | Padding byte 0
   6    | 0x00 | Padding byte 1
   7    | 0x00 | Padding byte 2
   8    | 0x26 | EXIT opcode (0x26 = 38)
```

**What Layer 5 fetches** (first-step):
- Head 2 queries address 2 → gets 0x00 (not the opcode!)
- Head 3 queries address 3 → gets 0x00 (not the immediate!)

This explains why IMM doesn't work correctly.

## Open Questions

1. **Why does EXIT generate HALT?** If opcode fetch is broken, how does EXIT know to halt?
   - Hypothesis: Maybe EXIT is hardcoded or uses a different mechanism?

2. **Where does 0x00010000 come from?** Why specifically byte 2 = 0x01?
   - Hypothesis: Could be related to PC=2 being copied somewhere?
   - Or address 2 byte (0x00) being written to wrong position?

3. **Was the model ever tested with IMM?**
   - The recent commits mention JSR fixes, but no IMM fixes
   - Perhaps IMM has never worked correctly?

4. **Can the addressing be fixed during retraining?**
   - Or is this fundamental to the architecture?

## Recommendations

### Short Term
1. **Document the IMM bug** as a known issue
2. **Test other instructions** to see if they have similar addressing bugs
3. **Check if runner handlers compensate** for the wrong AX value

### Long Term
1. **Retrain with correct addressing** (PC_OFFSET=0 or proper opcode/immediate fetch)
2. **Add integration tests** for basic instructions like IMM
3. **Verify all Layer 5 fetch logic** for consistency

## Test Results

All tests from this session:
- ✅ PC update to 0x0a (step 0)
- ✅ PC update to 0x12 (step 1)
- ✅ HALT generation
- ✅ HAS_SE flag behavior
- ❌ IMM sets AX=42 (gets 65536 instead)

## Files Modified

- `neural_vm/vm_step.py`: Layer 5 Head 3 HAS_SE gate fix (lines 2728-2737)

## Test Files Created

- `test_step_structure.py`: Examine step 0 structure and PC values
- `test_has_se_during_step.py`: Verify HAS_SE during generation
- `test_imm_exit_e2e.py`: End-to-end IMM+EXIT test
- `verify_exit_code.py`: Extract and verify exit code
- Various other diagnostic scripts

## Conclusion

**SUCCESS**: Fixed HAS_SE gate contamination, verified PC updates and HALT generation
**DISCOVERY**: Found pre-existing IMM instruction bug (wrong addressing in Layer 5 fetch)
**STATUS**: IMM+EXIT program completes successfully but returns wrong exit code (65536 instead of 42)

The model's basic control flow works (PC, HALT), but instruction execution has addressing issues that prevent correct value propagation.
