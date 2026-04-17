# JSR Handler Removal Investigation - Session Summary

## Objective
Remove the JSR (Jump Subroutine) handler to achieve 100% neural execution of the C4 VM.

## Work Completed

### 1. Code Modifications (`neural_vm/vm_step.py`)

Added OP_JSR flag relays to three critical attention heads:

**L5 Head 7 - First-Step CODE→PC Relay** (lines 3272, 3292):
```python
attn.W_v[base + v_offset, BD.OP_JSR] = 1.0; v_offset += 1  # V matrix
attn.W_o[BD.OP_JSR, base + o_offset] = 1.0; o_offset += 1  # O matrix
```

**L5 Head 6 - Non-First-Step CODE→PC Relay** (lines 3211, 3231):
```python
attn.W_v[base + v_offset, BD.OP_JSR] = 1.0; v_offset += 1  # V matrix
attn.W_o[BD.OP_JSR, base + o_offset] = 1.0; o_offset += 1  # O matrix
```

**L6 Head 5 - PC→AX Relay** (lines 3714, 3735, 3752-3759):
```python
attn.W_v[base + 3, BD.OP_JSR] = 1.0   # V matrix
attn.W_o[BD.OP_JSR, base + 3] = 1.0   # O matrix
# FETCH positions shifted 17→18 to accommodate OP_JSR insertion
```

### 2. Verification Work

✅ **Embeddings Verified**: OP_JSR correctly set at ADDR_KEY=2 for JSR opcode bytes
✅ **ADDR_KEY Verified**: Immediate byte 0 correctly at ADDR_KEY=3
✅ **L5 Head 7 Weights Verified**: OP_JSR present in V matrix at expected position
✅ **Test Bug Fixed**: Corrected PC calculation (25→26 for instruction 3)

### 3. Testing Results

| Test | Target PC | Result | Exit Code |
|------|-----------|--------|-----------|
| `test_jsr_simple.py` | 25 (wrong) | FAIL | 0 |
| `test_jsr_final.py` | 25 (wrong) | FAIL | 0 |
| `test_jsr_corrected.py` | 26 (correct) | FAIL | 0 |
| `test_jsr_with_handler.py` | 26 | FAIL | 0 |
| `test_jsr_handler_fixed.py` | 26 | **RUNNING** | TBD |

**Key Finding**: All neural tests failed with PC=10 (normal increment) instead of PC=26 (jump target).

## Neural JSR Execution Architecture

The first-step JSR path should execute as follows:

```
1. L5 Head 2 (Opcode Fetch):
   - Queries PC marker (MARK_PC=1, HAS_SE=0)
   - Matches CODE byte at ADDR_KEY=2 (opcode position)
   - Copies CLEAN_EMBED_LO/HI → OPCODE_BYTE at PC marker

2. L5 Head 3 (Immediate Fetch):
   - Queries PC marker
   - Matches CODE byte at ADDR_KEY=3 (immediate byte 0)
   - Copies immediate value → FETCH (amplified 40x)

3. L5 FFN (JSR Decode) - lines 3383-3393:
   - Detects: OPCODE_BYTE=(3,0) + MARK_PC + NOT HAS_SE
   - Writes: TEMP[0] ≈ 5.0 (IS_JSR flag)

4. L6 FFN (PC Override) - lines 7451-7486:
   - Detects: TEMP[0] > 4.0 at PC marker
   - Cancels: OUTPUT (PC+5)
   - Writes: FETCH → OUTPUT (jump target)
```

## Critical Findings

### Finding 1: Neural JSR Completely Non-Functional

**Evidence**:
- Token-by-token trace shows PC=10 (next instruction)
- Expected PC=26 (jump target)
- No evidence of TEMP[0] being set or PC override activating

**Implication**: Either L5 FFN decode isn't firing OR L6 FFN override isn't detecting TEMP[0].

### Finding 2: First Handler Test Was Invalid

**Issue**: `test_jsr_with_handler.py` attempted to override non-existent `_init_handlers()` method.

**Result**: Handler was never actually enabled, test was invalid.

**Fix**: Created `test_jsr_handler_fixed.py` with proper `__init__` override.

### Finding 3: OP_JSR Relays Are For Multi-Step, Not First-Step

**Discovery**: The OP_JSR relays I added (L5 heads 6/7, L6 head 5) are for **subsequent steps** (when HAS_SE=1), not the primary first-step execution path.

**First-Step Path**: Uses L5 FFN opcode decode (lines 3383-3393), which writes directly to TEMP[0] based on OPCODE_BYTE matching.

**Implication**: The relays are architectural additions for completeness but won't fix the current JSR failure.

## Debugging Strategy

To isolate the failure point, we need to check:

1. **Is OPCODE_BYTE being set?**
   - L5 Head 2 should fetch opcode (3,0) to OPCODE_BYTE at PC marker
   - Check: After L5, does PC marker have OPCODE_BYTE_LO=3, OPCODE_BYTE_HI=0?

2. **Is TEMP[0] being written?**
   - L5 FFN unit should detect JSR and write TEMP[0] ≈ 5.0
   - Check: After L5 FFN, does PC marker have TEMP[0] > 4.0?

3. **Is L6 FFN override activating?**
   - L6 FFN should detect TEMP[0] > 4.0 and override PC
   - Check: After L6 FFN, does PC marker OUTPUT = 26 (not 10)?

4. **Is FETCH being set?**
   - L5 Head 3 should fetch immediate byte (26) to FETCH
   - Check: After L5, does PC marker have FETCH_LO byte 0 = 26?

## Test Files Created

### Diagnostic Scripts:
- `check_jsr_addr_keys.py` - Verify ADDR_KEY values ✅
- `inspect_l5_head7_weights.py` - Verify L5 head 7 has OP_JSR ✅
- `diagnose_jsr_detailed.py` - Check PC value after first step ✅
- `diagnose_jsr_conditions.py` - Check L5 layer outputs (incomplete - needs generation)
- `trace_jsr_step_by_step.py` - Token-by-token PC tracking ✅

### Test Scripts:
- `test_jsr_simple.py` - Basic JSR test (wrong PC=25) ❌
- `test_jsr_final.py` - Full JSR test (wrong PC=25) ❌
- `test_jsr_corrected.py` - Fixed PC=26 ❌
- `test_jsr_with_handler.py` - Handler test (broken patch) ❌
- `test_jsr_handler_fixed.py` - Proper handler test ⏳ RUNNING

## Next Steps

1. **Wait for `test_jsr_handler_fixed.py`** to complete
   - If passes: JSR handler logic is sound, issue is neural-specific
   - If fails: Fundamental issue with JSR execution (deeper problem)

2. **If handler works**, create layer-by-layer activation dump:
   ```python
   # Generate first step with intermediate layer captures
   # Check after L5: OPCODE_BYTE, TEMP[0], FETCH
   # Check after L6: PC OUTPUT value
   ```

3. **Identify specific failure point** in the 4-step chain:
   - Opcode fetch → Immediate fetch → TEMP[0] decode → PC override

4. **Fix the broken component** and re-test

5. **Once neural JSR works**, remove handler from `run_vm.py:242`

## Documentation Created

- `JSR_NEURAL_FIXES.md` - Initial documentation of relay additions
- `JSR_INVESTIGATION_SUMMARY.md` - This file

## Conclusion

Significant progress was made in adding the OP_JSR relay infrastructure, but the core first-step JSR execution remains non-functional. The issue appears to be in the L5 FFN opcode decode or L6 FFN PC override chain, not in the relay mechanisms. Further debugging requires instrumentation of the model's forward pass to inspect intermediate activations.

**Current Status**: Waiting for handler test to determine if the issue is neural-specific or a deeper JSR execution problem.

**Estimated Time to Fix** (if neural-specific): 2-4 hours once we identify the specific broken component.

**Blocker**: CPU execution is extremely slow (~1-2 hours per test). GPU access would enable 10-100x faster iteration.
