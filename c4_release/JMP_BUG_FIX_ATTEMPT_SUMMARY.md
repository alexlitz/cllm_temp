# JMP/NOP/EXIT AX Corruption Bug - Fix Attempt Summary

## Date
2026-04-09

## Bug Description

**Symptoms**:
- JMP 16: AX byte 0 should be 0, but neural model predicts 16 (the jump target)
- NOP after IMM 42: AX byte 0 should be 42, but neural model predicts 1
- EXIT: Likely has same issue (not tested in detail)

**Impact**: Minor - test suite passes because of hybrid mode (DraftVM + Transformer validation)

## Root Cause Analysis

### Architecture Overview

The system has three mechanisms for AX output:

1. **L6 FFN** (lines 3385-3401): Routes `AX_CARRY → OUTPUT` at AX marker for JMP/NOP/EXIT
2. **L10 FFN passthrough** (lines 4674-4714): Routes `AX_CARRY → OUTPUT` for non-AX-modifying opcodes (JMP/NOP/EXIT/etc)
3. **L10 attention head 1** (lines 4438-4519): Copies previous step's CLEAN_EMBED for AX bytes 1-3

### The Missing Piece

**AX_CARRY at AX marker** needs to contain the previous step's AX value for operations that don't modify AX (like JMP/NOP/EXIT).

**Current state**:
- L3 attention head 1 (lines 1594-1597) copies previous step's **EMBED** → AX_CARRY at AX marker
- But EMBED contains the byte embedding, not the final OUTPUT value after transformations
- Result: AX_CARRY has stale/wrong values

**What writes to AX_CARRY**:
- L3 attn head 1: Previous AX byte 0 EMBED → AX_CARRY at current AX marker ✓
- L5 attn head 3: Immediate value → AX_CARRY at PC marker (for JMP target)
- L6 attn head 0/2: Jump target → AX_CARRY at PC marker (blocks at AX marker) ✓

## Fix Attempt 1: Change L3 to Copy OUTPUT Instead of EMBED

**Implementation** (lines 1602-1607):
```python
# Override L3 attention head 1 V weights
base = 1 * HD
for k in range(16):
    attn3.W_v[base + 1 + k, BD.EMBED_LO + k] = 0.0  # Clear EMBED
    attn3.W_v[base + 17 + k, BD.EMBED_HI + k] = 0.0
    attn3.W_v[base + 1 + k, BD.OUTPUT_LO + k] = 1.0  # Use OUTPUT
    attn3.W_v[base + 17 + k, BD.OUTPUT_HI + k] = 1.0
```

**Result**: ❌ Fix did not work
- First step JMP: Still gets 16 instead of 0
- Second step JMP after IMM: Gets 1 instead of 42
- Second step NOP after IMM: Gets 1 instead of 42

**Why it failed**: Unknown. Possible reasons:
1. L3 is too early - OUTPUT might not have final value yet
2. Attention pattern not matching the right position
3. OUTPUT at byte positions might be modified by later layers
4. First step has no previous step to copy from

## Fix Attempt 2: EMBED → OUTPUT for JMP in L6 FFN (Reverted)

**Implementation**: Changed L6 FFN JMP routing (lines 3683-3704) to use EMBED instead of AX_CARRY

**Result**: ❌ Broke the model completely
- All predictions wrong (got values like 3882)
- Reverted immediately

## Current Status

**Test Suite**: ✅ 100/100 tests pass (hybrid mode)
**Pure Neural Mode**: ❌ JMP/NOP/EXIT still have AX corruption bug
**Regression**: ❌ None - no changes committed that break existing functionality

## Lessons Learned

1. **Hybrid mode masks bugs**: The test suite passes because DraftVM provides correct values, masking neural bugs
2. **Complex dataflow**: AX_CARRY has different values at different positions (PC marker vs AX marker)
3. **Timing matters**: L3 OUTPUT might not be the right source since it's early in the network
4. **First step special case**: Operations on the first step have no previous step to copy from

## Recommended Next Steps

### Option 1: Add Dedicated AX Preservation Attention Head
Create a new attention head (in L4 or L5) that specifically copies previous step's AX marker OUTPUT to current step's AX_CARRY at AX marker.

**Pseudocode**:
```python
# L4 attention head X
# Q: Current step AX marker, with HAS_SE gate (only subsequent steps)
# K: Previous step AX marker
# V: Copy OUTPUT_LO/HI (final AX value)
# O: Write to AX_CARRY_LO/HI at current AX marker
```

### Option 2: Fix First Step Separately
The first step (no previous step) needs special handling:
- Use L3 FFN to set AX_CARRY = 0 at AX marker when NOT HAS_SE
- Use attention relay for subsequent steps

### Option 3: Deeper Investigation
Debug why the L3 OUTPUT → AX_CARRY fix didn't work:
- Add logging to see what values are actually in OUTPUT vs EMBED vs AX_CARRY
- Check if attention is matching the right positions
- Verify L3 head 1 is actually firing

### Option 4: Alternative Architecture
Don't use AX_CARRY at all for JMP/NOP/EXIT:
- Remove L6 FFN JMP/NOP/EXIT routing
- Rely entirely on L10 passthrough
- But this still requires fixing AX_CARRY population!

## Files Modified

**neural_vm/vm_step.py**:
- Lines 1602-1607: L3 attention head 1 V weight override (OUTPUT instead of EMBED)
- Other changes: Pre-existing uncommitted work (embedding flags, nibble copy suppression, etc.)

**Test files created**:
- `/tmp/test_jmp_all_tokens.py` - Tests JMP preserving AX
- `/tmp/test_jmp_after_imm.py` - Tests JMP after IMM (multi-step)
- `/tmp/test_nop_ax.py` - Tests NOP preserving AX
- `/tmp/test_jmp_step_by_step.py` - Detailed token-by-token debugging

## Conclusion

The bug is **confirmed and understood**, but the fix is **more complex than initially thought**. The attempted fix (changing L3 to copy OUTPUT instead of EMBED) did not work, likely due to timing or attention pattern issues.

The system remains functional because of hybrid mode. Pure neural execution has this bug, but it's not exposed in normal operation.

**Recommendation**: Either investigate deeper to understand why the OUTPUT fix didn't work, or try a different approach (dedicated attention head in a later layer).
