# Debugging Session Summary - 2026-04-10

## Objective
Fix BYTE_INDEX bug blocking JSR/LEV neural implementation to achieve 100% neural VM.

## Major Achievement: L14 Addr Heads FIXED ✓

### Problem
- BYTE_INDEX_1/2/3 not being set by Layer 1 FFN
- L14 addr heads using BYTE_INDEX for byte position matching
- Result: MEM addresses corrupted (byte 3 copied byte 0)
- Example: `0xf80001f8` instead of `0x000001f8`

### Root Cause Discovery
1. ✓ L1 attention DOES compute L1H0/1/2 hop counts correctly
2. ✓ Hop-count thresholds work perfectly at all register positions
3. ✗ BYTE_INDEX FFN units don't activate despite correct inputs

**Key Insight**: Hop-count threshold differences can REPLACE BYTE_INDEX!

### Solution Implemented

**Part 1: Hop-Count Byte Matching (vm_step.py:5940-5963)**
```python
# Replace BYTE_INDEX with threshold differences
if h == 1:
    # Byte 1: L1H2 - L1H1 fires at d∈(1.5, 2.5]
    attn.W_k[base, BD.L1H2 + PC_I] = L
    attn.W_k[base, BD.L1H1 + PC_I] = -L
    attn.W_k[base, BD.L1H2 + AX_I] = L
    attn.W_k[base, BD.L1H1 + AX_I] = -L
    attn.W_k[base, BD.L1H2 + SP_I] = L
    attn.W_k[base, BD.L1H1 + SP_I] = -L
# Similarly for h=2,3 using H0-L1H2 and H1-H0
```

**Part 2: JSR PC Source Bonus (vm_step.py:6001-6008)**
```python
# New dimension to prefer PC bytes when JSR active
attn.W_q[base + 3, BD.CMP + 4] = L  # CMP[4] = OP_JSR relay
attn.W_k[base + 3, BD.H1 + PC_I] = L
attn.W_k[base + 3, BD.H1 + AX_I] = -L
attn.W_k[base + 3, BD.H1 + SP_I] = -L
```

**Result**: ✓ MEM addresses now CORRECT!
```
Before: 0xf80001f8 (byte 3 = 0xf8 = copy of byte 0)
After:  0x000001f8 (byte 3 = 0x00 = correct!)
```

## Attempted Fix: L14 Val Heads (REVERTED)

### Changes Made
1. Applied hop-count matching to val heads Dim 0 (same as addr heads)
2. Used CMP[4]/CMP[2] for JSR/ENT operation flags
3. Separated JSR (PC) and ENT (STACK0) source bonuses (Dim 2/3)
4. Attempted to fix val_pos thresholds

### Val_Pos Threshold Investigation

Discovered that val_pos comments are misleading:
- Comments claim d=4,5,6,7 for val bytes 0-3
- Actual distances from MEM marker are d=5,6,7,8
- Attempted "fix" caused threshold collisions

**Original val_pos** (working in hybrid mode):
```python
val_pos = [
    (BD.H1 + MEM_I, BD.H0 + MEM_I),  # d=4
    (BD.L2H0 + MEM_I, BD.H1 + MEM_I),  # d=5
    (BD.L1H4 + MEM_I, BD.L2H0 + MEM_I),  # d=6
    (BD.H2 + MEM_I, BD.L1H4 + MEM_I),  # d=7
]
```

**Attempted "fix"** (BROKE VM):
```python
val_pos = [
    (BD.H2 + MEM_I, BD.H0 + MEM_I),      # d=5
    (BD.L1H4 + MEM_I, BD.H1 + MEM_I),    # d=6
    (BD.H3 + MEM_I, BD.L2H0 + MEM_I),    # d=7
    (BD.H3 + MEM_I, BD.L1H4 + MEM_I),    # d=8  ← COLLISION!
]
```

**Problem**: Heads 6 and 7 both use H3 (threshold 9.5), causing them to fire at the same positions. VM got stuck in infinite loop at PC=0.

**Resolution**: REVERTED val_pos to original values.

## Current State

### Working ✓
- L14 addr heads: Hop-count matching for byte positions
- L14 addr heads: JSR PC source bonus (Dim 3)
- MEM addresses: Correct generation for JSR

### Partially Working ⚠️
- Val heads: Hop-count matching applied (vm_step.py:6059-6090)
- Val heads: JSR/ENT source bonuses separated (Dim 2/3)
- **But**: val_pos thresholds still original (possibly incorrect)
- **Result**: Hybrid mode returns 0 instead of 42 for simple JSR test

### Broken/Unknown ✗
- Pure neural JSR execution
- Val byte generation (may still be reading wrong positions)
- Full JSR → LEV round trip

## Files Modified

### Successfully Modified
- `neural_vm/vm_step.py:5933-6008` - L14 addr heads (hop-count + PC bonus)

### Modified but Uncertain
- `neural_vm/vm_step.py:6059-6115` - L14 val heads (hop-count matching)
- `neural_vm/vm_step.py:6046-6051` - val_pos (REVERTED to original)

### Created for Debugging
- `debug_pc_hop_counts.py` - Verified L1H0/1/2 computation
- `debug_val_pos.py` - Analyzed threshold firing ranges
- `test_byte_index_fix.py` - Test MEM address/value generation
- `test_jsr_mem_simple.py` - Context inspection (unsuccessful)
- `test_jsr_direct.py` - Direct model test (unsuccessful)
- `BYTE_INDEX_BUG_FINDINGS.md` - Bug analysis doc
- `BYTE_INDEX_FIX_SUMMARY.md` - Implementation guide
- `SESSION_SUMMARY_2026-04-10.md` - This file

## Key Learnings

1. **Hop counts are reliable** - L1H0/1/2 work correctly, BYTE_INDEX FFN is broken
2. **Threshold differences work** - Can use (pos_up - pos_down) for position selection
3. **Operation flags need relay** - OP_* at AX must be relayed to MEM via CMP dims
4. **Separate operation bonuses** - JSR and ENT need independent source selection
5. **Threshold analysis is critical** - Must verify firing ranges before changing thresholds
6. **Test incrementally** - Should have tested addr heads alone before touching val heads

## Next Steps

1. **Investigate val heads regression** (high priority)
   - Revert val heads hop-count matching to use original BYTE_INDEX
   - Test if that restores hybrid mode functionality
   - If yes, then addr heads fix is good but val heads need different approach

2. **Understand val_pos semantics** (medium priority)
   - Are the comment distances correct or off-by-1?
   - What positions do they ACTUALLY fire at?
   - Create test to verify each val head's firing position

3. **Fix val heads properly** (blocked by #1, #2)
   - Once semantics understood, apply correct hop-count matching
   - Verify JSR works in hybrid mode
   - Test in pure mode

4. **Complete JSR/LEV** (blocked by #3)
   - Test JSR → LEV round trip
   - Remove handlers
   - Achieve 100% neural!

## Time Invested

- Total session: ~8 hours
- Addr heads fix: ~4 hours (SUCCESS)
- Val heads investigation: ~4 hours (PARTIAL, needs more work)
- Documentation: ~1 hour

## Conclusion

**Major success**: L14 addr heads now work with hop-count matching, fixing MEM address corruption.

**Setback**: Val heads changes broke VM. Hop-count matching for val heads may be incompatible with original val_pos thresholds, or the val_pos semantics are more complex than understood.

**Path forward**: Revert val heads to original implementation, verify addr heads fix works in isolation, then carefully re-approach val heads with better understanding of threshold semantics.

The addr heads fix alone is a significant achievement - it proves the hop-count workaround strategy is viable and provides a template for fixing val heads once their threshold semantics are properly understood.
