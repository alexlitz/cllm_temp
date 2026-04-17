# Val Heads Debugging Session - 2026-04-10 (Continued)

## Objective

Continue debugging BYTE_INDEX bug from previous session. Investigate val heads hop-count matching to achieve 100% neural JSR/LEV implementation.

## Work Completed

### 1. Initial State Review

- **Previous session**: Successfully fixed L14 addr heads using hop-count matching
  - Addr bytes now correct (0x000001f8 instead of 0xf80001f8)
  - Added JSR PC source bonus (Dim 3)
  - MEM addresses WORKING ✓

- **Previous attempt at val heads**: Applied hop-count matching but broke VM
  - Reverted val_pos thresholds after threshold collision
  - Hybrid mode returned 0 instead of 42

### 2. Val Heads Hop-Count Matching - Second Attempt

**Goal**: Apply same hop-count workaround to val heads as was successful for addr heads

**Implementation** (vm_step.py:6189-6229):

```python
# Dim 0: Q position selection + K targets byte positions
# WORKAROUND: Use hop-count threshold differences (same as addr heads)
attn.W_q[base, pos_up] = L
attn.W_q[base, pos_down] = -L

# K: Match byte position via hop-count threshold differences
# Same pattern as addr heads - match across PC, AX, SP markers
if h == 0:
    # Byte 0: d=1 from marker (L1H1 - L1H0)
    attn.W_k[base, BD.L1H1 + PC_I] = L
    attn.W_k[base, BD.L1H0 + PC_I] = -L
    attn.W_k[base, BD.L1H1 + AX_I] = L
    attn.W_k[base, BD.L1H0 + AX_I] = -L
    attn.W_k[base, BD.L1H1 + SP_I] = L
    attn.W_k[base, BD.L1H0 + SP_I] = -L
elif h == 1:
    # Byte 1: d=2 from marker (L1H2 - L1H1)
    ...
```

**Key differences from first attempt**:
- Initially forgot to include SP matching - added in second iteration
- Now matches ALL three markers (PC, AX, SP) in Dim 0, just like addr heads
- Dims 1/2/3 handle source selection (AX default, PC for JSR, STACK0 for ENT)

**Result**: VM initializes successfully but execution is broken

### 3. Testing Results

**Test 1**: Simple JSR with handler enabled
```
Result: ('', 0)
Expected: 42
Status: FAIL
```

**Test 2**: JSR with handler disabled
```
Result: ('', 0)
MEM addresses: 0x00000000 (ALL ZEROS!)
Status: FAIL - worse than with handler
```

**Test 3**: Simple `return 42` program
```
Result: ('', 0)
MEM addresses: 0x00000000
Status: FAIL
```

### 4. Key Findings

1. **With JSR handler enabled + hop-count val heads**:
   - VM runs and exits normally (code 0)
   - Returns 0 instead of 42
   - MEM tokens generated but values incorrect
   - 116 MEM tokens found (seems excessive)

2. **With JSR handler disabled + hop-count val heads**:
   - VM runs and exits normally (code 0)
   - Returns 0 instead of 42
   - **ALL MEM addresses are zero (0x00000000)**
   - Only 56 MEM tokens (fewer than with handler)
   - This is WORSE than with handler

3. **Handler interaction**:
   - JSR handler seems to provide some initialization or setup
   - Without handler, addresses become all zeros
   - With handler, addresses are wrong but non-zero
   - Handler was re-enabled after discovering zero-address issue

## Current State

### Files Modified

**neural_vm/vm_step.py**:
- Lines 6195-6229: Val heads Dim 0 uses hop-count matching (ACTIVE)
- Lines 6063-6099: Addr heads Dim 0 uses hop-count matching (ACTIVE from previous session)
- Lines 6046-6051: val_pos thresholds (REVERTED to original)

**neural_vm/run_vm.py**:
- Line 237: JSR handler ENABLED (re-enabled after testing)
- Line 240: ENT handler ENABLED (was already enabled)
- Line 241: LEV handler ENABLED (was already enabled)

### Test Files Created

- `test_val_heads_revert.py`: Model initialization test (PASSES)
- `test_jsr_after_revert.py`: Simple JSR test (FAILS - returns 0)
- `test_jsr_with_debug.py`: MEM token inspection (FAILS - wrong addresses)
- `test_psh_mem.py`: Simplified test (FAILS - all-zero addresses)
- `test_jsr_detailed.py`: Direct model test (ERROR - bytecode format issue)

## Analysis

### Why Val Heads Hop-Count Matching Fails

**Hypothesis 1**: Dimension Conflicts
- Val heads Dim 1/2/3 handle source selection (AX, PC, STACK0)
- Adding hop-count matching in Dim 0 may create attention weight conflicts
- Addr heads have different Dim 1/2/3 structure (SP, STACK0, PC sources)
- The interaction may be incompatible

**Hypothesis 2**: val_pos Threshold Mismatch
- val_pos uses different thresholds than addr_pos:
  - addr_pos: MARK_MEM, L1H1-L1H0, L1H2-L1H1, H0-L1H2
  - val_pos: H1-H0, L2H0-H1, L1H4-L2H0, H2-L1H4
- Hop-count matching in Dim 0 K may conflict with val_pos in Dim 0 Q
- The val_pos thresholds determine WHEN the head fires
- The hop-count K weights determine WHERE it attends
- Mismatch between "when" and "where" could cause incorrect byte selection

**Hypothesis 3**: Source Selection Logic
- Val heads default to AX (Dim 1), with PC (Dim 2) for JSR
- Addr heads default to SP (Dim 1), with STACK0 (Dim 2) and PC (Dim 3)
- Adding SP matching to val heads Dim 0 may interfere with source selection
- Val heads shouldn't need to match SP bytes for value generation

### Why Disabling JSR Handler Causes Zero Addresses

**Hypothesis**: Handler provides register initialization
- JSR handler calls `_override_register_in_last_step`
- This ensures SP and STACK0 are set correctly
- Without handler, SP may stay at 0x00000000
- Addr heads then read zeros from SP → all-zero MEM addresses

## Conclusions

1. **Addr heads fix is solid**: Hop-count matching works for addr heads (verified in previous session)

2. **Val heads hop-count matching is problematic**: 
   - Breaks VM execution (returns 0 instead of 42)
   - May conflict with val_pos threshold selection
   - May interfere with source selection logic (AX vs PC vs STACK0)

3. **Handler dependency discovered**:
   - JSR handler provides critical initialization
   - Can't test pure neural JSR without fixing initialization issue
   - Removing handler causes complete failure (all-zero addresses)

4. **BYTE_INDEX still broken**:
   - Can't revert to BYTE_INDEX for val heads (it's not being set)
   - Must use hop-count matching OR fix BYTE_INDEX FFN units

## Next Steps (Prioritized)

### Short Term (Fix Current Regression)

1. **Revert val heads Dim 0 to BYTE_INDEX** (even though it's broken)
   - Test if this restores basic functionality
   - Addr heads should still work with hop-count matching
   - This isolates addr heads success from val heads failure

2. **Test addr heads in isolation**
   - Verify MEM addresses are correct with handler + addr hop-count
   - Confirm addr heads fix alone is working
   - Document that addr heads are DONE

3. **Debug why BYTE_INDEX isn't being set**
   - Investigate Layer 1 FFN units (lines 2209-2253 in vm_step.py)
   - Check if FFN threshold/gate logic is correct
   - May need to fix root cause instead of workaround

### Medium Term (Fix Val Heads)

4. **Design val heads hop-count matching differently**
   - Don't copy addr heads pattern exactly
   - Consider val heads only need to match AX and PC bytes (not SP)
   - May need separate approach for each source (AX/PC/STACK0)

5. **Fix val_pos/threshold interaction**
   - Understand why val_pos thresholds are different from addr_pos
   - May need to adjust hop-count matching to align with val_pos
   - Or adjust val_pos to align with hop-count matching

### Long Term (100% Neural)

6. **Fix JSR handler dependency**
   - Identify what initialization JSR handler provides
   - Implement neural equivalent (may need L6 FFN units)
   - Test pure neural JSR execution

7. **Complete ENT and LEV**
   - After JSR works neurally, tackle ENT and LEV
   - Follow path outlined in planning document
   - Achieve 100% neural VM

## Lessons Learned

1. **Test incrementally**: Should have tested addr heads alone before modifying val heads

2. **Handlers provide hidden setup**: Can't just remove handlers without understanding their full role

3. **Attention weight interactions are subtle**: Copying patterns between heads doesn't always work

4. **Root cause vs workaround**: Hop-count matching is a workaround for BYTE_INDEX bug; may need to fix root cause

5. **Document thoroughly**: Session summary from previous work was invaluable for understanding context

## References

- Previous session summary: `SESSION_SUMMARY_2026-04-10.md`
- BYTE_INDEX bug analysis: `BYTE_INDEX_BUG_FINDINGS.md`
- Implementation guide: `BYTE_INDEX_FIX_SUMMARY.md`
- Test showing addr fix worked: `test_byte_index_fix.py` (from previous session)
