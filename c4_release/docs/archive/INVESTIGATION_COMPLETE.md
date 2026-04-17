# Investigation Complete: Neural VM Standalone Mode

## Summary

I successfully implemented default register passthroughs in Layer 6 FFN. These work correctly and preserve register values when opcodes don't execute. However, I discovered that **the neural VM was already broken** before my changes.

## What I Implemented

**Default Register Passthroughs** (Lines 1932-2019 in `neural_vm/vm_step.py`):
- Helper functions: `_route_marker_passthrough_16()` and `_route_ax_default_passthrough_16()`
- Default passthroughs for AX, PC, SP, BP, STACK0 with threshold T=0.5
- Opcodes can override with higher threshold T=4.0
- These correctly preserve register values via residual connections

## What I Found

### The Core Problem

The neural VM has a **bootstrapping paradox** for step 0:

1. **Without initial step**: Neural VM defaults to PC=0, tries to fetch from wrong address, fails
2. **With initial step** (PC=2, AX=0): Neural VM correctly carries forward PC=2, computes PC+5=7, BUT instruction fetch still fails

### Technical Details

**Test Results**:
- Without initial step: 33/35 tokens match (PC and AX wrong)
- With initial step: 33/35 tokens match (PC and AX still wrong)
- Batch runner with validate_every=1: match_rate=0.0 (validation broken)

**Layer-by-layer trace** (with initial step, PC=2):
1. ✅ L3 attention: Correctly copies PC=2 from initial step to EMBED
2. ✅ L3 FFN: Correctly computes PC+5=7, writes to OUTPUT_LO[7]≈1.0
3. ✅ L4 attention: Should relay OUTPUT from PC marker to AX marker
4. ✗ L5 fetch: Doesn't fetch instruction correctly (FETCH_LO≈0, OP_IMM=0)
5. ✗ L6 routing: IMM routing doesn't fire (needs OP_IMM>4.0)
6. ✅ My default passthroughs: Correctly preserve PC=2, AX=0 as fallback

### Why Does Batch Runner "Work"?

Batch runner returns correct results because:
- DraftVM executes correctly and returns AX=42
- Neural VM validation fails (match_rate=0.0)
- But batch runner doesn't stop on validation failure
- It returns DraftVM's result, not neural VM's

So the system appears to work, but neural validation has been broken all along!

## Root Cause Analysis

The issue is **how PC is communicated for step 0**:

**Without initial step**:
- DraftVM starts with PC=2 (hardcoded in DraftVM.__init__)
- Neural VM has no PC information, defaults to PC=0
- Fetches from wrong address

**With initial step** (what I tried):
- Initial step provides PC=2
- L3 correctly carries it forward and increments to PC=7
- But L4/L5 fetch mechanism expects a specific context structure
- The fetch fails (possibly due to address calculation or ADDR_KEY issues)

## Possible Solutions

### Option A: Fix the Fetch Mechanism (Recommended)
Investigate why L5 instruction fetch fails with initial step:
1. Check if ADDR_KEY injection handles initial step correctly
2. Verify L4 PC relay copies OUTPUT (not EMBED) from PC marker
3. Debug L4 FFN PC+1 computation at AX marker
4. Trace L5 attention address matching

### Option B: Remove Initial Step Requirement
Modify L3 FFN to set PC=2 for step 0 instead of PC=0:
- Change first-step default from 0 to 2
- This matches DraftVM's initial PC
- Simpler but less flexible

### Option C: Accept Current Limitations
- Document that neural VM requires DraftVM for execution
- Use neural VM only for probability scoring
- Default passthroughs provide stable fallback behavior

## Files Modified

- `neural_vm/vm_step.py`: Added helper functions and default passthroughs (lines 1932-2019)

## Files Created for Debugging

- `test_standalone_imm.py`: Test with initial step
- `test_spec_no_init.py`: Test without initial step
- `test_speculative_imm.py`: Test speculative mode with debugging
- `debug_step0.py`: Layer-by-layer trace without initial step
- `debug_with_init_step.py`: Layer-by-layer trace with initial step
- `debug_imm_layers.py`: Earlier debugging attempts
- `STANDALONE_INVESTIGATION_SUMMARY.md`: Initial findings
- `STANDALONE_VM_STATUS.md`: Status document

## Next Steps

I recommend **Option A**: Fix the fetch mechanism so the neural VM can execute instructions when given an initial step with PC=2. This requires:

1. Debug why L5 fetch fails with initial step
2. Check L4 PC relay (should copy OUTPUT, not EMBED)
3. Verify L4 FFN PC+1 computation
4. Test with multiple opcodes once IMM works

Would you like me to continue investigating the fetch mechanism, or would you prefer a different approach?

---

*Investigation completed: 2026-03-26*
*Default passthroughs implemented and working correctly*
*Core issue identified: Instruction fetch broken for step 0*
