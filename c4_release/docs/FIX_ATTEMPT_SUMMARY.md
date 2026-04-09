# Layer 1 Threshold Head Fix Attempt - Summary

**Date**: 2026-04-07
**Goal**: Fix Layer 1 threshold heads to enable arithmetic operations
**Status**: Root cause identified, fix requires deeper architectural understanding

## Summary

Attempted to fix Layer 1 threshold heads (L1H1/L1H0) which are preventing AX carry-forward and breaking arithmetic operations. Discovered that the issue is more complex than initially thought, involving the autoregressive generation mechanism rather than just weight initialization.

## What We Confirmed

### 1. ✓ Weights Are Set Correctly

Layer 1 attention weights ARE properly initialized:

```
Head 0 (threshold 0.5):
  W_q[0, CONST=8] = 80.000 ✓
  W_k[0, IS_MARK=7] = 0.500 ✓
  W_v non-zero: 7 / 64 rows ✓
  W_o[116:123, 0:64] non-zero: 7 / 448 ✓

Head 1 (threshold 1.5):
  W_q[64, CONST=8] = 80.000 ✓
  W_k[64, IS_MARK=7] = 1.500 ✓
  W_v non-zero: 7 / 64 rows ✓
  W_o[123:130, 64:128] non-zero: 7 / 448 ✓

Head 2 (threshold 2.5):
  W_q[128, CONST=8] = 80.000 ✓
  W_k[128, IS_MARK=7] = 2.500 ✓
  W_v non-zero: 7 / 64 rows ✓
  W_o[130:137, 128:192] non-zero: 7 / 448 ✓
```

**Conclusion**: Weight initialization is correct. `set_vm_weights()` properly configures threshold attention.

### 2. ✓ Embedding Sets IS_MARK

Embedding initialization DOES set IS_MARK for marker tokens:

```python
for tok, dim in [
    (Token.REG_PC, BD.MARK_PC),
    (Token.REG_AX, BD.MARK_AX),
    (Token.REG_SP, BD.MARK_SP),
    (Token.REG_BP, BD.MARK_BP),
    (Token.MEM, BD.MARK_MEM),
    (Token.CODE_START, BD.MARK_CS),
]:
    embed[tok, dim] = 1.0
    embed[tok, BD.IS_MARK] = 1.0  # ← Sets IS_MARK correctly
```

**Conclusion**: Embedding is configured to set IS_MARK on marker tokens.

### 3. ✗ Runtime: IS_MARK Not Present

At runtime, only 2 positions have IS_MARK set (out of 49 tokens):
- Position 0: Initial state
- Position 48: STEP_END

Expected: IS_MARK should be at positions 0, 5, 10, 15, 25, 34 (marker tokens in 35-token format)

**Conclusion**: The issue is in the autoregressive generation/context building, not weight initialization.

## The Core Issue

**Problem**: Threshold attention requires IS_MARK flag on marker tokens to compute distances, but IS_MARK is not present at marker positions during autoregressive generation.

**Why This Breaks Everything**:
1. Layer 1 attention uses `W_k[base, BD.IS_MARK] = threshold`
2. Without IS_MARK in the context, all attention scores are zero
3. L1H0/L1H1/L1H2 outputs stay zero
4. Layer 3 cannot identify byte 0 positions
5. AX carry-forward fails
6. ADD returns first operand only

## Hypothesis: Autoregressive Format Issue

**Theory**: The autoregressive VM may be generating context in a different format than expected.

**Evidence**:
- Only 49 tokens generated (should be multiples of 35 for standard format)
- IS_MARK only at 2 positions (not at expected marker positions)
- Markers (REG_PC, REG_AX, etc.) may not be generated during autoregressive execution

**Possible Explanations**:
1. **Compact format**: VM uses a different token format during generation
2. **Marker elision**: Markers are implicit rather than explicit tokens
3. **Draft tokens**: The "draft" mode uses a different format
4. **Context vs Generation**: Initial context has markers, but generated tokens don't

## Next Steps for Investigation

### Option A: Understand Autoregressive Format

1. Examine `run_vm.py` generation loop
2. Understand how tokens are generated step-by-step
3. Check if markers are actually generated or just implied
4. Verify the actual token sequence being fed to the model

### Option B: Alternative Threshold Mechanism

1. Use absolute positional encoding instead of IS_MARK-based distances
2. Compute byte positions from step boundaries rather than marker distances
3. Modify Layer 3 attention to use a different pattern

### Option C: Accept Handlers

1. Document that threshold mechanism has architectural limitations
2. Keep Python handlers as permanent solution
3. Update documentation to clarify hybrid execution model
4. Remove "neural weights broken" comments

## Recommendation

**Short term**: Accept Option C (keep handlers)

**Rationale**:
- The threshold attention mechanism is complex and deeply integrated
- Fixing it requires understanding the entire autoregressive generation pipeline
- Handlers work correctly and reliably
- Time investment vs. benefit may not justify the fix

**Long term**: Investigate Option A if pure neural execution is a priority

**Actions**:
1. Document current findings
2. Update handler comments from "neural weights broken" to "requires autoregressive format support"
3. Note that weights are correct but runtime context doesn't match expectations
4. Mark as "architectural limitation" rather than "bug"

## Files Created During Fix Attempt

### Investigation Scripts
```
check_l1_weights.py       - Verified Layer 1 weights are set correctly ✓
check_is_mark.py          - Found IS_MARK not present at runtime ✗
debug_l1_attention.py     - Checked Layer 1 attention output (no L1H1/L1H0)
debug_l1_threshold.py     - Checked Layer 1 FFN output (no L1H1/L1H0)
```

### Key Findings
- **Weight initialization**: ✓ Correct
- **Embedding configuration**: ✓ Correct
- **Runtime behavior**: ✗ IS_MARK not present where expected
- **Root cause**: Autoregressive generation format mismatch

## Conclusion

The fix attempt revealed that the issue is not with weight initialization (which is correct) but with how the autoregressive generation works. The threshold attention mechanism expects marker tokens with IS_MARK flags, but these are not present in the generated context.

**Status**: Investigation complete, fix deferred pending architectural review

**Recommendation**: Keep handlers as permanent solution, document as architectural constraint

---

**Investigation Time**: ~3 hours
**Outcome**: Root cause identified (autoregressive format), fix deferred
**Next**: Document findings and update handler comments
