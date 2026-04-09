# ADD Root Cause - Updated Investigation

**Date**: 2026-04-07
**Status**: Root cause identified - AX_CARRY lost between Layer 3 and Layer 7

## Executive Summary

The original investigation (docs/FIX_ATTEMPT_SUMMARY.md) incorrectly concluded that Layer 1 threshold heads were broken due to missing IS_MARK flags. **This was wrong** - the investigation only examined the initial prefix, not the generated tokens.

**Actual root cause**: AX_CARRY_LO/HI are correctly set by Layer 3 attention but are **overwritten/lost** before reaching Layer 7, preventing ADD from accessing the second operand.

## What Works ✓

### 1. IS_MARK Flag
**Previous conclusion**: IS_MARK not present at runtime ✗
**Actual truth**: IS_MARK IS set correctly on marker tokens ✓

**Evidence**:
- Embedding hook on ALL forward passes shows REG_AX at position 54 has IS_MARK=1.000
- MEM tokens have IS_MARK=1.00
- The original check only captured the prefix (first forward pass), missing the generated tokens

**Script**: `check_is_mark_all_passes.py`

### 2. Layer 1 Threshold Heads
**Previous conclusion**: L1H1/L1H0 not being set ✗
**Actual truth**: L1H1/L1H0 ARE working correctly ✓

**Evidence**:
- Layer 1 attention output shows L1H1[AX]=1.000, L1H0[AX]=0.993
- Byte 0 positions correctly identified (L1H1=1, L1H0=0 pattern)
- Original investigation only looked at first forward pass

**Script**: `debug_l1_mechanism.py`

```
Layer 1 Attention Configuration:
  ALiBi slopes: [10. 10. 10.  0. 10. 10. 10. 10.]

Threshold Head Configuration:
  Head 0 (threshold 0.5):
    W_q[0, CONST=8] = 80.00
    W_k[0, IS_MARK=7] = 0.50
  Head 1 (threshold 1.5):
    W_q[64, CONST=8] = 80.00
    W_k[64, IS_MARK=7] = 1.50

Last Layer 1 attention output:
  Max L1H0[AX]: 0.993 ✓
  Max L1H1[AX]: 1.000 ✓
```

### 3. Layer 3 AX Carry-Forward
**Status**: Layer 3 attention DOES populate AX_CARRY ✓

**Evidence**:
- Layer 3 attention output shows AX_CARRY_LO=1.000, AX_CARRY_HI=1.000
- Layer 3 is correctly using L1H1/L1H0 byte index patterns
- Carry-forward attention mechanism is working as designed

**Script**: `debug_add_final_check.py`

```
Final Layer 3 attention output:
  Max AX_CARRY_LO: 1.000 ✓
  Max AX_CARRY_HI: 1.000 ✓
```

## What's Broken ✗

### Layer 3 → Layer 7: AX_CARRY Lost

**Problem**: AX_CARRY_LO/HI are set by Layer 3 attention but have disappeared by Layer 7.

**Evidence**:
- Layer 3: AX_CARRY_LO=1.000 ✓
- Layer 7: AX_CARRY_LO=0.00 ✗

**Script**: `debug_layer8_add.py`

```
Layer 7 output (gather operands):
  ALU_LO (first operand): nibble 0 (max=0.71) ← Some value present
  AX_CARRY_LO (second op): nibble 0 (max=0.00) ← ZERO!
```

## The Complete Failure Chain

```
Layer 0: Embedding sets IS_MARK on marker tokens ✓
    ↓
Layer 1: Threshold attention sets L1H1[AX]=1, L1H0[AX]=0 ✓
    ↓
Layer 3: Carry-forward attention sets AX_CARRY_LO/HI=1.0 ✓
    ↓
Layer 3 FFN or Layers 4-6: ??? (AX_CARRY gets overwritten/lost) ✗
    ↓
Layer 7: AX_CARRY_LO/HI = 0.0 ✗
    ↓
Layer 8: ADD has no second operand (3-way AND incomplete) ✗
    ↓
Result: Returns first operand only (10 instead of 42) ✗
```

## Root Cause Hypothesis

**Most likely**: Layer 3 FFN or one of Layers 4-6 is **writing to the AX_CARRY dimensions**, overwriting the values set by Layer 3 attention.

**Why this happens**:
1. Layer 3 attention sets AX_CARRY_LO/HI (dims 272-303)
2. These values are in the residual stream
3. Layer 3 FFN output is added to residual: `x = x + ffn(x)`
4. If FFN writes to AX_CARRY dims, it will overwrite them
5. Layers 4-6 may also write to these dims

**Contract violation**: AX_CARRY_LO/HI should be "reserved" for Layer 3 attention output and should NOT be written by FFN layers between Layer 3 and Layer 7.

## Investigation Scripts Summary

### Corrected Understanding
- `check_is_mark_all_passes.py` - Shows IS_MARK IS set during generation
- `debug_l1_mechanism.py` - Shows Layer 1 threshold heads ARE working
- `debug_add_final_check.py` - Shows Layer 3 AX_CARRY IS populated
- `debug_layer8_add.py` - Shows AX_CARRY is LOST by Layer 7

### Incorrect Original Scripts
- `check_is_mark.py` - Only checked first forward pass (prefix)
- `debug_l1_threshold.py` - Only checked first forward pass
- `debug_l1_attention.py` - Only checked first forward pass

**Key mistake**: Capturing only the first embedding/attention pass misses the generated tokens where actual VM execution happens!

## Next Steps

### Option 1: Identify Which Layer Overwrites AX_CARRY

1. Hook Layer 3 FFN output to see if it writes to AX_CARRY dims
2. Hook Layers 4-6 FFN outputs to find the culprit
3. Check weight configuration: which layers have non-zero W_down[:, AX_CARRY_LO:AX_CARRY_HI]?

### Option 2: Fix Weight Configuration

Once the offending layer is found:
1. Set W_down[AX_CARRY_LO:AX_CARRY_HI, :] = 0 for that layer
2. Ensure FFN doesn't write to reserved dimensions
3. Add contract check to enforce dim reservation

### Option 3: Accept Handlers (Original Recommendation)

**Still valid**: If fixing the weight configuration is complex, keeping handlers as a permanent solution is reasonable.

## Conclusion

The previous investigation was fundamentally flawed because it only examined the initial context prefix, not the generated tokens. The actual neural mechanisms (IS_MARK, Layer 1 thresholds, Layer 3 carry-forward) **are all working correctly**.

The real bug is that AX_CARRY values are being overwritten by an FFN layer between Layer 3 and Layer 7, breaking the carry-forward mechanism that ADD depends on.

**Status**: Need to identify which FFN layer is writing to AX_CARRY dimensions.

---

**Investigation Time**: 4 hours total
**Outcome**: Real root cause identified (AX_CARRY overwrite)
**Next**: Find and fix the layer that overwrites AX_CARRY
