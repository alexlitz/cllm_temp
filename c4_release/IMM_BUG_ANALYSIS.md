# IMM Bug Analysis: AX Byte Relay Failure

## Summary

IMM 42 fails at **token 7** (AX byte 1): predicts 42 instead of 0.

This is an **AX-specific byte relay bug** - PC byte relay works correctly.

---

## Test Results

```
Token  0 (REG_PC): 257 ✓
Token  1 (PC_b0):    8 ✓  ← PC byte 0
Token  2 (PC_b1):    0 ✓  ← PC byte 1 works!
Token  3 (PC_b2):    0 ✓  ← PC byte 2 works!
Token  4 (PC_b3):    0 ✓  ← PC byte 3 works!
Token  5 (REG_AX): 258 ✓
Token  6 (AX_b0):   42 ✓  ← AX byte 0
Token  7 (AX_b1):    0 ✗  ← AX byte 1 FAILS (predicts 42)
```

---

## Root Cause

### What SHOULD Happen

At position 16 (after token 6 = AX_b0 = 42):
- **BYTE_INDEX_0**: Should be ~1.0 (identifying byte 1 position)
- **OUTPUT**: Should encode 0 (for AX byte 1)
- **Next token**: 0

### What ACTUALLY Happens

At position 16:
- **BYTE_INDEX_0**: 0.970 ✓ (correct - incremented from 0 to ~1)
- **MARK_AX**: 0.000 ✗ (not propagated to byte position)
- **OUTPUT**: 42 ✗ (stale value from previous position)
- **Next token**: 42 ✗ (wrong)

---

## Technical Details

### Layer 6 FFN: IMM Logic

IMM sets OUTPUT at the AX marker (lines 2509-2522):

```python
# Activation condition
up = OP_IMM + MARK_AX - threshold

# Only fires when BOTH are active:
# - At AX marker: OP_IMM=1, MARK_AX=1 → fires ✓
# - At AX bytes:  OP_IMM=1, MARK_AX=0 → doesn't fire ✗
```

At AX byte positions:
1. **MARK_AX = 0** (not propagated from marker)
2. **IMM units don't activate**
3. **OUTPUT carries forward via residuals** (stays at 42)
4. **No logic zeros OUTPUT for byte 1**

### Why PC Bytes Work But AX Bytes Don't

PC bytes correctly emit [8, 0, 0, 0], but AX fails. This suggests:

**Hypothesis 1**: PC has byte-clearing logic that AX lacks
**Hypothesis 2**: PC OUTPUT is set differently (not via opcode units)
**Hypothesis 3**: A later layer (L7-L15) modifies OUTPUT for AX but not PC

### OUTPUT Flow

| Position | Token      | L6 INPUT | L15 OUTPUT | Predicted | Expected |
|----------|------------|----------|------------|-----------|----------|
| 14       | REG_AX(258)| 16 (0x10)| 42 (0x2A)  | 42        | 42 ✓     |
| 15       | AX_b0(42)  | 16 (0x10)| 42 (0x2A)  | 42        | 0  ✗     |

OUTPUT stays at 42 from position 14 to 15. No layer updates it to 0.

---

## Marker Propagation Issue

**MARK_AX is not propagated to byte positions:**

| Position | Token      | Embedding MARK_AX | L6 FFN MARK_AX | Expected |
|----------|------------|-------------------|----------------|----------|
| 14       | REG_AX(258)| 1.000 ✓           | 1.000 ✓        | 1.0      |
| 15       | AX_b0(42)  | 0.000 ✗           | 0.000 ✗        | 1.0      |
| 16       | ?          | 0.000 ✗           | 0.000 ✗        | 1.0      |

**Why this matters:**
- Opcode units (IMM, ALU ops) require MARK_AX to activate
- Without MARK_AX at byte positions, these units can't set OUTPUT
- This breaks byte emission for AX

---

## Missing Architecture

The model lacks **byte relay logic** for AX:

### What's Needed (but missing)

At AX byte positions, need logic to:
1. **Detect byte position**: Use BYTE_INDEX (already works ✓)
2. **Read register value**: From EMBED or carry dimensions
3. **Select correct byte**: Based on BYTE_INDEX
4. **Write to OUTPUT**: Emit that byte

### Current State

- **BYTE_INDEX computation**: ✓ Works (L1 FFN sets it correctly)
- **Marker propagation**: ✗ Missing (MARK_AX not propagated)
- **Byte selection**: ✗ Missing (no logic uses BYTE_INDEX for output)
- **OUTPUT clearing**: ✗ Missing (OUTPUT not zeroed for byte 1+)

---

## Why PC Works

PC bytes work despite similar architecture. Possible reasons:

1. **PC is set earlier** (L3 carry-forward) vs AX (L6 opcode)
2. **PC has dedicated relay attention** that AX lacks
3. **PC OUTPUT cleared by later layers** (L7-L15)
4. **PC only uses byte 0** in current tests (bytes 1-3 happen to default to 0)

Need to investigate PC-specific logic to understand the difference.

---

## Next Steps

1. **Find why PC bytes work**: Identify PC byte emission mechanism
2. **Replicate for AX**: Add same logic for AX byte positions
3. **Test with multi-byte values**: Ensure bytes 1-3 work for non-zero values
4. **Check other registers**: Verify SP, BP don't have same issue

---

## Related Issues

This may be connected to:
- JMP fix (selective OUTPUT gating for PC)
- PSH broken fix (CMP[0] leakage affecting MARK_SP)
- IS_BYTE unreliability (cleared somewhere)

All point to marker propagation and byte relay issues.
