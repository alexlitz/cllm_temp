# AX Byte Architecture Issue

## Problem Summary

The Neural VM architecture doesn't properly handle first-step multi-byte AX values, causing either IMM or LEA to fail depending on the fix attempted.

---

## Current Status

After L5 JMP fix:
- ✅ JMP 8: PASS
- ✅ JMP 16: PASS
- ❌ IMM 42: FAIL (with AX byte logic removed)
- ❌ IMM 255: FAIL (with AX byte logic removed)
- ❌ LEA 8: FAIL (OUTPUT=1 instead of 8)
- ❌ LEA 16: FAIL (OUTPUT=0 instead of 16)

---

## Root Cause

### First-Step AX Byte Propagation

**The Issue**: At AX byte positions on the first step, how should OUTPUT be set for bytes 1-3?

**Current Architecture** (L10 byte passthrough, lines 3690-3755):
```
Copies CLEAN_EMBED from previous step's AX bytes 1-3 → OUTPUT at current step's AX byte 0-2 positions.

At step 0 (no prev AX bytes): Q attends to same-step PC bytes (all 0),
producing OUTPUT for token 0 (correct: AX=0 initially).
```

**Assumption**: First-step AX bytes 1-3 should be 0 (from PC bytes).

**Reality**:
- **Single-byte IMM**: Bytes 1-3 should be 0 ✓ (assumption holds)
- **Multi-byte LEA**: Bytes 1-3 can be non-zero ✗ (assumption breaks)

---

## Attempted Fixes

### Fix 1: L3 AX Byte Defaults (Broke LEA)

**Location**: `vm_step.py` lines 2074-2095 (now removed)

**Logic**:
```python
# At BYTE_INDEX_K position → predict byte K+1 (OUTPUT = 0)
for byte_idx_dim in [BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2]:
    ffn.W_up[unit, BD.H1 + AX_I] = S
    ffn.W_up[unit, byte_idx_dim] = S
    ...
    ffn.W_down[BD.OUTPUT_LO + 0, unit] = 2.0 / S  # Force 0
```

**Result**:
- ✅ IMM 42/255: PASS (bytes 1-3 = 0)
- ✗ LEA 8/16: FAIL (bytes 2-3 forced to 0 when should be non-zero)

**Why It Failed**: Unconditionally sets OUTPUT=0 for all instructions, but LEA needs multi-byte values.

### Fix 2: Remove AX Byte Logic (Broke IMM)

**Location**: Same lines, replaced with TODO comment

**Result**:
- ✗ IMM 42/255: FAIL (bytes 1-3 get stale EMBED values)
- ✗ LEA 8/16: Still FAIL (different issue - L10 interference)

**Why IMM Fails**: Without defaults, EMBED at byte positions contains token value, not register byte value.

**Why LEA Still Fails**: L10 AX passthrough or other logic interfering (OUTPUT=1 at marker).

---

## Data Flow Analysis

### For LEA 8 (AX = 0x00010008 = [8, 0, 1, 0])

**At AX Marker (position 14, predicting AX_b0):**

| Layer | OUTPUT | Notes |
|-------|--------|-------|
| L3 | 16 | Initial |
| L6 | 17 | ? |
| L8 | 24 | ? |
| L9 | 8 | ✓ Correct (low byte of 0x00010008) |
| L10 | 1 | ✗ Changed to byte 2 value! |
| L15 | 1 | Propagated |

**Prediction**: 1 (expected 8) ✗

**Issue**: L10 changes OUTPUT from 8 (correct) to 1 (byte 2 of multi-byte value).

**Investigation**:
- OP_LEA = 4.942 ✓ (correctly set)
- AX_CARRY = 8 ✓ (correct value)
- AX passthrough should be suppressed (OP_LEA in suppressed list)
- But OUTPUT still changes

**Hypothesis**: Either:
1. AX passthrough suppression not working (compaction issue?)
2. Different L10 logic interfering
3. Attention head copying wrong byte

---

## Architectural Constraints

### L10 Byte Passthrough Design

**Assumptions**:
1. First step: AX = 0 (all bytes)
2. Subsequent steps: Copy from previous step
3. HAS_SE flag distinguishes first vs subsequent

**Reality**:
1. First step can have multi-byte AX (LEA, future ADD with carry, etc.)
2. L10 design doesn't account for this

### Why PC Works But AX Doesn't

**PC** has dedicated L3 logic:
- First-step default: PC = PC_OFFSET
- Increment: PC += INSTR_WIDTH
- Carry propagation for multi-byte

**AX** relies on:
- L6/L8/L9: Set value at marker
- L10: Byte passthrough (assumes first-step = 0)
- L15: Nibble copy (uses token EMBED, not register value)

**Missing**: First-step multi-byte propagation from marker to byte positions.

---

## Proper Solutions

### Option 1: L10 First-Step Multi-Byte Handling

**Change L10 byte passthrough** to:
- First step (HAS_SE=0): Attend to CURRENT STEP's AX marker OUTPUT
- Subsequent steps (HAS_SE=1): Attend to PREVIOUS STEP's AX bytes

**Implementation**:
```python
# Q: fires at byte positions
# When HAS_SE=0: attend to same-step marker
# When HAS_SE=1: attend to prev-step bytes

# Add HAS_SE-based routing in attention
```

**Pros**:
- Handles both IMM and LEA correctly
- Preserves existing architecture for subsequent steps

**Cons**:
- Complex attention pattern (needs HAS_SE-based K selection)
- May require separate heads for first vs subsequent

### Option 2: Marker OUTPUT Relay (Simpler)

**Add L11 FFN logic**:
- At AX byte positions when HAS_SE=0
- Copy OUTPUT from AX marker (via carry-forward or staging dim)

**Pros**:
- Simpler than Option 1
- Works for any multi-byte value

**Cons**:
- Requires staging dimension or intra-layer communication
- May conflict with carry override logic

### Option 3: Conditional AX Byte Defaults

**Make L3 defaults conditional on OP_IMM**:
- Move logic to L7+ (after OP_IMM is set)
- Only set bytes 1-3 = 0 when OP_IMM active

**Pros**:
- Preserves existing byte passthrough
- Minimal changes

**Cons**:
- OP_IMM not available until L6
- Would need to move defaults to L7+ FFN
- Still doesn't solve first-step multi-byte relay

---

## Investigation Needed

### L10 OUTPUT Change Mystery

**Question**: Why does OUTPUT change from 8 to 1 at L10 for LEA?

**Observations**:
- OP_LEA = 4.942 (correctly set)
- AX passthrough should be suppressed (OP_LEA in suppressed list)
- But OUTPUT changes anyway

**Next Steps**:
1. Check L10 attention output (head 0 and 1)
2. Check L10 FFN intermediate activations
3. Verify model compaction didn't corrupt weights
4. Check if there's a different L10 unit activating

---

## Recommended Next Step

**Priority 1**: Fix L10 OUTPUT change for LEA
- Debug why L10 changes OUTPUT from 8 to 1
- May be simpler bug than architectural redesign

**Priority 2**: Implement Option 1 or 2 for first-step multi-byte
- Choose based on complexity vs correctness tradeoff

**Priority 3**: Test with multi-byte ADD
- Verify solution works for carry propagation too

---

## Files Involved

- `neural_vm/vm_step.py`
  - Lines 2074-2078: AX byte defaults (removed, needs replacement)
  - Lines 3690-3755: L10 byte passthrough
  - Lines 3934-3949: L10 AX passthrough (may be interfering)

---

## Test Cases

**Minimal test set**:
1. IMM 1: Single-byte, non-zero
2. IMM 42: Single-byte, non-zero
3. LEA 8: Multi-byte (0x00010008)
4. LEA 16: Multi-byte (0x00010010)

**Expected results**:
- All should pass with correct AX bytes [b0, b1, b2, b3]
- IMM: [val, 0, 0, 0]
- LEA 8: [8, 0, 1, 0]
- LEA 16: [16, 0, 1, 0]

---

## Session Context

After fixing JMP (L5 attention softmax spreading), attempted to fix IMM byte relay but created conflict with LEA. The fundamental issue is that IMM and LEA have different multi-byte requirements, and the current architecture doesn't handle both correctly on first step.

Token budget: ~100k used, need efficient solution.
