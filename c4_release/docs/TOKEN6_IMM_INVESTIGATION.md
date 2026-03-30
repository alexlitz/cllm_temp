# Token 6 (AX Byte 0) IMM Prediction Failure - Investigation

## Issue
IMM operations fail on first step with Token 6 (AX byte 0) predicting 0 instead of the immediate value.

```
IMM 1: FAIL ✗ (9/10) - Token 6: pred=0 exp=1
IMM 8: FAIL ✗ (9/10) - Token 6: pred=0 exp=8
```

## Token Structure

```
Token 0 = PC marker
Tokens 1-4 = PC bytes 0-3
Token 5 = AX marker
Tokens 6-9 = AX bytes 0-3
...
```

In autoregressive generation:
- To predict token N, logits from position N-1 are used
- To predict token 6 (AX byte 0): **logits from position 5 (AX marker)** are used

## Expected Architecture Flow (First Step)

### 1. Layer 5 Attention Head 3 (lines 2408-2442)
**Purpose**: Fetch immediate byte at fixed address PC_OFFSET+1

- **Q**: Fires at token 0 (MARK_PC)
- **K**: Matches ADDR_KEY at address PC_OFFSET+1 (e.g., address 3)
- **V**: Copies CLEAN_EMBED_LO/HI from matched memory position
- **O**: Writes to FETCH_LO/HI and AX_CARRY_LO/HI **at token 0**

**Result**: Immediate byte value in FETCH dimensions at token 0 (PC marker)

### 2. Layer 5 FFN - Opcode Decode (lines 2445-2503, called at line 1438)
**Purpose**: Decode OPCODE_BYTE → OP_IMM flag

- **Condition**: Fires at MARK_AX (line 2501) = token 5
- **Input**: OPCODE_BYTE_LO/HI (written by Layer 5 head 2 at token 0)
- **Output**: OP_IMM ≈ 5.0 at token 5 (AX marker)

**Note**: This is in Layer 5 FFN, so OP_IMM is available BEFORE Layer 6.

### 3. Layer 6 Attention Head 4 (lines 2642-2657)
**Purpose**: First-step FETCH relay from PC marker to AX marker

- **Q**: Fires at MARK_AX when NOT HAS_SE (token 5, first step only)
- **K**: Targets MARK_PC (token 0)
- **V**: Copies FETCH_LO/HI from token 0
- **O**: Writes FETCH_LO/HI **at token 5**

**Result**: FETCH dimensions (immediate value) now available at token 5 (AX marker)

### 4. Layer 6 FFN - IMM Routing (lines 2679-2696)
**Purpose**: Write immediate value to OUTPUT for AX byte 0 prediction

- **Condition**: `up = OP_IMM * S + MARK_AX * S - S * 4.0`
  - At token 5: OP_IMM ≈ 5.0, MARK_AX ≈ 1.0+
  - up ≈ 500 + 100 - 400 = 200 > 0 ✓
- **Gate**: `W_gate[unit, BD.FETCH_LO + k] = 1.0`
  - Reads FETCH_LO/HI at token 5
- **Down**: `W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S`
  - Writes OUTPUT_LO/HI at token 5

**Result**: OUTPUT dimensions at token 5 contain immediate value

### 5. Logits Prediction
- Logits at position 5 (AX marker) are computed from OUTPUT dimensions
- Token 6 (AX byte 0) is sampled from these logits

**Expected**: Token 6 = immediate value byte 0

## Potential Failure Points

### 1. Memory Setup
**Hypothesis**: Memory entries at address PC_OFFSET+1 might not be initialized correctly.

- Bytecode/data section needs to be embedded as memory entries before execution
- ADDR_KEY dimensions must match for head 3's attention to work
- CLEAN_EMBED_LO/HI must contain the correct immediate value

**Diagnostic**: Check if memory entries exist at address PC_OFFSET+1 with correct ADDR_KEY values.

### 2. OPCODE_BYTE Propagation
**Hypothesis**: OPCODE_BYTE_LO/HI might not be available at token 5 for opcode decode.

- Layer 5 head 2 (first-step opcode fetch, lines 2373-2407) writes OPCODE_BYTE at token 0
- Opcode decode FFN (Layer 5) needs OPCODE_BYTE at token 5 (MARK_AX gate)
- Residual stream dimensions are token-specific!

**Issue**: Head 2 writes OPCODE_BYTE at token 0, but opcode decode reads at token 5!

**Possible fix**: Opcode decode shouldn't gate on MARK_AX for first-step, or there should be a relay mechanism.

### 3. FETCH Relay Timing
**Hypothesis**: Layer 6 head 4 might run before Layer 5 head 3 completes?

- No, this is impossible - Layer 6 always comes after Layer 5
- But head 4 runs in Layer 6 ATTENTION, before Layer 6 FFN

**Flow**:
1. Layer 5 attention (head 3 writes FETCH at token 0)
2. Layer 5 FFN (opcode decode writes OP_IMM at token 5??)
3. Layer 6 attention (head 4 copies FETCH from token 0 to token 5)
4. Layer 6 FFN (IMM routing reads OP_IMM and FETCH at token 5)

**Critical issue**: In step 2, opcode decode fires at MARK_AX (token 5) but reads OPCODE_BYTE which was written at token 0! This is a **token position mismatch**.

### 4. Subsequent Step Interference
**Hypothesis**: Head 3 runs on ALL steps (no HAS_SE suppression), potentially interfering.

- Head 3 always fetches at fixed address PC_OFFSET+1
- On subsequent steps, this wrong value is written to FETCH at token 0
- But head 4 only runs on first step, so no interference with head 0's FETCH write at token 5

**Verdict**: Unlikely to be the issue for first-step prediction.

## Root Cause: OPCODE_BYTE Token Position Mismatch

**The Critical Bug**:

Layer 5 head 2 (first-step opcode fetch) writes OPCODE_BYTE_LO/HI **at token 0 (PC marker)**.

Layer 5 FFN opcode decode reads OPCODE_BYTE_LO/HI **at token 5 (AX marker)** due to the `W_gate[unit, BD.MARK_AX] = 1.0` condition (line 2501).

In transformers, each token position has its own residual stream. Dimensions written at token 0 are NOT automatically available at token 5!

**Result**: Opcode decode reads OPCODE_BYTE ≈ 0 at token 5, fails to set OP_IMM flag, and IMM routing doesn't fire.

## Solution Options

### Option 1: Broadcast OPCODE_BYTE via Attention
Add a Layer 6 attention head that copies OPCODE_BYTE from PC marker (token 0) to all register markers (tokens 5, 10, 15, 20).

### Option 2: Decode Opcode at PC Marker
Change opcode decode to fire at MARK_PC instead of MARK_AX:
- Remove `W_gate[unit, BD.MARK_AX] = 1.0` (line 2501)
- Add `W_gate[unit, BD.MARK_PC] = 1.0`
- OP flags would be written at token 0

**Problem**: Then Layer 6 FFN IMM routing can't read OP_IMM at token 5!

### Option 3: Broadcast OP Flags via Attention (Recommended)
Add a Layer 6 attention head (before FFN) that copies all OP_* flags from PC marker (token 0) to AX marker (token 5).

**Implementation**:
- Q: Fire at MARK_AX
- K: Target MARK_PC
- V: Copy all OP_* dimensions (34 opcodes)
- O: Write OP_* dimensions to AX marker

This would make OP flags available at token 5 for IMM routing.

### Option 4: Opcode Decode in Layer 4
Move opcode decode to Layer 4 FFN, before Layer 5 attention:
- Decode at PC marker (token 0)
- Layer 5 attention head copies OP flags to other markers
- Layer 6 FFN can read OP flags at any marker

## Diagnostic Steps

1. **Check OPCODE_BYTE at token 5 after Layer 5 FFN**
   - If ≈ 0, confirms token position mismatch

2. **Check OP_IMM at token 5 after Layer 5 FFN**
   - If ≈ 0, confirms opcode decode didn't fire

3. **Check FETCH at token 0 after Layer 5 attention**
   - If ≈ 0, head 3 fetch failed (memory issue)

4. **Check FETCH at token 5 after Layer 6 attention**
   - If ≈ 0, head 4 relay failed

5. **Check OUTPUT at token 5 after Layer 6 FFN**
   - If ≈ 0, IMM routing didn't fire (likely due to missing OP_IMM)

## Related Issues

This same bug likely affects:
- **LEA operations**: Also need OP_LEA flag at AX marker
- **All arithmetic operations**: Need OP_* flags at AX marker
- **All operations that route through Layer 6 FFN**

The architecture assumes opcode flags are available globally, but they're actually token-position-specific.

## Next Steps

1. Run diagnostic script to confirm OPCODE_BYTE / OP_IMM values at token 5
2. Implement Option 3 (OP flag broadcast via Layer 6 attention head)
3. Test IMM, LEA, and other operations
4. Verify no regressions on subsequent-step execution

---

**Date**: 2026-03-29
**Status**: Root cause identified, fix pending
