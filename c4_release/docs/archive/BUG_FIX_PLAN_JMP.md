# JMP/NOP/EXIT AX Corruption Bug - Fix Plan

## Root Cause Analysis

The system has a **missing attention head** that should populate AX_CARRY with the previous AX value.

### Current Architecture

1. **L6 FFN** (lines 3683-3699): Routes `AX_CARRY → OUTPUT` at AX marker for JMP/NOP/EXIT
2. **L10 FFN passthrough** (lines 4674-4714): Routes `AX_CARRY → OUTPUT` at AX marker for non-AX-modifying opcodes
3. **L10 attention head 1** (lines 4438-4519): Handles AX bytes 1-3 passthrough from previous step

### The Bug

- **AX_CARRY at PC marker**: Populated with jump target by L5/L6 attention (correct for PC update)
- **AX_CARRY at AX marker**: **NOT populated** with previous AX value ← THIS IS THE BUG!

Result: When L6/L10 FFN routes `AX_CARRY → OUTPUT` at AX marker, it gets garbage/zero instead of the previous AX value.

### Why Tests Pass

The test suite uses `BakedC4Transformer` with hybrid mode (DraftVM + Transformer validation). The DraftVM (Python C4 interpreter) provides correct values, masking the neural bug.

## The Fix

Add an attention head that copies previous step's AX OUTPUT to current step's AX_CARRY at the AX marker.

### Proposed Solution: Add Layer 3 Attention Head

**Location**: Layer 3 attention, using an available head (need to find which head is free)

**Functionality**:
```
At: Current step's AX marker
Attend to: Previous step's AX marker (distance = 35 tokens)
Copy: OUTPUT_LO/HI → AX_CARRY_LO/HI
```

**Implementation**:
```python
# New head in L3 attention
base = <available_head> * HD

# Q: Fire at AX marker on subsequent steps (HAS_SE=1)
attn.W_q[base, BD.MARK_AX] = L
attn.W_q[base, BD.HAS_SE] = L  # Only on subsequent steps
attn.W_q[base, BD.CONST] = -L * 1.5  # Threshold

# K: Match previous step's AX marker
attn.W_k[base, BD.MARK_AX] = L

# V: Copy OUTPUT_LO/HI from previous AX marker
for k in range(16):
    attn.W_v[base + 1 + k, BD.OUTPUT_LO + k] = 1.0
    attn.W_v[base + 17 + k, BD.OUTPUT_HI + k] = 1.0

# O: Write to AX_CARRY_LO/HI at current AX marker
for k in range(16):
    attn.W_o[BD.AX_CARRY_LO + k, base + 1 + k] = 1.0
    attn.W_o[BD.AX_CARRY_HI + k, base + 17 + k] = 1.0
```

**ALiBi slope**: Use slope=5.0 (steep) to strongly prefer nearest previous step (d=35)

### Testing Plan

1. Test JMP: `JMP 16` should preserve AX=0 (currently gets AX=16)
2. Test NOP: `IMM 42; NOP` should preserve AX=42 (currently gets AX=1)
3. Test EXIT: `EXIT` should preserve AX=0 (likely also broken)
4. Run full test suite to ensure no regressions

### Alternative Fix (Simpler but Incomplete)

Remove L6 FFN routing for JMP/NOP/EXIT and rely entirely on L10 passthrough. But this still requires populating AX_CARRY, so it's not simpler.

### Next Steps

1. ✅ Identified root cause
2. Find available attention head in L3 or L4
3. Implement the AX OUTPUT → AX_CARRY relay
4. Test with JMP/NOP/EXIT
5. Verify no regressions in full test suite
