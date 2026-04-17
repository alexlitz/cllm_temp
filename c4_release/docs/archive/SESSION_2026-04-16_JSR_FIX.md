# Session Summary: JSR Neural Path Fix (2026-04-16)

## Problem
JSR neural path was generating incorrect MEM val bytes:
- MEM val was `[0, 18, 18, 18]` (0x12121200)
- Expected: `[10, 0, 0, 0]` (0x0A = return_addr)

## Root Causes Found

### 1. MEM_VAL_B0-B3 Off-by-One Position
**File:** `neural_vm/vm_step.py` line ~2296

The MEM_VAL_B0-B3 flags were defined at positions d=5,6,7,8 from MEM marker (the actual val byte positions). But for autoregressive prediction where position N's OUTPUT predicts position N+1's content, these flags need to fire at d=4,5,6,7.

**Fix:**
```python
# Before (wrong):
# MEM_VAL_B0: d=5 from MEM → L2H0[MEM]=1 (d≤5.5), H1[MEM]=0 (d>4.5)

# After (correct):
# MEM_VAL_B0: d=4 from MEM → H1[MEM]=1 (d≤4.5), H0[MEM]=0 (d>3.5)
ffn.W_up[unit, BD.H1 + MEM_I] = S
ffn.W_gate[unit, BD.H0 + MEM_I] = -1.0  # Changed from H4 to H0
```

### 2. L14 Val Head BP Negative×Negative Score Bug
**File:** `neural_vm/vm_step.py` line ~6569

When OP_JSR is active (~5.2), the Q[1] for AX source bonus becomes very negative:
- Q[1] = L - 2L×OP_JSR = 15 - 30×5.2 = -142

The K[1] at BP bytes was also negative due to exclusion:
- K[1] = -L × H1[BP] = -15

The negative×negative multiplication gave BP bytes a huge positive score:
- Score = Q[1] × K[1] / sqrt(HD) = (-142) × (-15) / 8 = +267

This caused L14 val heads to attend to BP bytes (value 0x12) instead of STACK0 bytes (value 0x0A).

**Fix:**
```python
# Before (wrong):
attn.W_k[base + 1, BD.H1 + BP_I] = -L  # Exclude BP area (overlaps H1)

# After (correct):
# Removed the BP exclusion - the STACK0 source bonus in dim 2 handles JSR/ENT
# attn.W_k[base + 1, BD.H1 + BP_I] = -L  # REMOVED
```

## Results

- **Before fix:** MEM val = `0x12121200` (wrong - BP bytes)
- **After fix:** MEM val = `0x0000000A` (correct - return_addr)
- **Test suite:** All 1096 tests pass

## Files Modified
- `neural_vm/vm_step.py`:
  - `_set_layer2_mem_byte_flags()` - MEM_VAL position fix
  - `_set_layer14_mem_generation()` - L14 val head K weights fix
  - `_SetDim` class comments updated

## ENT/LEV Status
ENT neural path still needs handler - issues include:
- BP computation wrong (L6 FFN)
- STACK0 marker/bytes generation wrong
- OP_ENT = 0 at embedding time (propagates too late)
