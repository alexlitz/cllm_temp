# Plan: Complete First-Step JMP Fix

## Current Status (as of last commit)

**Working:**
- ✅ Layer 5 head 2: Fetches opcode from address 0 at PC marker (first step)
- ✅ Layer 5 head 3: Fetches immediate from address 1 at PC marker (first step)
- ✅ Layer 5 FFN: Decodes JMP opcode → OP_JMP = 5.0 at PC marker
- ✅ Layer 6 FFN: JMP PC override → OUTPUT = 16 at PC marker

**Not Working:**
- ❌ Final prediction: PC_b0 = 0 instead of 16
- ❌ Something between Layer 6 and final prediction zeros OUTPUT

**Test Results:**
- 7/9 tests passing (same as before): NOP, IMM 0/42/255, EXIT, ADD
- 2/9 tests failing: JMP 16, LEA 8

---

## Root Cause Analysis

### Hypothesis 1: Layer 15 Nibble Copy Overwrites OUTPUT
Layer 15 FFN copies EMBED to OUTPUT for byte positions. It should be suppressed at PC positions (like we did for SP/BP), but the suppression might not be working.

**Check:**
- Debug script: trace OUTPUT through layers 6-15
- Look for: `OUTPUT_LO/HI` values changing from 16 to 0

**Fix:**
- Ensure Layer 15 suppression includes PC positions
- May need to strengthen suppression (increase S value)

### Hypothesis 2: Layer 3 PC Increment Overwriting
Layer 3 FFN increments PC and writes to OUTPUT. On first step, it might be overwriting the JMP target.

**Check:**
- Debug Layer 3 output at PC marker on first step
- See if OUTPUT is being set by Layer 3 increment logic

**Fix:**
- Add HAS_SE gate to Layer 3 PC increment
- Only increment when HAS_SE (not first step)

### Hypothesis 3: Missing PC Position Flag
The PC position might not be properly flagged, so suppressions aren't working.

**Check:**
- Debug H1 flags at PC marker position
- Check MARK_PC, IS_BYTE flags

**Fix:**
- Ensure PC marker position has correct flags
- Add additional position flags if needed

---

## Implementation Plan

### Step 1: Debug OUTPUT Propagation (15 min)
Create `debug_output_layers.py` to trace OUTPUT through all layers:

```python
"""Trace OUTPUT_LO/HI through all layers for JMP."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM

# ... setup code ...

for layer_idx in range(17):  # 0-15 + final head
    if layer_idx < 16:
        x = model.blocks[layer_idx](x)
    else:
        logits = model.head(x)

    output_lo = [x[0, pc_marker_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
    output_hi = [x[0, pc_marker_pos, BD.OUTPUT_HI + k].item() for k in range(16)]
    output_val = decode_nibbles(output_lo, output_hi)

    print(f"After Layer {layer_idx}: OUTPUT = {output_val}")
```

**Expected Output:** Identify which layer zeros OUTPUT

### Step 2: Fix Layer 15 Suppression (10 min)
If Layer 15 is the culprit, ensure PC suppression is active:

**File:** `neural_vm/vm_step.py` line ~1764

**Current code:**
```python
# Suppress at PC/SP/BP positions
ffn.W_up[unit, BD.H1 + PC_I] = -S  # Suppress at PC
ffn.W_up[unit, BD.H1 + SP_I] = -S  # Suppress at SP
ffn.W_up[unit, BD.H1 + BP_I] = -S  # Suppress at BP
```

**Potential fix:** Strengthen suppression or add MARK_PC directly
```python
# Suppress at PC marker explicitly
ffn.W_up[unit, BD.MARK_PC] = -S * 10  # Strong suppression at PC marker
```

### Step 3: Fix Layer 3 PC Increment (15 min)
If Layer 3 is overwriting, gate the increment by HAS_SE:

**File:** `neural_vm/vm_step.py` line ~1870

**Find:** PC increment units in `_set_layer3_ffn()`

**Add gate:**
```python
# PC increment: only when HAS_SE (not first step)
ffn.W_up[unit, BD.MARK_PC] = S
ffn.W_up[unit, BD.HAS_SE] = S  # NEW: only increment on subsequent steps
ffn.b_up[unit] = -S * 1.5      # Changed threshold for two conditions
```

### Step 4: Verify and Test (10 min)
Run tests after each fix:

```bash
# Quick verification
python debug_output_layers.py

# Run JMP test
python -m pytest neural_vm/tests/test_strict_neural_predictions.py::TestJMP::test_jmp_16 -xvs

# Run all tests
python -m pytest neural_vm/tests/test_strict_neural_predictions.py -v
```

### Step 5: Fix LEA (if time permits) (20 min)
Once JMP works, apply similar pattern to LEA:
- LEA = 0 (opcode 0, immediate = offset)
- Should compute: AX = BP + offset
- Likely needs same first-step handling as JMP

---

## Acceptance Criteria

✅ **JMP test passes:** `test_jmp_16` and `test_jmp_8` both pass
✅ **No regressions:** All 7 currently passing tests still pass
✅ **OUTPUT preserved:** Layer 6 FFN output (16) reaches final prediction
✅ **First-step works:** JMP on first step (HAS_SE=0) predicts correct PC

---

## Risk Assessment

**Low Risk:**
- Layer 15 suppression fix (just adding flags)
- Debug scripts (no code changes)

**Medium Risk:**
- Layer 3 gating by HAS_SE (might affect normal PC increment)
- Need to verify multi-step programs still work

**High Risk:**
- None identified

---

## Estimated Time: 1-1.5 hours

1. Debug (15 min)
2. Fix identified issue (10-15 min)
3. Test and iterate (20-30 min)
4. LEA fix (20 min, optional)
5. Final testing (10-15 min)

---

## Fallback Plan

If OUTPUT corruption is in multiple layers:
1. Add explicit "preserve OUTPUT at PC marker" logic in Layer 15
2. Create dedicated output preservation FFN units
3. Use strong negative weights to block all writes to OUTPUT at PC positions
