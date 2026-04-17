# Magic Number Elimination - Refactoring Guide

## Overview

This document describes the elimination of magic numbers from the Neural VM codebase, replacing hard-coded numeric literals with named constants for better maintainability and clarity.

## What Are Magic Numbers?

**Magic numbers** are hard-coded numeric literals in code that lack context:
- ❌ `draft_tokens[6]` - What is position 6?
- ❌ `if threshold > 2.5:` - Why 2.5?
- ❌ `model.blocks[8]` - What happens in layer 8?

**Named constants** provide clarity:
- ✅ `draft_tokens[POS_AX_BYTE0]` - Clearly the first byte of AX
- ✅ `if threshold > THRESHOLD_ALU_3WAY:` - Explains the purpose
- ✅ `model.blocks[LAYER_L8]` - Layer 8 is ALU lo nibble

## New Constant Modules

### 1. `neural_vm/token_layout.py`

Defines the 35-token output format positions:

```python
# Token positions (clear names instead of magic numbers)
POS_PC_MARKER = 0      # Instead of: 0
POS_PC_BYTE0 = 1       # Instead of: 1
POS_AX_MARKER = 5      # Instead of: 5
POS_AX_BYTE0 = 6       # Instead of: 6 (the most critical!)
POS_BP_BYTE0 = 16      # Instead of: 16

# Helper functions
get_ax_bytes(tokens)   # Extract AX value
get_pc_value(tokens)   # Parse PC as integer
bytes_to_int32(bytes)  # Convert LE bytes to int
```

**Usage:**
```python
# Before:
pc_bytes = draft_tokens[1:5]
pc = pc_bytes[0] | (pc_bytes[1] << 8) | (pc_bytes[2] << 16) | (pc_bytes[3] << 24)

# After:
pc_bytes = get_pc_bytes(draft_tokens)
pc = bytes_to_int32(pc_bytes)
```

### 2. `neural_vm/layer_constants.py`

Defines layer indices, thresholds, and architectural constants:

```python
# Layer indices (instead of 0, 1, 2, ..., 15)
LAYER_L5 = 5          # Fetch & decode
LAYER_L6 = 6          # Call routing
LAYER_L8 = 8          # ALU lo nibble

# Threshold values (instead of scattered 2.5, 1.5, etc.)
THRESHOLD_ALU_3WAY = 2.5        # 3-way AND for ALU
THRESHOLD_LEA_LO = 15.5         # LEA lo nibble (amplified)
THRESHOLD_PC_INCREMENT = 1.5    # PC + HAS_SE activation

# Attention constants
ATTN_WEIGHT_LARGE = 50.0        # Strong attention
ATTN_WEIGHT_BLOCKING = 500.0    # Anti-leakage

# FFN scaling
FFN_SCALE_S = 16.0              # Primary scaling factor
```

**Usage:**
```python
# Before:
ffn.W_up[unit, BD.MARK_AX] = 16.0
ffn.b_up[unit] = -16.0 * 2.5

# After:
ffn.W_up[unit, BD.MARK_AX] = FFN_SCALE_S
ffn.b_up[unit] = -FFN_SCALE_S * THRESHOLD_ALU_3WAY
```

## Files Updated

### ✅ `neural_vm/lea_correction.py`

**Before:**
```python
# Magic numbers everywhere
if len(draft_tokens) < 20:
    return predicted_ax_byte0

pc_bytes = draft_tokens[1:5]
pc = pc_bytes[0] | (pc_bytes[1] << 8) | ...

bp_bytes = draft_tokens[16:20]
bp = bp_bytes[0] | (bp_bytes[1] << 8) | ...
```

**After:**
```python
# Named constants
if len(draft_tokens) < POS_BP_BYTE3 + 1:
    return predicted_ax_byte0

pc_bytes = get_pc_bytes(draft_tokens)
pc = bytes_to_int32(pc_bytes)

bp_bytes = get_bp_bytes(draft_tokens)
bp = bytes_to_int32(bp_bytes)
```

## Benefits

### 1. **Readability**
```python
# Before: What does 6 mean?
predicted = logits[0, -1, 6].item()

# After: Clearly the first byte of AX
predicted = logits[0, -1, POS_AX_BYTE0].item()
```

### 2. **Maintainability**
If the token format changes, update constants in ONE place instead of hunting through code.

### 3. **Documentation**
Constants serve as inline documentation explaining architectural decisions.

### 4. **Refactoring Safety**
IDEs can find all uses of a constant; searching for "6" finds hundreds of unrelated instances.

### 5. **Type Safety**
Named constants can be validated and type-checked; raw numbers cannot.

## Common Magic Number Categories

### Token Positions
- ✅ Replaced with `POS_*` constants in `token_layout.py`
- Examples: `POS_AX_BYTE0`, `POS_PC_MARKER`

### Layer Indices
- ✅ Replaced with `LAYER_L*` constants in `layer_constants.py`
- Examples: `LAYER_L5`, `LAYER_L8`

### Threshold Values
- ✅ Replaced with `THRESHOLD_*` constants in `layer_constants.py`
- Examples: `THRESHOLD_ALU_3WAY`, `THRESHOLD_LEA_LO`

### Attention Weights
- ✅ Replaced with `ATTN_WEIGHT_*` constants
- Examples: `ATTN_WEIGHT_LARGE`, `ATTN_WEIGHT_BLOCKING`

### Scaling Factors
- ✅ Replaced with `FFN_SCALE_*` constants
- Examples: `FFN_SCALE_S`, `FFN_OUTPUT_SCALE`

## Remaining Magic Numbers to Address

### In `neural_vm/vm_step.py` (TODO)
Many threshold values, layer indices, and dimension indices still use magic numbers:
```python
# Examples needing replacement:
ffn.b_up[unit] = -S * 2.5              # Use THRESHOLD_ALU_3WAY
attn = model.blocks[6].attn            # Use LAYER_L6
attn.W_q[base + 32, BD.MARK_PC] = L    # Why 32? Needs constant
```

### In Test Files (Lower Priority)
Test files use magic numbers for test data, which is acceptable:
```python
# This is OK - test data values
test_cases = [(8, 65544), (100, 65636)]
```

## Migration Guide

### For New Code
Always use named constants:
```python
from neural_vm.token_layout import POS_AX_BYTE0, get_ax_bytes
from neural_vm.layer_constants import THRESHOLD_ALU_3WAY, FFN_SCALE_S

# Extract AX
ax_bytes = get_ax_bytes(tokens)

# Set threshold
ffn.b_up[unit] = -FFN_SCALE_S * THRESHOLD_ALU_3WAY
```

### For Existing Code
Replace incrementally:
1. Identify magic number
2. Find or create appropriate constant
3. Replace with constant
4. Test to ensure behavior unchanged

## Testing

All tests pass with the new constants:
```bash
python test_lea_with_correction.py        # ✓ LEA correction works
python test_lea_comprehensive.py          # ✓ All LEA cases pass
```

## Next Steps

1. **Phase 2**: Replace magic numbers in `vm_step.py`
   - Layer indices (use `LAYER_L*`)
   - Threshold values (use `THRESHOLD_*`)
   - Dimension offsets (create new constants)

2. **Phase 3**: Create dimension index constants
   - `DIM_FETCH_LO`, `DIM_ALU_LO`, etc.
   - Replace hard-coded dimension arithmetic

3. **Phase 4**: Document all constants
   - Add docstrings explaining why each value
   - Link to architectural design docs

## Summary

**Completed:**
- ✅ Token position constants (`token_layout.py`)
- ✅ Layer constants (`layer_constants.py`)
- ✅ Updated `lea_correction.py` to use constants
- ✅ Helper functions for common operations
- ✅ All tests passing

**Impact:**
- Code is more readable and self-documenting
- Easier to maintain and refactor
- Safer to modify (change in one place)
- Better IDE support (autocomplete, find usages)

**Files Changed:** 2 new modules + 1 updated file
**Lines of Code:** ~200 lines of well-documented constants
**Magic Numbers Eliminated:** ~15 in `lea_correction.py` alone
