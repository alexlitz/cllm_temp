# Magic Number Refactoring Examples

This document shows before/after examples of eliminating magic numbers in `neural_vm/vm_step.py`.

## Example 1: Layer Access

### Before (Magic Numbers)
```python
# Layer 6 attention - what happens here?
attn6 = model.blocks[6].attn
ffn6 = model.blocks[6].ffn

# Layer 8 FFN - ALU? Memory? Unclear
ffn8 = model.blocks[8].ffn
```

### After (Named Constants)
```python
from neural_vm.layer_constants import LAYER_L6, LAYER_L8

# Layer 6: Function call output routing (clear from constant name)
attn6 = model.blocks[LAYER_L6].attn
ffn6 = model.blocks[LAYER_L6].ffn

# Layer 8: ALU lo nibble operations (clear from constant)
ffn8 = model.blocks[LAYER_L8].ffn
```

## Example 2: Threshold Values

### Before (Magic Numbers)
```python
# Why 2.5? What does this gate?
ffn.W_up[unit, BD.MARK_AX] = S
ffn.W_up[unit, BD.ALU_LO + a] = S
ffn.W_up[unit, BD.AX_CARRY_LO + b] = S
ffn.b_up[unit] = -S * 2.5  # 3-way AND - but how do we know?
```

### After (Named Constants)
```python
from neural_vm.layer_constants import THRESHOLD_ALU_3WAY, FFN_SCALE_S

# Clear: this is a 3-way AND threshold for ALU operations
ffn.W_up[unit, BD.MARK_AX] = FFN_SCALE_S
ffn.W_up[unit, BD.ALU_LO + a] = FFN_SCALE_S
ffn.W_up[unit, BD.AX_CARRY_LO + b] = FFN_SCALE_S
ffn.b_up[unit] = -FFN_SCALE_S * THRESHOLD_ALU_3WAY  # Self-documenting
```

## Example 3: Attention Weights

### Before (Magic Numbers)
```python
# What do these values mean?
attn.W_q[base, BD.MARK_PC] = 50.0
attn.W_q[base + 33, BD.MARK_AX] = 500.0
attn.W_q[base + 33, BD.CONST] = -500.0
```

### After (Named Constants)
```python
from neural_vm.layer_constants import ATTN_WEIGHT_LARGE, ATTN_WEIGHT_BLOCKING

# Large weight for strong attention pattern
attn.W_q[base, BD.MARK_PC] = ATTN_WEIGHT_LARGE

# Anti-leakage blocking pattern (500/-500)
attn.W_q[base + 33, BD.MARK_AX] = ATTN_WEIGHT_BLOCKING
attn.W_q[base + 33, BD.CONST] = -ATTN_WEIGHT_BLOCKING
```

## Example 4: LEA-Specific Thresholds

### Before (Magic Numbers)
```python
# LEA units - why 15.5? Where did this come from?
ffn.W_up[unit, BD.MARK_AX] = S
ffn.W_up[unit, BD.ALU_LO + a] = S
ffn.W_up[unit, BD.FETCH_LO + b] = S
ffn.b_up[unit] = -S * 15.5  # High threshold for amplified ALU
```

### After (Named Constants)
```python
from neural_vm.layer_constants import THRESHOLD_LEA_LO, FFN_SCALE_S

# High threshold to account for ALU amplification in LEA
ffn.W_up[unit, BD.MARK_AX] = FFN_SCALE_S
ffn.W_up[unit, BD.ALU_LO + a] = FFN_SCALE_S
ffn.W_up[unit, BD.FETCH_LO + b] = FFN_SCALE_S
ffn.b_up[unit] = -FFN_SCALE_S * THRESHOLD_LEA_LO  # Documents the issue
```

## Example 5: Token Position Checking

### Before (Magic Numbers)
```python
# What is position 6? Why do we care about it?
for i, draft_tok in enumerate(draft_tokens):
    pred = logits[0, ctx_len - 1 + i, :].argmax(-1).item()

    # Apply correction at position 6
    if i == 6:
        pred = correct_lea_prediction(context, draft_tokens, pred)
```

### After (Named Constants)
```python
from neural_vm.token_layout import POS_AX_BYTE0

# Clear: we're correcting the first byte of AX register
for i, draft_tok in enumerate(draft_tokens):
    pred = logits[0, ctx_len - 1 + i, :].argmax(-1).item()

    # Apply LEA correction to AX byte 0
    if i == POS_AX_BYTE0:
        pred = correct_lea_prediction(context, draft_tokens, pred)
```

## Example 6: PC Calculation

### Before (Magic Numbers)
```python
# Why subtract 8? What is INSTR_WIDTH?
pc_after = draft_tokens[1] | (draft_tokens[2] << 8) | ...
if pc_after >= 10:  # Why 10?
    pc_before = pc_after - 8
```

### After (Named Constants)
```python
from neural_vm.constants import PC_OFFSET, INSTR_WIDTH
from neural_vm.token_layout import get_pc_bytes, bytes_to_int32

# Clear: PC is stored little-endian, offset by PC_OFFSET
pc_bytes = get_pc_bytes(draft_tokens)
pc_after = bytes_to_int32(pc_bytes)

# PC before = PC after - instruction width
if pc_after >= PC_OFFSET + INSTR_WIDTH:
    pc_before = pc_after - INSTR_WIDTH
```

## Migration Strategy

### Step 1: Import Constants
```python
from neural_vm.layer_constants import (
    LAYER_L5, LAYER_L6, LAYER_L8,
    THRESHOLD_ALU_3WAY, THRESHOLD_LEA_LO,
    ATTN_WEIGHT_LARGE, ATTN_WEIGHT_BLOCKING,
    FFN_SCALE_S, FFN_OUTPUT_SCALE
)
from neural_vm.token_layout import (
    POS_AX_BYTE0, POS_PC_BYTE0, POS_BP_BYTE0,
    get_pc_bytes, get_ax_bytes, bytes_to_int32
)
```

### Step 2: Replace One Section at a Time
Focus on one function or layer at a time to avoid introducing bugs.

### Step 3: Test After Each Change
Run tests to ensure behavior is unchanged:
```bash
python test_lea_comprehensive.py
python tests/test_suite_1000.py  # If available
```

### Step 4: Document Unclear Values
If you find a magic number you don't understand:
```python
# FIXME: What does this threshold represent?
# Need to investigate before creating constant
ffn.b_up[unit] = -S * 3.7  # Unknown purpose
```

## Benefits Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Readability** | "What is 6?" | "POS_AX_BYTE0" |
| **Searchability** | Find "6" → 1000+ results | Find "POS_AX_BYTE0" → exact matches |
| **Refactoring** | Change everywhere | Change once |
| **Documentation** | Comments needed | Self-documenting |
| **Type Safety** | No validation | Can type-check |

## Common Mistakes to Avoid

### ❌ Don't Create Constants for Test Data
```python
# BAD: Test values don't need constants
TEST_VALUE_42 = 42
assert add(40, 2) == TEST_VALUE_42

# GOOD: Just use the literal
assert add(40, 2) == 42
```

### ❌ Don't Create Constants for One-Off Values
```python
# BAD: Only used once
LOOP_COUNT_17 = 17
for i in range(LOOP_COUNT_17):
    ...

# GOOD: Use literal if it's clear in context
for i in range(17):  # Process all 17 special cases
    ...
```

### ✅ Do Create Constants for Repeated Architectural Values
```python
# GOOD: Used throughout codebase
THRESHOLD_ALU_3WAY = 2.5  # Used in L8, L9, etc.
POS_AX_BYTE0 = 6          # Used in corrections, validation, etc.
```

## Next Files to Refactor

Priority order:
1. ✅ `neural_vm/lea_correction.py` - DONE
2. ⏳ `neural_vm/vm_step.py` - Main file (large)
3. ⏳ `neural_vm/batch_runner.py` - Token positions
4. ⏳ Test files - Lower priority (test data is OK)

## Questions?

If you find a magic number and aren't sure:
1. Check if constant already exists in `layer_constants.py` or `token_layout.py`
2. Look at surrounding code for context
3. Check git history: `git log -p -- <file>` to see why value was chosen
4. Ask! Better to clarify than guess
