# Fix Summary: L10 AX Byte Passthrough Leakage

## Problem
Neural VM was predicting incorrect PC byte values due to L10 attention head 1 leaking to non-AX positions.

### Symptoms
- PC byte 1 predicted as 5 instead of 0 (copied AX value)
- IMM 5 test: 97.1% match rate (34/35 tokens)
- Other opcodes also affected by similar leakage

### Root Cause
L10 attention head 1 (AX byte passthrough) was designed to copy CLEAN_EMBED from previous step's AX bytes to current step's OUTPUT at AX byte positions only.

Anti-leakage mechanism used Q[33] = -5000 to suppress non-AX positions, but this wasn't strong enough:
- Softmax always picks the "least negative" option
- At PC byte positions, some context position would get ~100% attention weight
- This caused OUTPUT_LO to be set incorrectly

## Fix
Increased anti-leakage suppression weight from -5000 to -50000 in `vm_step.py` line 3038:

```python
# Before:
attn.W_q[base + 33, BD.CONST] = -5000.0
attn.W_q[base + 33, BD.H1 + AX_IDX] = 5000.0

# After:
attn.W_q[base + 33, BD.CONST] = -50000.0
attn.W_q[base + 33, BD.H1 + AX_IDX] = 50000.0
```

This creates much stronger suppression:
- Score contribution: Q[33] * K[33] / sqrt(64) ≈ -50000 * 5 / 8 ≈ -31250
- Makes leakage negligible even after softmax

## Test Results

### Before Fix
```
IMM 5:  97.1% (34/35 tokens) - PC byte 1 mismatch
PSH:    97.1% (34/35 tokens) - PC byte 1 mismatch
ADD:    97.1% (34/35 tokens) - PC byte 1 mismatch
JMP 12: 94.3% (33/35 tokens) - PC byte 0+1 mismatch
```

### After Fix
```
IMM 5:  100.0% (35/35 tokens) ✓
PSH:    100.0% (35/35 tokens) ✓
ADD:    100.0% (35/35 tokens) ✓
JMP 12:  97.1% (34/35 tokens) - PC byte 0 mismatch is expected (one-step delay by design)
```

## Note on JMP One-Step Delay
JMP has intentional one-step delay in neural VM (by design, comment at vm_step.py:1909-1912):
- Step N: Executes JMP, outputs old PC+8
- Step N+1: JMP relay fires, overrides with target PC

This is different from DraftVM which immediately sets PC to target.
Single-step tests will always show a mismatch for JMP - use two-step tests to validate.

## Files Modified
- `neural_vm/vm_step.py`: Line 3038-3039 (anti-leakage weight increase)
