# PC Byte Swap Bug - Root Cause Analysis

## Summary
The Neural VM predicts wrong PC bytes due to multiple compounding bugs in the weight-setting code, causing a systematic byte swap pattern observed across all opcodes.

## Bug Chain

### Bug 1: L3 FFN First-Step Default (vm_step.py:1602)
**Location**: `_set_layer3_ffn`, line 1602
**Issue**: Writes to `OUTPUT_LO + 2` instead of `OUTPUT_LO + 0`
```python
# Line 1602 - INCORRECT
ffn.W_down[BD.OUTPUT_LO + 2, unit] = 2.0 / S  # PC=2 for fetching
```

**Effect**: Unit 0 writes to OUTPUT_LO[1] in the actual weight matrix, encoding PC byte=1 instead of PC byte=0

**Comment mismatch**: Code comment says "PC=2" but formula `idx * 5 + 2` for idx=0 gives PC=2, which should encode byte value 2, not write to OUTPUT_LO[2]

### Bug 2: L6 FFN PC Increment (vm_step.py:2127-2135)
**Location**: `_set_layer6_routing_ffn`, lines 2120-2135
**Issue**: Increments PC by +8 instead of +5

```python
# Lines 2127-2135 - INCORRECT
# Comment says "add 8" but should be "add 5"
for k in range(16):
    new_k = (k + 8) % 16  # SHOULD BE (k + 5) % 16
```

**Effect**: Creates units 224-239 (including unit 256 at k=0→new_k=8) that rotate OUTPUT_LO nibbles by +8 instead of +5

**Comment**: Line 2122 says "Each instruction is 8 bytes" but line 1592 correctly says "Each instruction is 5 tokens"

## Observed Behavior

### First Step Execution:
1. **L3 FFN**: Sets OUTPUT_LO[0]=1.0 (instead of OUTPUT_LO[2]=1.0)
   - Encodes PC byte value 0 (incorrect, should be 2)
2. **L6 FFN (unit 256)**: Rotates OUTPUT_LO[0]→OUTPUT_LO[8]
   - Adds +8: byte 0 + 8 = byte 8
   - Should add +5: byte 0 + 5 = byte 5
3. **Final OUTPUT**: OUTPUT_LO[8]=1.0, OUTPUT_HI[0]=1.0
   - Encodes PC byte value 8
   - Expected: byte 2 (from formula `0*5+2=2`)

### Layer-by-Layer Trace:
```
L0-L2: OUTPUT_LO all zero
L3:    OUTPUT_LO[0] = 1.0  ← Bug 1: wrong index
L4-L5: OUTPUT_LO[0] = 1.0  (unchanged)
L6:    OUTPUT_LO[8] = 1.0  ← Bug 2: +8 rotation
L7-L15: OUTPUT_LO[8] = 1.0 (preserved)
```

## Files Affected
- `neural_vm/vm_step.py`:
  - Line 1602: `_set_layer3_ffn` (first-step default)
  - Lines 2127-2135: `_set_layer6_routing_ffn` (PC increment)
  - Line 1591-1592: Comment mentions PC=2 but unclear if formula or byte index

## Test Results
- Baseline test: 94.3% token match rate (33/35 tokens)
- PC bytes 0 and 1 consistently swapped across ALL opcodes
- Other 33 tokens predict correctly

## Required Fixes
1. **Fix L3 FFN unit 0**: Write to correct OUTPUT_LO index
   - Clarify whether PC=2 means "byte value 2" or "instruction at address 2"
   - Adjust OUTPUT_LO index accordingly
2. **Fix L6 FFN PC increment**: Change +8 to +5
   - Line 2129: `new_k = (k + 5) % 16`
   - Update comment on line 2122
3. **Verify consistency**: Ensure L3 increment (+5), L6 increment (+5), and DraftVM (`idx*5+2`) all align

## Notes
- S=100 in actual weights (not S=50 as might be expected from some comments)
- Unit indexing is correct; the bug is in which OUTPUT_LO index is written
- The +8 increment appears to be a design remnant from when instructions were 8 bytes instead of 5 tokens
