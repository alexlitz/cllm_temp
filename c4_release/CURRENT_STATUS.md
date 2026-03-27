# Neural VM PC Addressing Fix - Current Status

## Problem Identified
The Neural VM was using a 5-token instruction format with PC formula `idx*5+2`, but needed to be updated to 8-byte format with `PC = idx*8`.

## Fixes Applied to neural_vm/vm_step.py

### 1. ADDR_KEY Injection (Line 536) ✓ APPLIED
```python
# Changed from:
addr = i - cs_pos - 1 + 2  # Add 2 to match DraftVM's PC offset
# To:
addr = i - cs_pos - 1  # Addresses start at 0
```

### 2. L3 FFN (Lines 1589-1658) ✓ APPLIED
- Sets PC=0 for first step (not PC=2)
- Passthrough for subsequent steps (no PC increment in L3)

### 3. L4 FFN (Lines 1683-1727) ✓ APPLIED  
- Computes PC+1 for immediate fetch (in TEMP)
- Keeps PC for opcode fetch (in EMBED)

### 4. L5 Fetch Comments (Lines 1730-1776) ✓ APPLIED
- Head 0: Fetches immediate from PC+1
- Head 1: Fetches opcode from PC

### 5. L6 FFN PC Increment (Inserted before line 1937) ✓ APPLIED
- Adds +8 increment for all steps
- Handles nibble wrap and carry correctly

## Fixes NEEDED for neural_vm/speculative.py

### DraftVM PC Formula - ❌ KEEPS GETTING REVERTED

The file neural_vm/speculative.py keeps getting reverted by a linter or pre-commit hook.

**Required changes:**
```python
# Line 76 - Change from:
self.pc = 2           # PC = idx * 8 + 2
# To:
self.pc = 0           # PC = idx * 8

# Line 110 - Change from:
self.pc = self.idx * 8 + 2
# To:
self.pc = self.idx * 8

# Lines 120, 125, 129, 133, 148 - Change ALL occurrences of:
(imm - 2) // 8
# or:
(ret_addr - 2) // 8
# To:
imm // 8
# and:
ret_addr // 8
```

## Test Results

### Manual Tests (debug scripts) ✓ PASS
- Step 0 (IMM 42): PC=8, AX=42 ✓
- Step 1 (PSH): Opcode fetch correct, PC carry-forward=8 ✓

### Automated Tests: ❌ FAIL
- Fails because speculative.py keeps getting reverted to PC=idx*8+2

## Next Steps

1. **Disable or identify the linter** that's reverting speculative.py
2. **Manually apply fixes** to speculative.py and verify they stick
3. **Run comprehensive tests** once speculative.py is fixed
4. **Update batch_runner.py** if it has similar PC formula issues

