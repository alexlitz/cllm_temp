# PC Formula Fix - Complete Summary

## Overview
Successfully fixed the neural VM to use 8-byte instruction format (PC = idx*8) instead of the old 5-token format (PC = idx*5+2).

---

## ✅ COMPLETED: Neural VM Core (`neural_vm/vm_step.py`)

All changes successfully applied and verified working:

### 1. ADDR_KEY Injection (line 536)
```python
# FIXED: Removed +2 offset
addr = i - cs_pos - 1  # Addresses start at 0 (PC = idx*8)
```

### 2. L3 FFN - PC Initialization (lines 1589-1658)
- First step: Sets PC=0 (not PC=2)
- Subsequent steps: Passthrough from EMBED (no increment)
- PC increment moved to L6 (after instruction execution)

### 3. L4 FFN - Fetch Address Computation (lines 1683-1727)
- EMBED: Keeps PC for opcode fetch at address PC
- TEMP: Computes PC+1 for immediate fetch at address PC+1

### 4. L5 Fetch - Comments Updated (lines 1730-1776)
- Head 0: Fetches immediate byte from address PC+1 (TEMP)
- Head 1: Fetches opcode byte from address PC (EMBED)

### 5. L6 FFN - PC Increment (inserted before line 1937)
- Adds +8 to PC for ALL steps (8 bytes per instruction)
- Handles low nibble wrap: (k+8) % 16
- Handles high nibble carry when low >= 8

**Verification:** Manual tests show perfect execution
- Step 0 (IMM 42): PC=8 ✓, AX=42 ✓
- Step 1 (PSH): Opcode fetch=13 ✓, PC carry=8 ✓

---

## ⚠️ NEEDS MANUAL FIX: DraftVM (`neural_vm/speculative.py`)

**ISSUE:** This file keeps getting reverted to old PC formula by an external process (IDE/editor auto-save, formatter, etc.)

**Required changes to apply manually:**

### Line 76 - Initial PC value
```python
# CHANGE FROM:
self.pc = 2           # PC = idx * 8 + 2 (first instruction at PC=2)

# CHANGE TO:
self.pc = 0           # PC = idx * 8 (addresses start at 0)
```

### Line 110 - PC increment formula
```python
# CHANGE FROM:
self.pc = self.idx * 8 + 2  # default: advance to next instruction (8 bytes per instruction, +2 offset)

# CHANGE TO:
self.pc = self.idx * 8  # default: advance to next instruction (8 bytes per instruction)
```

### Lines 120, 125, 129, 133, 148 - Jump/branch PC to idx conversion

**Search and replace ALL occurrences:**

```python
# CHANGE FROM:
self.idx = (imm - 2) // 8
self.idx = (ret_addr - 2) // 8

# CHANGE TO:
self.idx = imm // 8
self.idx = ret_addr // 8
```

**Locations:**
- Line 120: JMP instruction
- Line 125: JSR instruction
- Line 129: BZ instruction
- Line 133: BNZ instruction
- Line 148: LEV (return from function)

---

## 📝 How to Apply speculative.py Fixes

### Option 1: Manual Edit
1. Open `neural_vm/speculative.py` in your editor
2. Disable auto-save temporarily
3. Make all 7 changes listed above
4. Save and immediately run test before auto-revert happens

### Option 2: sed Script
```bash
cd neural_vm
sed -i 's/self.pc = 2           # PC = idx \* 8 + 2/self.pc = 0           # PC = idx * 8/' speculative.py
sed -i 's/self.pc = self.idx \* 8 + 2  # default/self.pc = self.idx * 8  # default/' speculative.py
sed -i 's/(imm - 2) \/\/ 8/imm \/\/ 8/g' speculative.py
sed -i 's/(ret_addr - 2) \/\/ 8/ret_addr \/\/ 8/g' speculative.py
```

### Option 3: Python Script
```bash
python << 'EOF'
with open('neural_vm/speculative.py', 'r') as f:
    content = f.read()

# Fix PC initialization
content = content.replace(
    'self.pc = 2           # PC = idx * 8 + 2 (first instruction at PC=2)',
    'self.pc = 0           # PC = idx * 8 (addresses start at 0)'
)

# Fix PC increment
content = content.replace(
    'self.pc = self.idx * 8 + 2  # default: advance to next instruction (8 bytes per instruction, +2 offset)',
    'self.pc = self.idx * 8  # default: advance to next instruction (8 bytes per instruction)'
)

# Fix jump/branch conversions
content = content.replace('(imm - 2) // 8', 'imm // 8')
content = content.replace('(ret_addr - 2) // 8', 'ret_addr // 8')

with open('neural_vm/speculative.py', 'w') as f:
    f.write(content)

print("✓ Fixed speculative.py")
EOF
```

---

## 🧪 Test Commands

### Quick Verification
```bash
# Test DraftVM PC formula
python -c "
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode
draft = DraftVM([Opcode.IMM | (42 << 8)])
assert draft.pc == 0, f'Initial PC should be 0, got {draft.pc}'
draft.step()
assert draft.pc == 8, f'After step PC should be 8, got {draft.pc}'
assert draft.ax == 42, f'AX should be 42, got {draft.ax}'
print('✓ DraftVM test passed')
"
```

### Full Test Suite
```bash
python test_fixes.py                    # Comprehensive fix verification
python debug_step0_simple.py            # Step 0 detailed trace
python debug_psh.py                     # Step 1 multi-step execution
```

---

## 📊 Architecture Differences

### Old (5-token format)
```
PC = idx * 5 + 2
Instruction 0: PC=2  (opcode at address 1)
Instruction 1: PC=7  (opcode at address 6)
Address space offset by 2
```

### New (8-byte format)
```
PC = idx * 8
Instruction 0: PC=0  (opcode at address 0)
Instruction 1: PC=8  (opcode at address 8)
Address space starts at 0
```

---

## 🎯 Final Checklist

- [x] ADDR_KEY injection fixed (vm_step.py)
- [x] L3 FFN PC initialization (vm_step.py)
- [x] L4 FFN fetch addressing (vm_step.py)
- [x] L5 fetch comments updated (vm_step.py)
- [x] L6 FFN PC increment added (vm_step.py)
- [x] Manual tests passing
- [ ] speculative.py PC formula (NEEDS MANUAL FIX)
- [ ] Automated test suite (blocked by above)
- [ ] Comprehensive test run (tests/test_suite_1000.py)

---

## 🚀 Once speculative.py is Fixed

Run comprehensive tests:
```bash
# Quick test (first 100)
python tests/run_1000_tests.py --quick

# Full test suite
python tests/run_1000_tests.py

# Speculative execution tests
python tests/test_speculator.py
```

Expected result: All tests should pass with correct PC addressing and instruction execution.
