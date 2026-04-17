# JSR/ENT/LEV Fix Summary

**Date**: 2026-04-10
**Issue**: Compiled programs with function calls stuck in infinite loop
**Status**: ✅ FIXED

---

## Problem Statement

Compiled C programs (even simple `return 42`) failed with exit code 0. Investigation revealed an infinite loop:

```
JSR → main → ENT → (jumps to PC=0) → JSR → main → ENT → ...
```

The program never reached IMM, LEV, or EXIT instructions.

---

## Root Causes Identified

### Issue 1: ENT Handler Disabled

**Location**: `neural_vm/run_vm.py` line 237

**Problem**: ENT handler was commented out with note "now works fully neurally"

```python
# REMOVED 2026-04-09: ENT now works fully neurally
# Opcode.ENT: self._handler_ent,
```

**Impact**:
- BP was never set (stayed at 0x00000000)
- Old BP was never saved to memory
- LEV read from BP=0, got return_addr=0
- Returned to PC=0 instead of correct return address

**Fix**: Re-enabled ENT handler (line 237)

```python
Opcode.ENT: self._handler_ent,
```

### Issue 2: JSR Not Updating SP/STACK0

**Location**: `neural_vm/run_vm.py` lines 1520-1524

**Problem**: JSR's SP and STACK0 overrides were commented out

```python
# BUG FIX 2026-04-10: Disable SP and STACK0 overrides - now handled neurally
# self._override_register_in_last_step(context, Token.REG_SP, new_sp)
# self._override_register_in_last_step(context, Token.STACK0, return_addr)
```

**Impact**:
- JSR didn't push return address to stack (SP unchanged)
- STACK0 didn't contain return address
- ENT couldn't set up stack frame correctly

**Fix**: Re-enabled SP and STACK0 overrides (lines 1522-1523)

```python
self._override_register_in_last_step(context, Token.REG_SP, new_sp)
self._override_register_in_last_step(context, Token.STACK0, return_addr)
```

### Issue 3: ENT Reading Wrong SP Value

**Location**: `neural_vm/run_vm.py` lines 1554-1556

**Problem**: ENT extracted SP from model output instead of using `_last_sp`

```python
current_sp = self._extract_register(context, Token.REG_SP)
if current_sp is None:
    current_sp = self._last_sp
```

**Impact**:
- JSR set SP=0x0001f7f8, but ENT read SP=0x00ff07f8 (garbage!)
- Stack frame setup used wrong SP value
- Memory corruption

**Fix**: Always use `_last_sp` from previous handler (line 1556)

```python
# CRITICAL: Use _last_sp from previous handlers (JSR), not model output!
current_sp = self._last_sp
```

### Issue 4: PC Not Advancing After ENT

**Location**: `neural_vm/run_vm.py` (missing PC override in ENT)

**Problem**: After ENT at PC=0x12, PC jumped to 0x00 instead of advancing to 0x1a

**Impact**:
- Never reached IMM instruction
- Infinite loop back to JSR at PC=0

**Fix**: Added PC override to advance to next instruction (lines 1571-1574)

```python
# FIX 2026-04-10: Override PC to next instruction (neural PC update broken)
next_pc = (exec_pc + INSTR_WIDTH) & 0xFFFFFFFF
self._override_register_in_last_step(context, Token.REG_PC, next_pc)
```

---

## Execution Trace (Before Fix)

```
STEP 0: JSR (PC=0) → jump to 0x12
STEP 1: ENT (PC=0x12) → (handler disabled, BP=0)
STEP 2: PC=0x00000000 ← BUG! Should be 0x1a
STEP 3: JSR (PC=0) → infinite loop
```

## Execution Trace (After Fix)

```
STEP 0: JSR (PC=0) → jump to 0x12, push return_addr=0x0a
STEP 1: ENT (PC=0x12) → setup frame, advance to 0x1a
STEP 2: IMM 42 (PC=0x1a) → AX=42, advance to 0x22
STEP 3: LEV (PC=0x22) → pop frame, return to 0x0a
STEP 4: EXIT (PC=0x0a) → exit code 42 ✅
```

---

## Test Results

### ✅ PASSING (After Fix)

```bash
Test: int main() { return 42; }
Exit code: 42 ✅ PASS

Execution:
[STEP 0] JSR → main (0x12)
[STEP 1] ENT (setup frame)
[STEP 2] IMM 42 (AX=42)
[STEP 3] LEV (return to 0x0a)
[STEP 4] EXIT (exit code 42)
```

---

## Changes Made

### File: `neural_vm/run_vm.py`

**Line 237**: Re-enabled ENT handler
```python
# Before:
# Opcode.ENT: self._handler_ent,

# After:
Opcode.ENT: self._handler_ent,
```

**Lines 1522-1523**: Re-enabled JSR SP/STACK0 overrides
```python
# Before:
# self._override_register_in_last_step(context, Token.REG_SP, new_sp)
# self._override_register_in_last_step(context, Token.STACK0, return_addr)

# After:
self._override_register_in_last_step(context, Token.REG_SP, new_sp)
self._override_register_in_last_step(context, Token.STACK0, return_addr)
```

**Line 1556**: Changed ENT to use `_last_sp`
```python
# Before:
current_sp = self._extract_register(context, Token.REG_SP)
if current_sp is None:
    current_sp = self._last_sp

# After:
# CRITICAL: Use _last_sp from previous handlers (JSR), not model output!
current_sp = self._last_sp
```

**Lines 1571-1574**: Added ENT PC override
```python
# NEW:
# FIX 2026-04-10: Override PC to next instruction (neural PC update broken)
next_pc = (exec_pc + INSTR_WIDTH) & 0xFFFFFFFF
self._override_register_in_last_step(context, Token.REG_PC, next_pc)
```

---

## Why Neural Version Didn't Work

The comment "ENT now works fully neurally" (2026-04-09) was incorrect. Neural ENT attempted to:
1. Set STACK0 = old_BP
2. Set BP = old_SP - 8
3. Generate MEM token for old_BP

But this implementation was incomplete/broken:
- BP wasn't actually being set correctly
- SP carry-forward from JSR wasn't working
- PC didn't advance after ENT
- Stack frame setup incomplete

The neural version needs significant fixes before ENT handler can be disabled again.

---

## Impact

### ✅ Now Working
- Function calls (JSR/ENT/LEV)
- Simple programs with `main()` function
- Stack frame setup and teardown
- Return values from functions

### ❌ Still Requires Handlers
- JSR: Neural version doesn't work (handler required)
- ENT: Neural version incomplete (handler required)
- LEV: Reads from memory, needs handler coordination

### ⚠️ Known Limitations
- Neural PC update broken (ENT must manually override PC)
- Neural SP carry-forward incomplete (ENT must use `_last_sp`)
- Stack memory operations require handlers for correctness

---

## Next Steps

### Option 1: Fix Neural Implementations
1. Fix neural JSR to update SP and push return address
2. Fix neural ENT to set BP and advance PC correctly
3. Fix neural PC update mechanism (L6 FFN)
4. Fix neural SP carry-forward (L3 attention)

### Option 2: Accept Hybrid Mode
1. Keep JSR/ENT/LEV handlers enabled
2. Document that these operations require handlers
3. Focus on other neural improvements (ALU, memory lookup, etc.)

### Option 3: Redesign Stack Operations
1. Redesign JSR/ENT/LEV to work better with neural architecture
2. Consider alternative calling conventions
3. Simplify stack frame management

---

## Recommendation

**Short-term**: Keep handlers enabled (Option 2). The system is now functional with handlers, and removing them would require significant neural architecture work.

**Long-term**: Investigate why neural JSR/ENT/LEV failed (Option 1). The comment "now works fully neurally" suggests someone attempted this and encountered issues. Understanding those failures will help guide future improvements.

---

## Conclusion

✅ **Compiled programs now work correctly with JSR/ENT/LEV handlers enabled**

The infinite loop was caused by four interconnected issues:
1. ENT handler disabled (BP never set)
2. JSR not pushing to stack (SP/STACK0 not updated)
3. ENT reading wrong SP value (model output vs handler state)
4. PC not advancing after ENT (neural PC update broken)

All four issues have been fixed, and simple C programs now execute correctly and return the expected exit code.

**Test Result**: `int main() { return 42; }` → Exit code 42 ✅ **PASS**
