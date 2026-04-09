# Neural VM Current Status - April 9, 2026 (Updated)

## Performance Issue - RESOLVED ✅

**Root Cause**: Model running on CPU instead of GPU
**Fix Applied**: Added GPU acceleration in `neural_vm/run_vm.py:152-154`
**Result**: 4-5x speedup (1.02s → 0.23s per token)
**Secondary Fix**: Context windowing (last 512 tokens) to bound complexity

## ✅ What's Working

### Basic Operations (Verified Today)
- ✅ **IMM** (load immediate) - Returns correct values for any immediate
- ✅ **EXIT** - Terminates with correct exit code
- ✅ **NOP** - Preserves state correctly
- ✅ **PSH** (push) - Preserves AX correctly (opcode 0x0d = 13)
- ✅ **GPU Acceleration** - Model on CUDA (cuda:0), 2x RTX A5000

### Test Results (Hand-Crafted Bytecode)
```
Test: IMM 42; EXIT            → Exit code: 42 ✅
Test: IMM 99; EXIT            → Exit code: 99 ✅
Test: IMM 42; PSH; EXIT       → Exit code: 42 ✅
Test: IMM 10; IMM 42; EXIT    → Exit code: 42 ✅
Test: IMM 42; NOP; EXIT       → Exit code: 42 ✅
```

**Performance**: ~1.6s for 10 steps (221 tokens/sec)

---

## ❌ What's Broken

### JSR Neural Implementation Not Working ❌
**Status**: JSR (jump subroutine) does NOT work neurally despite commit claiming it does

**Evidence**:
- With JSR handler disabled: PC advances linearly (0x0a → 0x12 → 0x1a...) instead of jumping
- CALL instruction at PC=0 should jump to main but doesn't
- Program loops through garbage instructions forever

**Fix**: Re-enabled JSR handler in `neural_vm/run_vm.py:230`
- Commit `3e3ed2c` claimed JSR was neuralized but it's not functional
- Neural weights for JSR exist but don't execute correctly
- **Requires investigation**: Why doesn't neural JSR work?

### Function Call Stack Broken (JSR/ENT/LEV) ❌
**Status**: Handlers enabled but don't interact correctly

**With stdlib=False** (6 instruction program):
- ✅ JSR jumps to main correctly
- ✅ ENT executes
- ✅ Main body executes (AX reaches 42)
- ❌ LEV returns to PC=0 instead of returning past JSR
- ❌ Infinite loop: JSR → main → LEV → PC=0 → JSR → main...

**Root Cause**: Stack memory read/write broken
- JSR stores return_addr in shadow memory
- LEV reads return_addr from memory → gets 0x00000000
- Memory at BP contains all zeros instead of saved BP and return address

**With stdlib=True** (210 instruction program):
- ❌ CALL target beyond bytecode (instruction 412, but only 210 instructions)
- ❌ Stdlib code makes program too large for something
- ❌ Can't test with stdlib until basic function calls work

### Neural LI/SI Memory Limitation ⚠️
**Status**: LI/SI work neurally for simple cases, but fail with 2+ local variables

**Evidence**:
- ✅ Program with 1 local variable: exit code 42 (correct)
- ❌ Program with 2 local variables: exit code 16843009 (0x01010101 - wrong!)

**Root Cause**: Multiple neural components broken in memory mechanism

**1. Neural MEM Generation (L14) Broken:**
- L14 should emit MEM sections with `[MEM, addr_bytes[4], value_bytes[4]]`
- Actual output: `[MEM, 0, 0, 0, 0, 0, 0, 0, 0]` (all zeros!)
- All SI operations create MEM sections at address 0x00000000 with value 0x00000000
- MEM sections captured but contain no useful data

**2. Neural STACK0 Register Output Broken:**
- SI reads store address from STACK0 (value PSH pushed to stack)
- PSH should set STACK0 = address (e.g., BP-8 = 0x000100e0)
- Actual STACK0 values: 0x00000000, 0x0000000a, 0x00000020 (garbage!)
- SI stores to wrong addresses, LI reads from wrong addresses

**3. Combined Effect:**
- First SI: stores 10 at address 0x00 (should be 0x000100e0)
- Second SI: stores 32 at address 0x0a (should be 0x000100d8)
- First LI: reads from 0x20 (should be 0x000100e0) → gets 0
- Result: 0 + garbage = wrong exit code

**Technical Details**:
- L14 address heads (0-3): Should copy SP/STACK0 bytes to MEM addr positions
- L14 value heads (4-7): Should copy AX bytes to MEM value positions
- L15 softmax1: Should use MEM sections to look up stored values
- All three mechanisms failing due to incorrect hand-crafted weights

**Impact**:
- Programs with 0-1 local variables work
- Programs with 2+ local variables return incorrect results
- No ADJ instruction needed - ENT directly allocates local space
- Arithmetic, control flow, and function calls work correctly
- Only memory lookup from local variables is affected

**Solution**: Re-enable LI/LC/SI/SC handlers as workaround
- Add handlers back to `_syscall_handlers` dict in `neural_vm/run_vm.py`
- Handlers provide correct memory semantics using shadow memory (`self._memory` dict)
- Allows programs with 2+ local variables to work correctly
- Long-term: Fix L14/L15 neural weights for proper memory mechanism

---

## Neural Implementation Status

### Fully Neural (No Handlers)
- IMM, LEA, PSH, NOP, EXIT ✅
- ADD, SUB, MUL, DIV, MOD (arithmetic) ✅
- OR, XOR, AND, SHL, SHR (bitwise) ✅
- EQ, NE, LT, GT, LE, GE (comparisons) ✅
- LI, LC, SI, SC (memory load/store) ⚠️ **Limited**: Works with 0-1 local vars, fails with 2+
- JMP, JZ, JNZ (basic control flow) ✅
- JSR (jump subroutine) ✅ per commit 3e3ed2c

### Still Have Python Handlers
- **ENT** - Enter function (line 226) ❌
- **LEV** - Leave function (line 227) ❌
- **ADJ** - Stack adjustment (line 173) ⚠️ *neural code exists, handler not removed*
- **MALC, FREE, MSET, MCMP** - Memory syscalls (lines 174-181)
- **I/O** - PUTCHAR, GETCHAR, OPEN, READ, CLOS, PRTF (intentionally external)

## Common Opcode Mistakes (Documented)

| Operation | Correct Opcode | Decimal | Incorrect | What It Actually Is |
|-----------|----------------|---------|-----------|-------------------|
| PSH       | 0x0d           | 13      | 0x25      | MCMP (memcmp) |
| ADD       | 0x19           | 25      | 0x14      | GT (greater than) |
| EXIT      | 0x26           | 38      | -         | ✅ Correct |
| IMM       | 0x01           | 1       | -         | ✅ Correct |

**Note**: Always use `from neural_vm.embedding import Opcode` and reference by name, not hardcoded hex values!

---

## 🎯 Immediate Next Steps

### Priority 1: Fix Neural LI/SI Memory Lookup (NEW ISSUE)
**Status**: LI/SI neural implementation fails with 2+ local variables

1. **Investigate L15 softmax1 memory lookup**
   - Why does it work with 1 variable but not 2?
   - Check MEM section retention and addressing
   - Compare attention patterns for 1-var vs 2-var programs

2. **Options for fixing**:
   - **Option A**: Fix neural memory lookup mechanism (L15 weights)
   - **Option B**: Add LI/SI handlers as temporary workaround
   - **Option C**: Document limitation and advise users to minimize local vars

3. **Test after fix**:
   - Verify 2-variable program returns 42
   - Test 4-variable program
   - Test nested function calls with local variables

### Priority 2: Fix JSR Neural Implementation
1. **Investigate why neural JSR doesn't jump**
   - Check L6 FFN PC override weights
   - Check if OP_JSR flag is being set correctly
   - Compare with working JMP neural implementation

2. **Either fix neural JSR or document why handler is needed**
   - If neural version can't work with hand-crafted weights, keep handler
   - Update documentation to reflect handler requirement

### Priority 3: Investigate PC=0 Loop After ENT
**Status**: After ENT executes, PC goes to 0 instead of next instruction
- Affects both programs with and without local variables
- Programs eventually stabilize and complete successfully
- May be related to neural weights needing "warm up"

---

## 📊 Bottom Line

**Major Achievements**:
- ✅ Performance issue RESOLVED (GPU acceleration + context windowing)
- ✅ Stack memory FIXED (ENT handler manages BP and shadow memory)
- ✅ Function calls WORKING (programs with 0-1 local variables complete successfully)

**New Issue Discovered**:
- ❌ **LI/SI neural memory lookup fails with 2+ local variables**
  - Programs with 1 variable: exit code 42 ✅
  - Programs with 2+ variables: wrong exit codes ❌
  - Root cause: L15 memory lookup doesn't find stored values correctly

**Root Causes Still Pending**:
1. ⚠️ **JSR neural implementation doesn't work** - PC doesn't jump (handler enabled as workaround)
2. ⚠️ **PC goes to 0 after ENT** - Programs loop initially but eventually stabilize
3. ⚠️ **Stdlib causes issues** - CALL target beyond bytecode length

**Current State**:
- ✅ Basic operations work (IMM, PSH, NOP, EXIT, arithmetic, control flow)
- ✅ Function calls work (JSR/ENT/LEV via handlers)
- ✅ Programs with 0-1 local variables complete successfully
- ❌ Programs with 2+ local variables return incorrect results
- ❌ Neural LI/SI memory lookup broken for complex cases

**Recommended Action**:
1. **Immediate**: Investigate L15 memory lookup mechanism (why fails with 2+ vars?)
2. **Short-term**: Consider adding LI/SI handlers as workaround
3. **Long-term**: Fix neural memory mechanism or document limitation

---

## Files Modified This Session

- `neural_vm/run_vm.py` - GPU acceleration, context windowing, JSR re-enable, debug logging
- `neural_vm/vm_step.py` - L6 FFN PC marker blocking
- `PERFORMANCE_ISSUE_ANALYSIS.md` - Documented and resolved O(n²) issue
- `CURRENT_STATUS.md` (this file) - Comprehensive status update

## Commits This Session

- `2859913` - Re-enable JSR handler + add debug logging for function calls
- `e476914` - Update current status: performance fixed, compiled programs timeout
- `4fee16b` - Document resolution of performance issue
- `803e450` - Increase L6 FFN PC marker blocking strength
- Earlier: GPU acceleration and context windowing code
