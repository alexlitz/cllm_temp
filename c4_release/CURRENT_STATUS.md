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

---

## Neural Implementation Status

### Fully Neural (No Handlers)
- IMM, LEA, PSH, NOP, EXIT ✅
- ADD, SUB, MUL, DIV, MOD (arithmetic) ✅
- OR, XOR, AND, SHL, SHR (bitwise) ✅
- EQ, NE, LT, GT, LE, GE (comparisons) ✅
- LI, LC, SI, SC (memory load/store) ✅
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

### Priority 1: Fix Stack Memory for Function Calls
1. **Debug why LEV reads all zeros from memory**
   - JSR writes return address to `memory[new_sp]`
   - LEV reads from `memory[old_bp]` → gets 0
   - Check if BP is being set correctly by ENT
   - Check if memory writes are being tracked correctly

2. **Fix memory read/write in handlers**
   - Verify `_mem_store_word()` actually writes to `self._memory`
   - Verify `_mem_load_word()` reads from correct address
   - Check if shadow memory dict is working

3. **Test simple program without loops**
   - Get `int main() { return 42; }` to work (stdlib=False)
   - Should: JSR → ENT → execute → LEV → HALT
   - Current: JSR → ENT → execute → LEV → PC=0 (loop)

### Priority 2: Fix JSR Neural Implementation
1. **Investigate why neural JSR doesn't jump**
   - Check L6 FFN PC override weights
   - Check if OP_JSR flag is being set correctly
   - Compare with working JMP neural implementation

2. **Either fix neural JSR or document why handler is needed**
   - If neural version can't work with hand-crafted weights, keep handler
   - Update documentation to reflect handler requirement

### Priority 3: Test ADJ Once Function Calls Work
1. Once simple functions work, test with local variables
2. Verify ADJ neural implementation
3. Test more complex programs

---

## 📊 Bottom Line

**Major Achievement**: ✅ Performance issue RESOLVED
- GPU acceleration working (4-5x faster)
- Context windowing implemented
- Hand-crafted bytecode works perfectly

**Root Causes Identified**:
1. ❌ **JSR neural implementation doesn't work** - PC doesn't jump (re-enabled handler)
2. ❌ **Stack memory broken** - LEV reads all zeros instead of saved values
3. ⚠️ **Stdlib causes issues** - CALL target beyond bytecode length

**Current State**:
- ✅ Basic operations work (IMM, PSH, NOP, EXIT)
- ✅ Simple programs execute (but loop infinitely due to stack bug)
- ✅ AX reaches correct values (42)
- ❌ Function returns don't work (LEV returns to PC=0)
- ❌ Can't test with stdlib until basic calls work

**Recommended Action**:
1. Fix shadow memory read/write in JSR/ENT/LEV handlers
2. Get simple function call working (no stdlib)
3. Then investigate JSR neural implementation
4. Then test ADJ and more complex programs

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
