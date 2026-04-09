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

### Compiled Programs Timeout ❌
**Status**: Any program compiled with `compile_c()` hangs/times out

**Symptoms**:
- `int main() { return 42; }` - times out (never completes)
- `int main() { int a; a = 42; return a; }` - times out
- Function calls with ADJ - times out

**Likely Causes**:
1. **ENT/LEV handlers** - All compiled functions (including `main()`) use ENT for stack frame setup
2. **Excessive steps** - Programs may need >> 200 steps to complete
3. **Handler conflicts** - Python handlers may interfere with neural execution

### ENT/LEV Still Have Handlers ⚠️
**Location**: `neural_vm/run_vm.py` lines 226-227
**Status**: Not neuralized yet
**Impact**: Blocks testing of:
  - ADJ neural implementation
  - Function calls
  - Any compiled C code (even simple programs)

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

### Priority 1: Investigate Compiled Program Timeouts
1. Add extensive debug logging to ENT/LEV handlers
2. Try running with max_steps=1000+ to see if programs eventually complete
3. Check if ENT/LEV are causing infinite loops
4. Consider temporarily removing ENT/LEV handlers to test pure neural path

### Priority 2: Test ADJ Once Programs Run
1. Verify ADJ neural implementation with function calls
2. Remove ADJ handler from line 173 if neural version works
3. Document ADJ as fully neural

### Priority 3: Neuralize ENT/LEV (Per REMAINING_HANDLERS_PLAN.md)
1. Study ENT/LEV operations
2. Design neural implementation (multi-byte BP push/pop)
3. Implement weights
4. Test and validate
5. Remove handlers

---

## 📊 Bottom Line

**Major Achievement**: ✅ Performance issue RESOLVED
- GPU acceleration working (4-5x faster)
- Context windowing implemented
- Simple programs execute successfully

**Current Blocker**: ❌ Compiled programs timeout
- ENT/LEV handlers may be incompatible
- Can't test ADJ, function calls, or most C programs
- Hand-crafted bytecode works fine

**Recommended Action**: Debug why compiled programs hang before proceeding with further neural implementation work.

---

## Files Modified This Session

- `neural_vm/run_vm.py` - GPU acceleration, context windowing
- `neural_vm/vm_step.py` - L6 FFN PC marker blocking
- `PERFORMANCE_ISSUE_ANALYSIS.md` - Documented O(n²) resolution
- `CURRENT_STATUS.md` (this file) - Updated status

## Commits This Session

- `4fee16b` - Document resolution of performance issue
- `803e450` - Increase L6 FFN PC marker blocking strength
- Earlier: GPU acceleration and context windowing code
