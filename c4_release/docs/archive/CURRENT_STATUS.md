# Neural VM Current Status - April 9, 2026 (Updated 19:45)

## 🔍 ROOT CAUSE IDENTIFIED: AX_CARRY Population Missing

**Date**: 2026-04-09 19:45
**Investigation**: Deep-dive into PSH/STACK0 neural weights

**CRITICAL FINDING**: L14 fix (commit ea8718f) was **necessary but NOT sufficient**.

**ROOT CAUSE**: **AX_CARRY is not populated with the current AX value**, breaking PSH's ability to set STACK0 = AX neurally.

**Evidence**:
```
Test: int main() { return 42; }                                  → Exit 42 ✅ (no variables)
Test: int main() { int a; a=42; return a; }                      → Exit 0  ❌ (1 variable)
Test: int main() { int a, b; a=10; b=32; return a+b; }           → Exit 0  ❌ (2 variables)
Test: int main() { int a, b, c, d; ... }                         → Exit 0  ❌ (4 variables)

MEM sections: [261, 0, 0, 0, 0, 0, 0, 0, 0]  (all zeros!)
SI handlers extract: addr=0x00000000 (should be 0x000100e0)
```

**Historical Context** (vm_step.py:1660-1663):
> "Attempted fix for JMP/NOP/EXIT AX corruption by adding head 5 to copy previous AX marker OUTPUT to AX_CARRY. This broke pure neural mode (predictions became all 1's), so the fix was reverted. **The bug persists but system remains functional in hybrid mode.**"

**Current State**: Neural VM stuck in "hybrid mode" (requires handlers for PSH/STACK0/memory ops).

**Details**: See `PSH_STACK0_ROOT_CAUSE_ANALYSIS.md`

---

## 🎉 BREAKTHROUGH: Stack Memory Bug PARTIALLY FIXED! (Commit ea8718f)

**Critical Fix Applied**: L14 MEM generation now reads OUTPUT instead of CLEAN_EMBED

**Root Cause**: L14 MEM val heads were reading from CLEAN_EMBED (old token embeddings) instead of OUTPUT (updated register values). This caused LEV to read zeros from memory instead of saved BP/return_addr.

**Fix**: Changed `neural_vm/vm_step.py:5830-5831` to read from OUTPUT_LO/HI dims.

**Test Results**:
- ✅ Handcrafted bytecode (JSR → ENT → IMM 42 → LEV → EXIT): **Exit code 42** (PASS!)
- ✅ Stack memory now works correctly for basic function calls
- ⏭️ Stdlib-compiled programs (210 instructions): Still under investigation

**Impact**: Unblocks path to 100% neural VM - JSR/ENT/LEV handlers can now be removed after further testing.

**Details**: See `SESSION_SUMMARY_L14_FIX.md`

---

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

**Solution Attempted**: Re-enable LI/LC/SI/SC handlers ❌ **DOESN'T WORK**
- Handlers added back to `_syscall_handlers` dict (lines 174-177)
- BUT handlers depend on reading STACK0 register from model output
- SI handler reads addr from `_extract_stack0()` → gets garbage (0x00, 0x0a, 0x20)
- LI handler reads addr from `_extract_register(AX)` → gets garbage
- **Root cause**: Neural PSH doesn't set STACK0 correctly, so handlers fail too

**Real Solution Required**: Fix neural PSH/STACK0 register generation
- PSH should output STACK0 = address pushed (e.g., BP-8 = 0x000100e0)
- Currently outputs garbage values instead of correct addresses
- This is a fundamental neural weight issue in L6/L7 layers
- Handlers cannot work around this - they need correct register values from model

---

## Neural Implementation Status

### Fully Neural (No Handlers) - Actually Working
- IMM, LEA, NOP, EXIT ✅
- ADD, SUB, MUL, DIV, MOD (arithmetic) ✅
- OR, XOR, AND, SHL, SHR (bitwise) ✅
- EQ, NE, LT, GT, LE, GE (comparisons) ✅
- JMP, JZ, JNZ (basic control flow) ✅

### Claimed Neural But Actually Broken
- **PSH** ❌ - Neural weights exist but don't set STACK0 correctly (AX_CARRY not populated)
- **LI, LC, SI, SC** ❌ - Depend on STACK0, which PSH doesn't set correctly
- **JSR** ❌ - per commit 3e3ed2c claimed neural, but PC doesn't jump correctly (handler re-enabled)

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

### Priority 1: Fix AX_CARRY Population (ROOT CAUSE)
**Status**: ✅ **ROOT CAUSE IDENTIFIED** - AX_CARRY not populated with current AX value

**The Problem**:
- L3 head 1 copies only previous step's AX byte 0 EMBED → AX_CARRY
- PSH needs current step's full AX OUTPUT in AX_CARRY
- Without correct AX_CARRY, PSH can't set STACK0 = AX
- Cascading failure: STACK0 wrong → SI/LI wrong → MEM wrong → all zeros

**Previous Fix Attempt (FAILED)**:
- Tried adding L3 head 5 to copy AX OUTPUT → AX_CARRY
- Result: "predictions became all 1's" (catastrophic failure)
- Fix was reverted, bug accepted as requiring hybrid mode

**Options Going Forward**:
1. **Investigate failed L3 head 5 fix**
   - Find git commit that attempted the fix
   - Understand why it caused "all 1's" predictions
   - Try alternative approach to avoid the failure mode

2. **Use different layer for AX relay**
   - L4 attention (currently PC → AX relay only)
   - L5 attention (currently opcode decode)
   - L6 attention (add intra-step AX OUTPUT → AX_CARRY)

3. **Redesign PSH mechanism**
   - Make PSH read directly from AX OUTPUT (not AX_CARRY → ALU)
   - Requires rewriting L6 head 6 and L6 FFN PSH units
   - High risk architectural change

4. **Accept hybrid mode (current state)**
   - Keep handlers for PSH/SI/LI/SC operations
   - Document that pure neural mode doesn't work for 2+ variables
   - System functional but not 100% neural

### Priority 2: Fix JSR Neural Implementation
**Status**: Handler required (neural path broken)
- Investigate why neural JSR doesn't jump PC correctly
- May be related to AX_CARRY issue or separate problem

### Priority 3: Test L14 Fix with Handlers
**Status**: Pending - verify if L14 fix + handlers work correctly
- Re-enable LI/SI handlers
- Test 2+ variable programs
- Verify MEM sections populated correctly when handlers assist

---

## 📊 Bottom Line

**Investigation Complete** ✅:
- ✅ **ROOT CAUSE IDENTIFIED**: AX_CARRY not populated with current AX value
- ✅ **L14 fix necessary but insufficient**: Reads OUTPUT correctly, but OUTPUT contains garbage
- ✅ **Historical context found**: Previous fix attempt failed catastrophically ("all 1's")
- ✅ **System stuck in hybrid mode**: Pure neural broken, handlers required

**What Works** ✅:
- Basic operations (IMM, NOP, EXIT, arithmetic, bitwise, comparisons, basic control flow)
- Programs without local variables
- GPU acceleration and performance optimizations

**What's Broken** ❌:
- **PSH**: Can't set STACK0 = AX (AX_CARRY not populated)
- **SI/SC**: Can't use STACK0 for address (depends on PSH)
- **LI/LC**: Can't use STACK0 for address (depends on PSH)
- **Programs with ANY local variables**: All return exit code 0 (1, 2, 4+ variables all fail)
- **JSR**: Neural path doesn't jump PC (handler enabled)

**The Cascade of Failure**:
```
AX_CARRY empty → PSH writes zeros to STACK0 → SI/LI read addr=0x00000000 →
MEM sections all zeros → L15 lookup fails → wrong results
```

**Path Forward** (4 Options):

1. **Fix L3 head 5** (Risky): Re-attempt AX OUTPUT → AX_CARRY, avoid "all 1's" failure
2. **Alternative relay** (Medium): Use L4/L5/L6 for AX relay instead
3. **Redesign PSH** (High risk): Rewrite to use AX OUTPUT directly
4. **Accept hybrid** (Safe): Keep handlers, document limitation

**Current Recommendation**: Option 4 (accept hybrid mode) until root cause of L3 head 5 failure is understood.

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
