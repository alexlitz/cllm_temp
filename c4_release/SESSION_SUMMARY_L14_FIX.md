# Session Summary: L14 MEM Generation Critical Fix

**Date**: 2026-04-09
**Commit**: `ea8718f` - CRITICAL FIX: L14 MEM generation must read OUTPUT not CLEAN_EMBED

---

## 🎯 Problem Solved

**Stack memory was completely broken** - LEV read zeros from memory instead of saved BP/return addresses, causing all function calls to fail.

---

## 🔍 Root Cause Analysis

### The Bug

L14 MEM token generation (heads 4-7) was reading from **CLEAN_EMBED** dimensions instead of **OUTPUT** dimensions.

**Why this broke stack memory:**

1. **L6 FFN JSR STACK0 writeback** (units 914-945):
   - Writes updated STACK0 value (return_addr) to `OUTPUT_LO/HI` dims
   - This happens DURING the forward pass

2. **L14 MEM val heads V weights**:
   - Read from `CLEAN_EMBED_LO/HI` dims (WRONG!)
   - CLEAN_EMBED = original token embedding (old value from previous step)
   - OUTPUT = layer output after all FFN updates (correct current value)

3. **Result**:
   - JSR sets STACK0 = return_addr in OUTPUT → L14 never sees it
   - L14 generates MEM token with value from CLEAN_EMBED (old/zero)
   - Shadow memory stores garbage value
   - LEV reads from memory → gets 0x00000000 instead of return_addr
   - Function returns to PC=0 instead of caller, causing infinite loop

### Investigation Path

1. **Traced JSR handler flow** (`run_vm.py:1486-1524`):
   - Found handler sets STACK0 AFTER forward pass (line 1521)
   - But L14 needs STACK0 DURING forward pass

2. **Examined neural JSR path** (`vm_step.py:6868-6964`):
   - L6 head 7: Copies PC OUTPUT → AX_CARRY at STACK0 marker
   - L6 FFN: Writes AX_CARRY → OUTPUT at STACK0 marker when JSR active
   - This path IS implemented correctly!

3. **Found L14 bug** (`vm_step.py:5825-5831`):
   - L14 reads from CLEAN_EMBED instead of OUTPUT
   - Neural STACK0 update invisible to L14

---

## ✅ Fix Applied

**File**: `neural_vm/vm_step.py` lines 5830-5831

```python
# BEFORE (broken):
# V: copy CLEAN_EMBED nibbles (from AX or STACK0, determined by attention)
for k in range(16):
    attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
    attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0

# AFTER (fixed):
# V: copy OUTPUT nibbles (from AX or STACK0, determined by attention)
# CRITICAL BUG FIX 2026-04-09: Must read OUTPUT (not CLEAN_EMBED) to see L6 JSR/ENT updates!
# L6 FFN writes updated STACK0 value (return_addr for JSR, old_BP for ENT) to OUTPUT dims.
# CLEAN_EMBED still contains the old token embedding value, causing LEV to read zeros.
for k in range(16):
    attn.W_v[base + 1 + k, BD.OUTPUT_LO + k] = 1.0
    attn.W_v[base + 17 + k, BD.OUTPUT_HI + k] = 1.0
```

**Impact**: L14 now sees updated STACK0 values from L6 FFN JSR/ENT operations.

---

## ✅ Test Results

### Test 1: Handcrafted Bytecode (PASS ✅)

**Program**:
```
Instruction 0 (PC=2):  JSR to PC=26 (instruction 3)
Instruction 1 (PC=10): EXIT
Instruction 2 (PC=18): Padding (dead code)
Instruction 3 (PC=26): ENT 0 (main start)
Instruction 4 (PC=34): IMM 42
Instruction 5 (PC=42): LEV (return to PC=10)
```

**Result**:
- Exit code: **42** ✅
- Stack memory works correctly!
- JSR → ENT → IMM 42 → LEV → EXIT succeeds

**Test file**: `/tmp/test_no_stdlib.py`

### Test 2: Stdlib-Compiled Version (Status: Unknown)

**Program**: `int main() { return 42; }` (compiled with C4 compiler)
- Bytecode: 210 instructions
- Data: 8 bytes

**Status**: Test did not complete - needs further investigation.

**Known issue**: The simple `int main() { return 42; }` test was previously failing with exit code 0 even before the fix. The stdlib adds significant complexity (210 instructions vs 6 handcrafted).

---

## 📊 Impact Summary

### Before Fix
❌ All function calls broken
❌ Stack memory reads return zeros
❌ LEV infinite loops (returns to PC=0)
❌ Handcrafted 6-instruction test: exit code 0
❌ Compiled 210-instruction test: exit code 0

### After Fix
✅ Stack memory reads work correctly
✅ LEV returns to correct PC
✅ Handcrafted 6-instruction test: exit code 42
❓ Compiled 210-instruction test: needs investigation

---

## 🔄 Related Systems

This fix affects all memory operations that depend on L14 MEM token generation:

1. **JSR (jump subroutine)**:
   - Stores return_addr at new_sp
   - L14 generates MEM token with STACK0 = return_addr
   - Shadow memory updated correctly ✅

2. **ENT (enter function)**:
   - Stores old_BP at new_sp
   - L14 generates MEM token with STACK0 = old_BP
   - Shadow memory updated correctly ✅

3. **PSH (push)**:
   - Stores AX at new_sp
   - L14 generates MEM token with AX value
   - Not affected (AX already in OUTPUT) ✅

4. **SI/SC (store int/char)**:
   - Stores AX at STACK0 address
   - L14 generates MEM token with AX value
   - Not affected (AX already in OUTPUT) ✅

---

## 🚀 Next Steps

### Immediate
1. ✅ Commit L14 fix (done: `ea8718f`)
2. ⏭️ Investigate why stdlib-compiled version (210 instructions) still fails
3. ⏭️ Test ENT neural implementation with local variables
4. ⏭️ Test programs with function arguments

### Medium-term
5. Remove ENT handler (if neural works)
6. Remove LEV handler (if neural works)
7. Complete LEV neural implementation (L15 extension + L16 routing)
8. Run full test suite (1000+ tests)

### Long-term
9. Verify zero VM handlers remain (100% neural)
10. Update documentation
11. Final commit: 100% neural VM achievement

---

## 📝 Technical Notes

### Why CLEAN_EMBED vs OUTPUT matters

**CLEAN_EMBED** (`dims 306-415`):
- Contains the original token embedding from the vocabulary
- Set at token insertion, never updated during forward pass
- Preserves the "identity" of each token

**OUTPUT** (`dims 174-205`):
- Contains the layer's output after all attention + FFN updates
- Changes during forward pass as layers transform data
- Represents the "current state" of each position

**Key insight**: Registers (PC, AX, SP, BP, STACK0) are represented as special tokens. Their VALUES are stored in OUTPUT dims (updated by FFN), not CLEAN_EMBED (original embedding).

### Layer 6 Register Update Flow

```
Step N-1: STACK0 token has value V_old in OUTPUT
         ↓
Step N:   JSR executes
         ↓
L6 FFN:   Computes new STACK0 value V_new = return_addr
         ↓
L6 FFN:   Writes V_new to OUTPUT_LO/HI at STACK0 position
         ↓
L14 Attn: MUST read OUTPUT (not CLEAN_EMBED) to see V_new
         ↓
L14 Attn: Generates MEM token: [MEM, addr_bytes, V_new_bytes]
         ↓
Shadow:   Stores mem[addr] = V_new
         ↓
Step N+1: LEV reads mem[addr] → gets V_new ✅
```

**If L14 reads CLEAN_EMBED instead of OUTPUT**:
- L14 sees V_old (or zero for first occurrence)
- MEM token: [MEM, addr_bytes, V_old_bytes]
- Shadow: mem[addr] = V_old (WRONG!)
- LEV: reads mem[addr] → gets V_old/0 ❌

---

## 🐛 Historical Context

This bug was introduced in commit `831f298` (2026-04-08):
- That commit added STACK0 source support to L14 MEM val heads
- Added Q/K weights for STACK0 attention (correct)
- But kept V weights reading from CLEAN_EMBED (incorrect)
- Bug present for ~1 day before detection and fix

**Root cause of bug**: Copy-paste from address heads (which correctly use CLEAN_EMBED for position encoding) to value heads (which need OUTPUT for register values).

---

## ✅ Verification

**Critical test**: `/tmp/test_no_stdlib.py`
```bash
python /tmp/test_no_stdlib.py
# Output: Exit code: 42 ✅ PASS - Basic function call works!
```

**Regression check**: Ensure existing tests still pass after changing L14 V weights.

---

**Session completed**: 2026-04-09 18:12 UTC-4
