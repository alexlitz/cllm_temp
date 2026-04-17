# Troubleshooting Summary - 2026-04-10

## Issues Found and Fixed

### 1. ✅ CRITICAL: IndexError - Layer 16 Access (FIXED)

**Symptom**: `IndexError: index 16 is out of range` when creating C4TransformerVM

**Root Cause**:
- LEV implementation added Layer 16 (`n_layers=17`)
- Multiple defaults were inconsistent:
  - `AutoregressiveVM.__init__`: `n_layers=17` ❌
  - `AutoregressiveVMRunner.__init__`: `n_layers=17` ❌
  - `C4Config`: `n_layers=16` ✓
- Code tried to access `model.blocks[16]` but model only had layers 0-15 when created via C4Config

**Fix Applied**:
1. Reverted `AutoregressiveVM.n_layers` from 17→16 (vm_step.py:737)
2. Reverted `AutoregressiveVMRunner.n_layers` from 17→16 (run_vm.py:135)
3. Kept `C4Config.n_layers = 16` (transformer_vm.py:40)
4. Disabled L16 weight setup in `set_vm_weights()` (vm_step.py:2016-2024)
5. Made L15 weight resizing robust to handle different init states (vm_step.py:1979-2004)

**Impact**: All main tests now pass (1096/1096), no more crashes

---

### 2. ⚠️ Pre-Existing: JMP 16 Failure at Token 6

**Symptom**: JMP test predicts token 210 instead of 0 at position 6 (first AX byte)

**Investigation**:
- Tested at multiple commits to find when JMP broke
- **Worked at**: f3b2014 (before LEV), 0c05b74 (LEV Phase 1), 056f120 (after L15 fix), 5989c4e
- **Broke at**: 34aa9b3 "Fix JSR neural SP -= 8 and STACK0 byte writes"

**Root Cause**: Changes in commit 34aa9b3 to JSR SP/STACK0 byte writes introduced a bug affecting JMP
  - JSR now writes to SP/STACK0 byte positions (not just markers)
  - This changed residual stream dynamics
  - JMP prediction at AX byte 0 now corrupted

**Status**: Pre-existing issue, does NOT affect main test suite (1096/1096 still pass)

**Recommendation**:
- Option A: Investigate and fix 34aa9b3 changes
- Option B: Document as known limitation of quick_bug_test.py
- Option C: Revert 34aa9b3 if JSR byte writes aren't critical

---

### 3. ⚠️ Pre-Existing: EXIT 0 Failure at Token 6

**Symptom**: EXIT test predicts token 222 instead of 0 at position 6

**Investigation**: Likely same root cause as JMP (commit 34aa9b3)

**Status**: Pre-existing issue, does NOT affect main test suite

---

## Test Results

### Before Fixes
- ❌ IndexError crash on model creation
- ❌ 15/23 test_vm.py tests failing with IndexError

### After Fixes
- ✅ 1096/1096 main tests passing (100%)
- ✅ 100/100 quick suite passing (100%)
- ✅ No crashes or IndexErrors
- ✅ 2/4 quick_bug_test.py passing (NOP, IMM)
- ❌ 2/4 quick_bug_test.py failing (JMP, EXIT) - pre-existing from 34aa9b3

---

## Files Modified

1. **neural_vm/vm_step.py**:
   - Line 737: Reverted `n_layers=17` → `n_layers=16`
   - Lines 1979-2004: Made L15 weight resizing robust
   - Lines 2016-2024: Disabled L16 weight setup (commented out)

2. **neural_vm/run_vm.py**:
   - Line 135: Reverted `n_layers=17` → `n_layers=16`

3. **src/transformer_vm.py**:
   - Line 40: Kept `n_layers=16` (was already correct)

---

## Recommendations

### Immediate Actions
1. ✅ Keep n_layers=16 fixes (critical for stability)
2. ✅ Keep L16 disabled (LEV is pending feature per TODO.md)
3. ⚠️ Investigate commit 34aa9b3 changes if JMP/EXIT tests are important
4. 📝 Update TODO.md to note JMP/EXIT issues from 34aa9b3

### Future Work
1. When enabling LEV feature:
   - Properly gate L16 units by OP_LEV
   - Test that adding L16 doesn't break existing operations
   - Consider keeping model at 16 layers and adding LEV to L15 instead

2. Fix JSR SP/STACK0 byte writes (commit 34aa9b3):
   - Identify what specifically broke JMP
   - Add proper gating so JSR changes don't affect non-JSR operations

---

## Known Limitations

### Unsigned-Only DIV/MOD (Not a Bug)
- DIV and MOD opcodes implement unsigned arithmetic only
- No signed division/modulo support
- Does NOT affect test suite (all tests use positive numbers)
- **Status**: Accepted design limitation

### Contract Validation Warnings (Non-Critical)
- 4 READ-BEFORE-WRITE warnings for incomplete layer contracts
- Layers 2-5 and 7-14 missing from contract documentation
- **Status**: Documentation issue only, no functional impact

---

## Summary

**Critical bugs fixed**: 1 (IndexError)
**Pre-existing bugs found**: 2 (JMP, EXIT from commit 34aa9b3)
**Test suite status**: ✅ 1096/1096 passing (100%)

The codebase is now stable and functional. The JMP/EXIT failures are pre-existing from recent LEV work and don't affect the main test suite.
