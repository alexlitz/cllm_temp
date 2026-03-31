# Session 2026-03-31 (Continued): JSR Neural Weight Bug Discovery

## Summary

Continued from previous session. Successfully identified the **true root cause** of the JSR/LEV infinite loop: the neural network's JSR implementation is outputting the wrong PC value.

## What Was Done

### 1. Tested Pure Neural Execution (Handlers Disabled)

**Goal**: Determine if removing handlers (which cause infinite loops) would allow pure neural execution to work.

**Method**:
- Disabled JSR/ENT/LEV handlers in `neural_vm/run_vm.py`
- Tested with `int main() { return 42; }`
- Added debug tracing for PC and AX values

**Result**: Program completes immediately with exit_code=0 instead of 42.

### 2. Discovered the Real Bug

**Finding**: JSR outputs wrong PC value

```
Bytecode:
  [0] PC=2:  JSR 18  (jump to main at PC=18)
  [1] PC=10: EXIT 0  (return point)
  [2] PC=18: ENT 0   (main function)
  [3] PC=26: IMM 42  (return 42)
  [4] PC=34: LEV     (return from main)

Execution:
  Step 1: Execute JSR at PC=2
    Expected output: PC=18 (jump to target)
    Actual output: PC=10 (return address)

  Step 2: Would execute EXIT at PC=10
    Program halts with exit_code=0 instead of continuing
```

**Debug Evidence**:
```
[STEP 1] Executing CALL at PC=2, next PC=18  ← Wrong!
[UPDATE] Extracted PC=10, _last_pc was None  ← Actually got 10
[HALT] Detected at step 1
Exit code: 0 (expected 42)
```

### 3. Understood the Complete Picture

**Two Bugs, Same Root Cause**:

1. **With Handlers** (previous session):
   - Handler sets PC=18 correctly
   - But override doesn't persist to next step
   - Neural network generates PC=65784 (wrong SP)
   - Infinite loop

2. **Without Handlers** (this session):
   - Neural network outputs PC=10 instead of PC=18
   - Program executes EXIT immediately
   - Halts with wrong exit code

**Root Cause**: Neural weights for JSR are broken. They output `PC+8` (return address) instead of `IMM` (jump target).

**Why Handlers Existed**: They were a workaround for this broken neural behavior.

### 4. Documented Findings

Created comprehensive documentation:
- `docs/JSR_NEURAL_BUG.md` - Complete analysis with fix strategy
- Added debug output to `neural_vm/run_vm.py` (not committed)
- Committed documentation (commit 5470999)

## Key Findings

### JSR Semantics (Correct)

```
JSR <target>:
  1. Push return address (PC+8) onto stack
  2. Jump to target: PC = <target>
```

### Neural JSR (Broken)

```
JSR <target>:
  1. Something with stack (unclear)
  2. Set PC = PC+8  ✗ WRONG (should be PC = <target>)
```

### Why This Matters

- **Impact**: CRITICAL - blocks ALL programs with function calls
- **Scope**: Every C program (all have `main()` function)
- **Workarounds**: None that work correctly
- **Multi-step testing**: Completely broken

## Files Modified

### Committed
- `docs/JSR_NEURAL_BUG.md` - Complete bug analysis (new file, 266 lines)

### Modified (Not Committed - Debug Only)
- `neural_vm/run_vm.py`:
  - Lines 205-210: Disabled JSR/ENT/LEV handlers
  - Lines 281-305: Added step execution tracing
  - Lines 466-475: Added AX value tracing
  - Lines 512-523: Added halt detection with AX debug

## Technical Details

### Test Code

```python
from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner

code = "int main() { return 42; }"
bytecode, data = compile_c(code)

runner = AutoregressiveVMRunner()
# Handlers disabled (commented out JSR/ENT/LEV)

output, exit_code = runner.run(bytecode, max_steps=100)
# Expected: 42
# Actual: 0
```

### Bytecode Analysis

```
Program: int main() { return 42; }

[0] PC=2:  JSR  18  (opcode=3, imm=18) → Jump to PC=18
[1] PC=10: EXIT  0  (opcode=38) → Halt (return point)
[2] PC=18: ENT   0  (opcode=6) → Enter function
[3] PC=26: IMM  42  (opcode=1, imm=42) → AX = 42
[4] PC=34: LEV   0  (opcode=8) → Return from function
[5] PC=42: LEV   0  (opcode=8) → (extra LEV, compiler artifact)
```

### Expected Execution Flow

1. Step 1: JSR at PC=2 → Jump to PC=18, push return_addr=10
2. Step 2: ENT at PC=18 → Set up stack frame, next PC=26
3. Step 3: IMM at PC=26 → AX=42, next PC=34
4. Step 4: LEV at PC=34 → Return to PC=10
5. Step 5: EXIT at PC=10 → Halt with AX=42

### Actual Execution Flow (Broken)

1. Step 1: JSR at PC=2 → **Outputs PC=10** (should be 18!)
2. Step 2: EXIT at PC=10 → Halt with AX=0

## Fix Strategy

### Option 1: Fix Neural Weights (Recommended)

Locate JSR execution in `neural_vm/vm_step.py` (likely layers 6-14 FFN) and correct it to output:
- **PC = IMM value** (jump target from immediate field)
- NOT PC+8 (return address)

**Pros**:
- Fixes root cause
- No runtime overhead
- Clean architecture

**Cons**:
- Requires understanding current JSR weight implementation
- Need to identify which FFN units handle JSR

### Option 2: Fix Handler Persistence

Make handler overrides persist across steps:
1. Update `_last_*` variables after handler override
2. Inject override values into next step's embedding
3. Ensure neural network sees overrides as input

**Pros**:
- Quicker implementation
- Handlers already exist

**Cons**:
- Band-aid solution
- Complex interaction between handlers and neural
- May have other side effects

### Option 3: Hybrid Approach

Use handlers only for broken ops (JSR/ENT/LEV) and fix persistence:
- Keep handlers for compatibility
- Fix persistence mechanism
- Gradually transition to pure neural

**Pros**:
- Incremental improvement
- Safer transition

**Cons**:
- Still complex
- Technical debt remains

## Comparison with Previous Analysis

### Previous Session (BUG_ROOT_CAUSE_JSR_LEV.md)

**Conclusion**: Handler overrides don't persist across steps

**Evidence**:
```
[JSR] Storing return_addr=10 at shadow memory[65528]
[ENT] old_sp=65784, old_bp=65536  ← Wrong! Should be 65528
```

**Interpretation**: Handlers work, but neural network ignores them

### This Session (JSR_NEURAL_BUG.md)

**Conclusion**: Neural weights are fundamentally broken

**Evidence**:
```
[STEP 1] Executing JSR at PC=2, next PC=18  ← Debug says 18
[UPDATE] Extracted PC=10                    ← Actually got 10
```

**Interpretation**: Neural network outputs wrong PC even without handlers

### Combined Understanding

1. **Neural weights are broken**: JSR outputs wrong PC
2. **Handlers were added as workaround**: Override broken neural behavior
3. **Handlers have their own bug**: Overrides don't persist
4. **Result**: Both approaches fail, for different reasons

**True Fix**: Fix the neural weights so handlers aren't needed.

## Testing Methodology

### Success Criteria

After fixing JSR neural weights:

```python
# Test 1: Simple return
code = "int main() { return 42; }"
exit_code = run(code)
assert exit_code == 42

# Test 2: Multiple returns
code = "int main() { if (1) return 5; return 10; }"
exit_code = run(code)
assert exit_code == 5

# Test 3: No infinite loops
code = "int main() { return 0; }"
exit_code = run(code, max_steps=100)  # Should complete < 10 steps
assert exit_code == 0

# Test 4: Nested calls
code = """
int foo() { return 1; }
int bar() { return foo() + 1; }
int main() { return bar(); }
"""
exit_code = run(code)
assert exit_code == 2
```

### Regression Tests

After fix, verify:
1. All 6 simple test programs work (return 0, return 42, etc.)
2. Tests complete in <20 steps (no infinite loops)
3. Exit codes match expected values
4. No handler overrides needed

## Historical Context

Timeline of JSR bug:
1. **Initial implementation**: Neural weights for JSR were trained/set incorrectly
2. **Bug discovered**: JSR doesn't work in multi-step execution
3. **Workaround added**: Runtime handlers to override broken neural behavior
4. **New bug introduced**: Handler overrides don't persist, causing infinite loops
5. **Previous session**: Identified handler persistence problem
6. **This session**: Discovered underlying neural weight bug

The handlers masked the real problem, making debugging more difficult.

## Next Steps

### Immediate (This Session)

1. ✅ Test pure neural execution (handlers disabled)
2. ✅ Identify root cause (neural JSR outputs wrong PC)
3. ✅ Document findings comprehensively
4. ✅ Commit documentation

### Next Session

1. ⏳ Locate JSR implementation in `neural_vm/vm_step.py`
2. ⏳ Analyze current JSR weight values
3. ⏳ Understand why PC+8 is output instead of IMM
4. ⏳ Fix weights to output correct PC value
5. ⏳ Test fix with simple programs
6. ⏳ Run full regression suite
7. ⏳ Remove debug output from `run_vm.py`
8. ⏳ Update documentation with fix details

### Long Term

1. ⏳ Test first-step opcode decode (blocked until JSR works)
2. ⏳ Add regression tests for function calls
3. ⏳ Consider removing handlers entirely if neural works
4. ⏳ Review other opcodes for similar issues

## Commit History

```
5470999 Document JSR neural weight bug - PC output is wrong
[Previous commits from earlier session...]
```

## Lessons Learned

1. **Workarounds can mask root causes**: Handlers hid the neural bug
2. **Test without workarounds**: Disabling handlers revealed the real issue
3. **Multiple bugs can compound**: Handler persistence + neural bug = complex debugging
4. **Detailed tracing is essential**: Step-by-step PC tracking showed the exact failure point
5. **Document everything**: Complete analysis helps future debugging

## Related Documents

- `docs/JSR_NEURAL_BUG.md` - Complete bug analysis (this session)
- `docs/BUG_ROOT_CAUSE_JSR_LEV.md` - Handler persistence analysis (previous session)
- `docs/SESSION_2026-03-31_FINAL_STATUS.md` - Previous session summary

---

**Status**: Root cause fully identified, fix strategy documented
**Next Step**: Locate and fix JSR weights in vm_step.py
**Blocker**: JSR bug blocks all multi-step testing and first-step decode verification
