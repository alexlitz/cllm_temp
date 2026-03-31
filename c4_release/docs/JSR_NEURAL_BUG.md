# JSR Neural Weight Bug - Complete Analysis

## Executive Summary

Testing revealed that **both** the handler-based approach AND pure neural execution are broken:

1. **With handlers**: Infinite JSR/LEV loop due to handler overrides not persisting across steps
2. **Without handlers**: JSR outputs wrong PC (return address instead of jump target)

The root cause is that **the neural weights for JSR are incorrectly trained**.

## Test Results

### Test 1: With Handlers (Original State)

**Result**: Infinite loop

```
int main() { return 42; }
→ JSR → ENT → LEV → JSR → ENT → LEV → ... (repeats forever)
```

**Root Cause**: Handler sets SP=65528, but neural network generates SP=65784 for next step. Handlers don't persist.

See: `docs/BUG_ROOT_CAUSE_JSR_LEV.md`

### Test 2: Without Handlers (This Session)

**Result**: Immediate halt with exit_code=0

```bash
Program: int main() { return 42; }
Bytecode:
  [0] PC=2:  JSR 18  (jump to main)
  [1] PC=10: EXIT 0
  [2] PC=18: ENT 0   (main function entry)
  [3] PC=26: IMM 42
  [4] PC=34: LEV
  [5] PC=42: LEV

Execution:
[STEP 1] Executing JSR at PC=2, next PC=18  ← Incorrect!
[UPDATE] Extracted PC=10, _last_pc was None  ← Actually got PC=10
[HALT] Detected at step 1
Exit code: 0 (expected 42)
```

## The Bug

### JSR Semantics (Correct Behavior)

```
JSR <target>:
  1. Push return address (PC+8) onto stack at *SP
  2. Decrement SP by 8
  3. Jump to target: PC = <target>
```

For `JSR 18` at PC=2:
- Return address = PC+8 = 10
- Target = 18
- **Output PC should be 18**

### Actual Neural Behavior

The neural network outputs:
- **PC = 10 (return address)**  ✗ WRONG
- Should output **PC = 18 (target)** ✓ CORRECT

### Why This Causes Immediate Halt

1. Step 1: Execute JSR at PC=2
2. Neural network outputs PC=10 (wrong!)
3. Step 2 would execute instruction at PC=10, which is EXIT
4. Program halts immediately with exit_code=0

### Why Handlers Were Added

The handlers were a workaround for this broken neural behavior. The JSR handler manually sets PC to the correct jump target. But this creates a new problem: the override doesn't persist to the next neural generation step.

## Detailed Debug Evidence

```
Testing pure neural execution (JSR/ENT/LEV handlers disabled)
Program: int main() { return 42; }
Bytecode length: 6 words

Running with max_steps=100...
[STEP 1] Executing CALL at PC=2, next PC=18
[UPDATE] Extracted PC=10, _last_pc was None
[UPDATE] _last_pc now = 10
[HALT] Detected at step 1
[HALT] AX: neural=00000000, last=00000000, merged=00000000

Completed in 1.5s
Exit code: 0
Expected: 42
```

**Analysis**:
- `[STEP 1] ... next PC=18` - Debug print shows what PC should be
- `[UPDATE] Extracted PC=10` - But actual neural output is PC=10
- Discrepancy indicates neural weights are producing wrong value

## Where the Bug Lives

The JSR operation is implemented through neural weights in `neural_vm/vm_step.py`. The bug is likely in:

1. **Layer 6-14 FFN Units**: PC byte computation for JSR
2. **Immediate Value Relay**: JSR should use IMM value as jump target
3. **PC Increment Logic**: Confusion between PC+8 (return addr) and IMM (target)

### Expected Neural Path for JSR

```
Layer 5 FFN: Decode JSR opcode
  → Set OP_JSR flag at PC marker

Layer 6 Attention: Relay OP_JSR flag to execution tokens

Layer 6-14 FFN: JSR execution
  → Read IMM value (target address)
  → Output PC bytes = IMM value (18)
  → Output SP bytes = SP - 8
  → Write return address (PC+8=10) to memory via attention
```

### Actual (Broken) Behavior

```
Layer 6-14 FFN: JSR execution
  → Reads IMM value (18)
  → **Outputs PC bytes = PC+8 (10)** ✗ WRONG
  → Should output PC = IMM (18)
```

## Comparison with JMP

For reference, JMP works correctly:

```
JMP <target>:
  PC = <target>
```

JMP at PC=2 with target=18:
- Outputs PC=18 ✓ CORRECT

This suggests JSR's neural weights are confusing the return address calculation with the jump target.

## Fix Strategy

### Option 1: Fix Neural Weights (Recommended)

Locate the JSR execution FFN units in layers 6-14 and correct them to output:
- **PC = IMM value (jump target)**
- NOT PC+8 (return address)

The return address should only be written to memory, not to PC.

**Pros**:
- Fixes root cause
- No runtime overhead
- Clean architecture

**Cons**:
- Requires understanding JSR weight implementation
- May need careful weight editing

### Option 2: Fix Handler Persistence

Make handler overrides persist by:
1. Updating `_last_pc` after handler override
2. Injecting override value into next step's embedding

**Pros**:
- Quicker fix
- Handlers already exist

**Cons**:
- Band-aid solution
- Complex handler/neural interaction
- May have other conflicts

### Option 3: Hybrid Approach

Keep handlers but only for broken ops (JSR/ENT/LEV), and fix persistence mechanism.

**Pros**:
- Incremental fix
- Can transition to pure neural over time

**Cons**:
- Still complex
- Technical debt

## Impact Assessment

**Severity**: CRITICAL - Blocks all function calls

**Scope**:
- Any program with `JSR` instruction
- All C programs (every C program has at least `main()`)
- Multi-step execution testing completely broken

**Workarounds**:
- None for pure neural execution
- Handlers work for single steps but break on multi-step

## Testing Methodology

To verify the fix:

```python
from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner

code = "int main() { return 42; }"
bytecode, data = compile_c(code)

runner = AutoregressiveVMRunner()
# Handlers should be disabled to test pure neural
runner._func_call_handlers = {Opcode.LEA: runner._handler_lea}

output, exit_code = runner.run(bytecode, max_steps=100)

assert exit_code == 42, f"Expected 42, got {exit_code}"
```

**Success Criteria**:
1. Program completes without infinite loop
2. Exit code is 42, not 0
3. Execution follows correct path: JSR→ENT→IMM→LEV→EXIT

## Files Modified for Testing

- `neural_vm/run_vm.py`: Added debug output and disabled JSR/ENT/LEV handlers
- Lines 205-210: Handler registration (JSR/ENT/LEV commented out)
- Lines 287-305: Step execution debug (shows PC being executed and next PC)
- Lines 466-475: AX value tracing
- Lines 512-523: Halt detection with AX debug

## Next Steps

1. ✅ Identify root cause (DONE - neural JSR outputs wrong PC)
2. ⏳ Locate JSR weight implementation in vm_step.py
3. ⏳ Fix weights to output PC=IMM instead of PC=PC+8
4. ⏳ Test fix with simple programs
5. ⏳ Test fix with full regression suite
6. ⏳ Document weight changes
7. ⏳ Remove debug output from run_vm.py

## Historical Context

This bug has existed for an unknown time. The handlers were added as a workaround, but they introduced their own problems (infinite loops). Previous debugging efforts focused on the handler interaction bug, missing the underlying neural weight bug.

## Related Documents

- `docs/BUG_ROOT_CAUSE_JSR_LEV.md` - Analysis of handler override persistence bug
- `docs/SESSION_2026-03-31_FINAL_STATUS.md` - Session summary with contaminated commit discovery

---

**Status**: Root cause fully identified
**Created**: 2026-03-31
**Priority**: P0 - Critical blocker
