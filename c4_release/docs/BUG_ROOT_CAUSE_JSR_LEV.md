# Root Cause Analysis: JSR/LEV Infinite Loop

## Summary

The infinite JSR/LEV loop is caused by **handler overrides not persisting across steps**. Runtime handlers override register values in the current step's context, but the neural network generates the next step based on its weights, ignoring these overrides. This causes register values (especially SP and BP) to diverge between handler expectations and neural predictions.

## Detailed Analysis

### Expected Flow

For `int main() { return 42; }`:

```
Bytecode:
  [0] PC=2:  JSR 18  (call main at PC=18)
  [1] PC=10: EXIT 0
  [2] PC=18: ENT 0   (enter main)
  [3] PC=26: IMM 42
  [4] PC=34: LEV
  [5] PC=42: LEV
```

**Expected execution:**
1. **Step 0 (JSR)**:
   - exec_pc=2, jump to PC=18, return_addr=10
   - Store return_addr=10 at mem[65528]
   - Override: SP=65528, PC=18

2. **Step 1 (ENT)**:
   - exec_pc=18
   - Should read old_sp=65528 (from JSR)
   - push_addr=65520, store old_bp=65536 at mem[65520]
   - Override: SP=65520, BP=65520, PC=26

3. **Steps 2-3 (IMM, LEV)**:
   - Execute IMM 42, then LEV

4. **LEV execution**:
   - old_bp=65520
   - saved_bp=mem[65520]=65536
   - return_addr=mem[65528]=10
   - Override: PC=10, SP=65536, BP=65536

5. **Step 4 (EXIT)**:
   - exec_pc=10 → bytecode[1] = EXIT
   - Exit with AX=42

### Actual Flow (Bug)

**What actually happens:**

1. **Step 0 (JSR)**: ✓ Works correctly
   - Stores return_addr=10 at mem[65528]
   - Overrides SP=65528 in step 0 context

2. **Step 1 (ENT)**: ❌ Gets wrong SP!
   - Neural network generates step 1
   - **Neural SP output = 65784** (based on weights, not override!)
   - ENT handler reads old_sp=65784 (from neural output)
   - push_addr=65776 (should be 65520!)
   - Stores old_bp=65536 at mem[65776] (should be mem[65520]!)

3. **LEV execution**: ❌ Reads from wrong address!
   - old_bp=65776 (from ENT)
   - saved_bp=mem[65776]=65536 ✓
   - return_addr=mem[65776+8]=mem[65784]=**0** ✗ (should read mem[65528]=10!)
   - Overrides PC=0 instead of PC=10

4. **Step N (PC=0)**: ❌ Infinite loop!
   - PC=0 → bytecode[0] = JSR 18
   - Loop repeats forever

### Debug Evidence

```
[JSR] Storing return_addr=10 at shadow memory[65528]
[JSR] Shadow memory now: [65528]=10

[ENT] Handler called
[ENT] old_sp=65784, old_bp=65536        ← WRONG! Should be 65528
[ENT] push_addr=65776, new_bp=65776     ← WRONG! Should be 65520
[ENT] Storing old_bp=65536 at mem[65776]

[LEV] Handler called
[LEV] old_bp=65776
[LEV] saved_bp=65536, return_addr=0     ← WRONG! return_addr should be 10
[LEV] Overriding PC to 0                ← WRONG! Should be PC=10
```

## Root Cause

**Handler overrides don't persist to the next neural generation step.**

The architecture has two execution paths:

1. **Neural execution**: Model generates next step based on learned weights
2. **Handler overrides**: Python code modifies context values

The problem:
- **Step N**: Handler overrides SP=65528 in step N context
- **Step N→N+1**: Model generates step N+1 using step N context as input
- **But**: Model's weights predict SP based on previous patterns, producing SP=65784
- **Step N+1**: Handler reads SP from neural output (65784), not the override (65528)

This creates a **conflict between neural predictions and handler overrides**.

## Why max_steps=10 Succeeded but max_steps=100 Failed

- **max_steps=10**: Program exits quickly via neural EXIT path before LEV is reached
  - JSR executes, but program hits EXIT at bytecode[1] before completing main()
  - Exit code = 0 (from EXIT instruction, not return value)

- **max_steps=100**: Program reaches LEV, triggering the bug
  - LEV tries to return but gets wrong return address (0 instead of 10)
  - Returns to PC=0, triggering JSR again
  - Infinite loop

## Impact

This bug affects **ALL programs with function calls**:
- Any program using JSR/ENT/LEV
- Includes trivial programs like `int main() { return 0; }`
- Makes multi-step testing impossible
- Breaks regression test suite

## Fix Approaches

### Option 1: Remove Handlers (Use Pure Neural)

Remove all runtime handlers and rely 100% on neural execution.

**Pros:**
- No conflict between handlers and neural
- Simpler architecture
- Forces neural weights to be correct

**Cons:**
- Requires neural weights to handle all operations correctly
- May need retraining or weight fixes
- Loses Python-side correctness guarantees

### Option 2: Fix Override Persistence

Make handler overrides persist by updating `_last_*` variables AND injecting override values into next step's input.

**Approach:**
```python
# After JSR handler overrides SP=65528
self._last_sp = 65528  # Update tracker
# AND: Inject into embedding for next step
# So neural network sees the override as input
```

**Pros:**
- Preserves both neural and handler execution
- Handlers can fix neural bugs incrementally

**Cons:**
- Complex interaction between neural and handlers
- May still have conflicts
- Requires careful coordination

### Option 3: Detect and Avoid Handler Conflicts

Only use handlers for operations where neural weights are known broken, and ensure they don't conflict.

**Approach:**
- Remove JSR/ENT/LEV handlers (let neural handle them)
- Keep only syscall handlers (GETCHAR, PUTCHAR, etc.)
- Or: Fix neural weights for JSR/ENT/LEV

**Pros:**
- Cleaner separation of concerns
- Reduces conflict surface

**Cons:**
- Requires identifying which operations need handlers
- May need neural weight fixes

## Recommended Fix

**Immediate**: Remove JSR/ENT/LEV handlers and test if pure neural execution works.

If neural execution has bugs, fix them in weights rather than handlers.

**Reasoning**: The conflict between neural and handlers is fundamental. Handlers were likely added as quick fixes for neural bugs, but they create worse problems. Better to fix the neural weights properly.

## Test to Verify Fix

```python
# Disable ALL function-call handlers
self._func_call_handlers = {
    # JSR, ENT, LEV removed - let neural handle them
}

# Test simple program
code = "int main() { return 42; }"
bytecode, data = compile_c(code)
output, exit_code = runner.run(bytecode, max_steps=100)

# Should get exit_code=42, not 0 or infinite loop
assert exit_code == 42, f"Expected 42, got {exit_code}"
```

If this fails, the neural weights need fixing. If it succeeds, handlers were the problem.

## Files Affected

- `neural_vm/run_vm.py`: Contains handler code
- Lines 1161-1253: JSR, ENT, LEV handlers
- Lines 202-222: Handler registration

## Related Issues

- Pre-existing at commit c344d6f (before first-step opcode work)
- Not related to first-step opcode decode changes
- Blocks all multi-step testing
- Makes commit 2e942bc's runtime handler additions even worse (adds more conflicts)

---

**Status**: Root cause identified, fix approach recommended
**Next Step**: Test with handlers disabled to see if neural execution works
