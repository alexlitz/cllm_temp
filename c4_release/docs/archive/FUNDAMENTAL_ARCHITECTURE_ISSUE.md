# FUNDAMENTAL ARCHITECTURE ISSUE

## Critical Discovery

**The tests pass even though the transformer predictions are wrong because DraftVM (Python interpreter) produces the final results, not the transformer.**

## How It Actually Works

### Execution Flow

```
Step 1:
  DraftVM.step() executes in Python ← Produces correct result
  ↓
  DraftVM.draft_tokens() generates 35 tokens
  ↓
  Transformer validates tokens ← May accept only 6/35 tokens
  ↓
  Context updated with accepted tokens only

Step 2:
  DraftVM.step() executes again ← Still uses Python execution
  ↓
  (DraftVM doesn't care about transformer validation)
  ↓
  ...continues...

Final Result:
  return vm.ax ← DraftVM's register, NOT transformer's prediction
```

### The Problem

Looking at `batch_runner_v2.py:122`:
```python
# Return exit codes
return [vm.ax for vm in self.draft_vms]
```

**The final result comes from DraftVM (Python), not the transformer!**

## Concrete Example: IMM 42

```
Program: IMM 42; EXIT

DraftVM execution:
  - Sets AX = 42 (correct, in Python)
  - Produces tokens including token[6] = 42

Transformer validation:
  - Predicts token[6] = 0 (WRONG!)
  - Accepts only 6/35 tokens (rejects AX byte 0)

Final result:
  - Exit code = vm.ax = 42 (from DraftVM)
  - Test passes ✓ (because it checks DraftVM.ax, not transformer)
```

## Why This Matters

### What We Thought:
- Transformer generates tokens autoregressively
- Tests validate transformer predictions
- PC_OFFSET=0 fix improves neural execution

### What Actually Happens:
- DraftVM (Python) generates tokens
- Transformer passively validates (but results are ignored)
- Tests check DraftVM results, not transformer
- **Transformer predictions can be completely wrong and tests still pass**

## Evidence

### Test 1: Token Validation
```bash
$ python test_validation_actually_works.py

Tokens accepted: 6/35  ← Transformer rejects most tokens
Exit code: 42          ← Test passes anyway (uses DraftVM.ax)
```

### Test 2: Raw Neural Predictions
```bash
$ python check_remaining_bugs.py

IMM: AX byte 0 = 0, expected 42  ← Transformer wrong
JMP: PC byte 0 = 8, expected 16  ← Transformer wrong
EXIT: END token = 262, expected 263  ← Transformer wrong

But all 59/59 tests pass! ← Because they use DraftVM results
```

## The "Speculative Execution" Design

The system is designed as:

**DraftVM = Speculative Executor (always correct, Python)**
**Transformer = Validator (tries to predict, but failures don't matter)**

This is a valid design for:
- Fast execution (DraftVM is much faster than autoregressive transformer)
- Correctness (DraftVM is a proven Python interpreter)
- Batching (transformer validates 256 programs in one pass)

But it means:
- ❌ NOT a neural VM (execution is Python, not neural)
- ❌ NOT autoregressive generation (transformer doesn't generate, just validates)
- ❌ Transformer bugs don't affect correctness (so they don't get caught)

## Implications

### For the PC_OFFSET=0 Fix
- ✓ Fixed transformer token predictions (good for neural accuracy)
- ✓ All tests still pass (because they always passed via DraftVM)
- ❌ Didn't actually improve "neural VM" because execution was never neural

### For "Pure Neural Execution"
Would require completely different architecture:
```python
# Instead of:
vm.step()  # Python execution
return vm.ax  # Python result

# Would need:
next_token = model.generate_next(context)  # Neural execution
# Build up state autoregressively
# Return neural-computed result
```

This is what `AutoregressiveVMRunner.run()` tries to do, but it's:
- Extremely slow (1000+ transformer calls per program)
- Broken for basic ops (IMM, JMP, EXIT)
- Not used by tests

## What Should We Do?

### Option 1: Accept Current Design
- System works correctly via DraftVM
- Fast and reliable
- But it's not a "Neural VM" - it's a Python VM with neural validation

### Option 2: Fix Pure Neural Execution
- Fix IMM, JMP, EXIT transformer predictions
- Make autoregressive generation work
- Would be much slower but actually neural

### Option 3: Hybrid Testing
- Keep speculative execution for speed
- Add tests that verify transformer predictions match DraftVM
- Fail tests if transformer predictions are wrong

## Recommendation

**The tests should fail if the transformer predictions are wrong.**

Even if using DraftVM for execution, we should verify that the transformer
**would** have predicted the correct tokens. Otherwise we're not testing
the neural VM weights at all.

### Proposed Fix

Change `verify_speculative_batch` to fail hard on mismatch:
```python
if accepted[b] < draft_lens[b]:
    raise RuntimeError(
        f"Transformer failed to predict token {accepted[b]}: "
        f"predicted {pred}, expected {contexts_with_draft[b][ctx_len + accepted[b]]}"
    )
```

This would make the tests fail (correctly!) and force us to fix the neural weights.

## Current Status

✅ **System works** (via DraftVM Python execution)
✅ **Tests pass** (because they check DraftVM results)
❌ **Transformer predictions wrong** (but tests don't catch it)
❌ **Not actually a Neural VM** (execution is Python, not neural)

The PC_OFFSET=0 conversion improved transformer predictions, which is good,
but the fundamental issue remains: the system is a Python VM, not a Neural VM.
