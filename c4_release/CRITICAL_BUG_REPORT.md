# CRITICAL BUG REPORT: Transformer Execution Failure

**Date:** 2026-03-27
**Severity:** CRITICAL
**Status:** ❌ SYSTEM NOT WORKING AS DESIGNED

---

## 🔴 EXECUTIVE SUMMARY

**The transformer does not execute programs correctly.** Tests appear to pass because the system falls back to DraftVM execution, not because the transformer works.

### Key Findings

1. ❌ Transformer predicts wrong PC values for JSR instructions
2. ❌ Speculative decoding uses DraftVM tokens instead of transformer tokens
3. ❌ "Successful" test results come from DraftVM, not transformer
4. ❌ Strict mode reveals 97% prediction failure rate

**The system is running DraftVM with a broken validator, not a neural VM.**

---

## 🔍 ROOT CAUSE ANALYSIS

### Bug #1: Transformer JSR Execution

**Instruction:** JSR 16 (Jump to Subroutine at address 16)

**Expected behavior:**
```
Input:  Context with JSR 16 instruction
Output: PC=16 (jump target)
```

**Actual behavior:**
```
Input:  Context with JSR 16 instruction
Output: PC=8 (return address, not jump target)
```

**Evidence:**
```
DraftVM:     JSR 16 → PC = 16 ✓ (correct jump target)
Transformer: JSR 16 → PC = 8  ✗ (return address)

Difference: 8 bytes (one INSTR_WIDTH)
```

**Diagnosis:**

The transformer learned to output the **return address** that JSR pushes to the stack, not the **jump target** where execution continues.

```
JSR 16 execution:
1. Advance PC: 0 → 8
2. Push return address: stack[sp] = 8
3. Jump to target: PC = 16  ← This is what should be output
4. Continue execution at PC=16

Transformer outputs: 8 (step 2)
Should output: 16 (step 3)
```

### Bug #2: Speculative Decoding Architecture

**File:** `neural_vm/batch_runner_v2.py:208`

**Code:**
```python
# Update contexts with accepted tokens
for i, (ctx, draft, accepted) in enumerate(zip(self.contexts, draft_tokens_batch, accepted_batch)):
    self.contexts[i] = ctx + draft[:accepted]  # ← Uses DRAFT tokens!
```

**Problem:**

When transformer predictions don't match DraftVM:
- `accepted` = 1 (only first token matches)
- Code adds `draft[:1]` from **DraftVM**, not transformer prediction
- Execution continues with DraftVM's tokens
- Transformer is never used for actual generation

**Correct speculative decoding:**
```python
# Should use TRANSFORMER'S predictions, not draft!
for i in range(accepted):
    self.contexts[i].append(transformer_predictions[i])
```

**What's happening now:**
```python
# Uses draft (DraftVM) tokens even when transformer disagrees
for i in range(accepted):
    self.contexts[i].append(draft[i])  # ← Wrong!
```

---

## 📊 TEST RESULTS EXPLAINED

### Why Tests "Pass" Without Strict Mode

```python
# Test
runner = UltraBatchRunner(batch_size=8, strict=False)
results = runner.run_batch([bytecode])
# Results: [42, 42, 42] ✓
```

**Why it works:**
1. DraftVM generates draft tokens with PC=16 (correct)
2. Transformer validates, predicts PC=8 (wrong)
3. System detects mismatch (accepted=1)
4. **Uses DraftVM's tokens anyway** (PC=16)
5. Program continues with DraftVM's correct execution
6. Final result: 42 (correct, because DraftVM executed it)

**This is NOT the transformer executing - it's DraftVM!**

### Why Strict Mode Fails

```python
runner = UltraBatchRunner(batch_size=8, strict=True)
# AssertionError: Match rate: 1/35 (2.9%)
```

**Why it fails:**
1. Strict mode checks every token prediction
2. First token matches (REG_PC marker)
3. Second token mismatches (PC value: 8 vs 16)
4. Strict mode raises AssertionError immediately
5. No fallback to DraftVM in strict mode

**Strict mode correctly identifies that transformer predictions are wrong.**

---

## 🧪 REPRODUCTION

### Minimal Test Case

```python
import torch
from neural_vm.vm_step import AutoregressiveVM, Token, set_vm_weights
from neural_vm.speculative import DraftVM

# Simple JSR instruction
bytecode = [0x1003]  # JSR 16

# Context
context = [Token.CODE_START, 3, 16, 0, 0, 0, Token.CODE_END, Token.DATA_START, Token.DATA_END]

# DraftVM
vm = DraftVM(bytecode)
vm.step()
print(f"DraftVM PC: {vm.pc}")  # Output: 16

# Transformer
model = AutoregressiveVM()
set_vm_weights(model)
token_ids = torch.tensor([context], dtype=torch.long)
logits = model.forward(token_ids)

# Predict PC
pred_tokens = []
for i in range(2):
    if i > 0:
        token_ids = torch.tensor([context + pred_tokens], dtype=torch.long)
        logits = model.forward(token_ids)
    pred = logits[0, -1, :].argmax(-1).item()
    pred_tokens.append(pred)

print(f"Transformer PC byte 0: {pred_tokens[1]}")  # Output: 8
```

**Expected:** Both output 16
**Actual:** DraftVM=16, Transformer=8
**Failure:** ✗

---

## 💥 IMPACT ASSESSMENT

### What Doesn't Work

❌ **Transformer execution** - Predicts wrong PC values
❌ **Speculative decoding** - Architectural bug, uses draft instead of target
❌ **Strict mode** - Correctly identifies failures
❌ **Neural VM** - Not actually running on transformer

### What Appears to Work (But Doesn't)

⚠️ **Batch runner** - Works only because it's running DraftVM
⚠️ **Program execution** - Correct results from DraftVM, not transformer
⚠️ **Test suites** - Passing because DraftVM works, not transformer

### What Actually Works

✓ **DraftVM** - Executes correctly
✓ **Purity architecture** - Forward pass is pure
✓ **Code structure** - Well organized
✓ **Test infrastructure** - Detects problems (strict mode)

---

## 🔧 REQUIRED FIXES

### Fix #1: Transformer JSR Behavior (CRITICAL)

**Options:**

**A. Re-train model weights**
- Train with correct JSR semantics (output jump target, not return address)
- Requires training infrastructure

**B. Fix weight loading**
- Check if JSR weights exist but aren't loaded correctly
- Investigate `set_vm_weights()` in `neural_vm/vm_step.py`

**C. Fix PC semantics**
- Investigate PC_OFFSET mismatch (trained with 2, running with 0)
- Check bytecode encoding matches training data

**D. Verify training data**
- Confirm training data had correct JSR behavior
- May need to regenerate training data

### Fix #2: Speculative Decoding Architecture (CRITICAL)

**File:** `neural_vm/batch_runner_v2.py`

**Current (wrong):**
```python
# Line 208
self.contexts[i] = ctx + draft[:accepted]  # Uses DraftVM tokens
```

**Should be:**
```python
# Use transformer's predictions, not draft
for i, (ctx, accepted_count) in enumerate(zip(self.contexts, accepted_batch)):
    # Get transformer's actual predictions
    transformer_tokens = self._get_transformer_predictions(i, accepted_count)
    self.contexts[i] = ctx + transformer_tokens
```

**Note:** This will cause all tests to fail until Fix #1 is implemented, because transformer predictions are wrong.

### Fix #3: Testing Strategy

**Short term:**
1. Keep strict=False until transformer is fixed
2. Document that system is running DraftVM
3. Add warnings about transformer not being used

**Long term:**
1. Fix transformer JSR behavior
2. Enable strict mode
3. Verify transformer can execute independently
4. Benchmark transformer-only execution

---

## 📋 INVESTIGATION CHECKLIST

- [ ] Check `set_vm_weights()` for JSR weight loading
- [ ] Verify PC_OFFSET matches training (2 vs 0)
- [ ] Examine bytecode encoding in training data
- [ ] Test other jump instructions (JMP, BZ, BNZ)
- [ ] Check if problem exists in other opcodes
- [ ] Verify ADDR_KEY augmentation is working
- [ ] Test with original training configuration
- [ ] Compare DraftVM vs Transformer execution for all opcodes

---

## 🎯 RECOMMENDATIONS

### Immediate Actions

1. **Document current state**
   - System is running DraftVM, not transformer
   - Mark transformer execution as "not working"
   - Update all documentation

2. **Disable misleading tests**
   - Tests without strict=True don't test transformer
   - Only strict mode tests are meaningful
   - Add warnings to non-strict tests

3. **Investigate weight loading**
   - Most likely cause of bug
   - Check if JSR weights exist
   - Verify all opcodes are loaded correctly

### Short Term

1. **Fix transformer JSR behavior**
   - Priority #1
   - Required for system to work as designed

2. **Fix speculative decoding architecture**
   - Priority #2
   - Must use transformer predictions, not draft

3. **Enable strict mode**
   - After fixes
   - Verify every token prediction

### Long Term

1. **Verify all opcodes**
   - Test each opcode individually
   - Build comprehensive opcode test suite

2. **Benchmark transformer performance**
   - Once working
   - Compare to DraftVM
   - Measure speedup from speculation

3. **Production deployment**
   - Only after transformer works correctly
   - Document performance characteristics

---

## 💬 CONCLUSION

**The system is not working as designed.**

While tests appear to pass, this is because:
1. DraftVM executes programs correctly
2. Batch runner falls back to DraftVM tokens
3. Transformer predictions are ignored

The transformer cannot execute JSR instructions correctly, predicting return addresses instead of jump targets. This is a fundamental execution bug that must be fixed before the system can be considered a "neural VM."

**Status:** ❌ CRITICAL - Requires immediate investigation and fixes

**Next Steps:**
1. Investigate weight loading for JSR
2. Fix transformer JSR behavior
3. Fix speculative decoding architecture
4. Re-test with strict mode
5. Verify transformer can execute independently

---

**Reported by:** Claude Sonnet 4.5
**Date:** 2026-03-27
**Files:** `neural_vm/batch_runner_v2.py`, `neural_vm/vm_step.py`, `neural_vm/speculative.py`
