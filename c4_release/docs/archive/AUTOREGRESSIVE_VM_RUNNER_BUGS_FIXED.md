# AutoregressiveVMRunner: Bugs Found and Fixed

## Summary

Investigation revealed **5 critical bugs** in the AutoregressiveVMRunner that prevented it from working correctly. Four have been fixed, one is a compiler/VM semantics issue that needs further investigation.

---

## ✅ BUG 1: Missing GPU Device Handling (FIXED)
**File**: `neural_vm/vm_step.py:816-818`  
**Symptom**: Model ran on CPU even when CUDA was available (12 seconds/token)  
**Root Cause**: `generate_next()` created tensors without specifying device

**Fix Applied**:
```python
# Before:
token_ids = torch.tensor([context], dtype=torch.long)

# After:
device = next(self.parameters()).device
token_ids = torch.tensor([context], dtype=torch.long, device=device)
```

**Impact**: 42x speedup (12s → 0.28s per token)

---

## ✅ BUG 2: Weights Never Loaded (FIXED)
**File**: `neural_vm/run_vm.py:141`  
**Symptom**: Model generated garbage tokens, FFN weights were all zeros  
**Root Cause**: `AutoregressiveVMRunner.__init__` never called `set_vm_weights()`

**Fix Applied**:
```python
class AutoregressiveVMRunner:
    def __init__(self, ...):
        self.model = AutoregressiveVM(...)
        # Added this line:
        set_vm_weights(self.model)
        self.model.eval()
```

**Impact**: Model now generates valid token sequences instead of garbage

---

## ✅ BUG 3: Wrong `_exec_pc()` Logic (FIXED)
**File**: `neural_vm/run_vm.py:1104-1120`  
**Symptom**: Function handlers dispatched for wrong instructions  
**Root Cause**: `_exec_pc()` computed `_last_pc + 5`, but `_last_pc` is the OUTPUT PC (next instruction), not previous

**Example of Bug**:
```
Step 0: Execute CALL at PC=0, output PC=16 (jump to main)
Step 1: _exec_pc() returns 21 (16+5), tries to execute RET at PC=20
        But should execute instruction at PC=16!
```

**Fix Applied**:
```python
def _exec_pc(self):
    """The output PC from the PREVIOUS step is where execution continues.
    So the instruction we're CURRENTLY executing is at that address."""
    if self._last_pc is None:
        return 0
    return self._last_pc  # Not +5!
```

**Impact**: Handlers now dispatch for correct instructions

---

## ✅ BUG 4: Stale Register Values in Handlers (FIXED)
**Files**: `neural_vm/run_vm.py` - `_handler_jsr`, `_handler_ent`, `_handler_lev`  
**Symptom**: JSR handler set SP to 4294967288 instead of 65611  
**Root Cause**: Handlers used `self._last_sp/_last_bp` which weren't updated yet (updated AFTER handlers run)

**Timeline**:
1. Step 0 generates: PC=16, SP=65619
2. JSR handler runs: uses `self._last_sp` (still 0 from init!)
3. new_sp = (0 - 8) & 0xFFFFFFFF = 4294967288 (wrong!)
4. Then runner updates `_last_sp = 65619` (too late)

**Fix Applied**:
```python
# Before (in _handler_jsr):
new_sp = (self._last_sp - 8) & 0xFFFFFFFF

# After:
current_sp = self._extract_register(context, Token.REG_SP)
if current_sp is None:
    current_sp = self._last_sp
new_sp = (current_sp - 8) & 0xFFFFFFFF
```

**Applied to**: `_handler_jsr`, `_handler_ent`, `_handler_lev`  
**Impact**: Handlers now use correct current values, not stale cached values

---

## ❌ BUG 5: Compiler CALL Target Misalignment (UNFIXED)
**File**: `src/compiler.py` (suspected)  
**Symptom**: Programs return exit code 0 instead of expected values  
**Root Cause**: Compiler generates CALL targets that don't align with instruction boundaries

**Example**:
```
Bytecode:
  PC=  0: CALL   16    ← Calls main() at "16"
  PC=  5: HALT   0
  PC= 10: ENT    0     ← main() actually starts here!
  PC= 15: IMM    42
  PC= 20: RET    0
```

**Problem**: 
- CALL targets PC=16
- But instructions are at PC=0, 5, 10, 15, 20, 25
- PC=16 is inside the IMM instruction!

**Status**: Needs investigation into:
1. Is this a compiler bug?
2. Does C4 VM use different PC semantics?
3. Does the working VM (C4TransformerVM) handle this differently?

---

## Testing Status

**After Fixes 1-4**:
- ✅ Model loads weights correctly
- ✅ Model runs on GPU (fast)
- ✅ Handlers dispatch for correct instructions
- ✅ Handlers use correct register values
- ❌ Programs still return wrong exit codes (Bug 5)

**Test Results**:
```
int main() { return 42; }  → Result: ('', 0)  ❌ Expected: ('', 42)
int main() { return 5+7; } → Result: ('', 0)  ❌ Expected: ('', 12)
```

---

## Next Steps

1. **Investigate Bug 5**: Check how the working VM handles CALL targets
2. **Compare with working VM**: The `test_vm.py` tests pass, so see how they avoid this issue
3. **Consider**: AutoregressiveVMRunner may be experimental/incomplete code that was documented as "working" prematurely

---

## Files Modified

- `neural_vm/vm_step.py` - GPU device handling
- `neural_vm/run_vm.py` - Weight loading, _exec_pc, handler fixes

## Commit Message Suggestion

```
Fix 4 critical bugs in AutoregressiveVMRunner

1. Add GPU device handling in generate_next() (42x speedup)
2. Call set_vm_weights() in runner init (fixes zero weights)
3. Fix _exec_pc() to return current PC, not +5 (fixes handler dispatch)
4. Fix handlers to read current registers, not stale _last_* values

Remaining issue: Compiler generates misaligned CALL targets
```
