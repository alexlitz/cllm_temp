# Runner Override Removal Status

## ✓ Completed: JSR (Jump to Subroutine)

**File**: `neural_vm/run_vm.py` lines 1153-1161

**All overrides removed**:
- ~~PC = target~~ (Line 4934-4969: Layer 6 FFN JSR PC override)
- ~~SP -= 8~~ (Line 4872-4895: Layer 6 FFN JSR SP)
- ~~STACK0 = return_addr~~ (Line 4897-4932: Layer 6 FFN JSR STACK0)

**Implementation**: Pure transformer weights via fetch architecture:
1. Layer 5 Head 3: Fetch immediate from CODE section → AX_CARRY
2. Layer 5 FFN: First-step JSR decode → TEMP[0] flag
3. Layer 6 FFN: Conditional routing based on TEMP[0] + AX_CARRY

**Test Result**: JSR 16 → PC=16 ✓

---

## ⏳ Remaining: ENT (Enter Function)

**File**: `neural_vm/run_vm.py` lines 1188-1190

**Current overrides**:
```python
self._override_register_in_last_step(context, Token.REG_SP, new_sp)
self._override_register_in_last_step(context, Token.REG_BP, new_bp)
self._override_register_in_last_step(context, Token.STACK0, old_bp)
```

**Transformer weights exist**: Lines 4970-5041 in vm_step.py
- ENT STACK0 = old_BP (units 978-1009)
- ENT BP = SP - 8 (units 1010-1041)
- ENT AX passthrough (units 1042-1073)

**Status**: Should be removable - weights already exist

**Approach**: Same as JSR - verify weights handle all cases, remove overrides, test

---

## ⏳ Remaining: LEV (Leave Function)

**File**: `neural_vm/run_vm.py` lines 1207-1209, 1212

**Current overrides**:
```python
self._override_register_in_last_step(context, Token.REG_SP, new_sp)
self._override_register_in_last_step(context, Token.REG_BP, saved_bp)
self._override_register_in_last_step(context, Token.REG_PC, return_addr)
self._override_register_in_last_step(context, Token.STACK0, stack0_val)
```

**Transformer weights exist**: Lines 5042-5107 in vm_step.py (LEV AX/PC/SP/BP passthrough)

**Status**: Should be removable - weights already exist

**Challenge**: LEV restores PC from STACK0 (popped from stack), similar architectural challenge as JSR but in reverse

---

## ⏳ Remaining: PSH (Push)

**File**: `neural_vm/run_vm.py` lines 874, 876, 894

**Current overrides**:
- SP -= 8
- STACK0 = AX (for PSH)
- SP -= 8 (for ADJ with negative immediate)

**Status**: Likely handled by transformer weights (PSH is a common operation)

---

## ⏳ Remaining: I/O Operations

**File**: `neural_vm/run_vm.py` lines 300, 306, 316, 361, 445, 481, 611

**Operations**: LI, LC, OPEN, READ, WRITE, etc.

**Current overrides**: Various AX register updates for I/O results

**Status**: I/O operations are special - they interface with external world
- May require different approach than pure ALU operations
- Consider if I/O handlers should be exceptions to purity requirement

---

## Summary

| Operation | Status | Transformer Weights | Notes |
|-----------|--------|---------------------|-------|
| **JSR** | ✓ DONE | Yes (4872-4969) | All overrides removed, fetch architecture works |
| **ENT** | TODO | Yes (4970-5041) | Should be straightforward |
| **LEV** | TODO | Yes (5042-5107) | PC restore from STACK0 needs attention |
| **PSH** | TODO | Likely | Common operation, likely handled |
| **ADJ** | TODO | Likely | SP adjustment |
| **I/O** | DEFER | N/A | External interface, may need special handling |

## Next Steps

1. **Remove ENT overrides** - weights exist, should be direct
2. **Remove LEV overrides** - more complex due to PC restore from stack
3. **Test PSH/ADJ** - verify transformer handles without overrides
4. **Evaluate I/O** - decide if purity requirement applies

## Architecture Pattern Established

**For pure instruction execution**:
1. Fetch immediate values to PC marker via Layer 5 attention
2. Route opcode flags via TEMP dimensions
3. Layer 6 FFN conditional override based on flags + fetched values
4. Remove runner overrides

**Key enabler**: ADDR_KEY content-based addressing allows backward attention to CODE section despite causal masking.
