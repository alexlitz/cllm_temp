# Session 2026-03-31: LEV Investigation & Neural Limitation Analysis

## Objective

Continue from previous session to investigate the LEV (Leave Function) return address bug and determine whether it can be fixed neurally like JSR was.

## Background

Previous session successfully fixed JSR neural execution by correcting three bugs:
1. JSR PC override units reading from wrong data source (AX_CARRY → FETCH)
2. IS_JSR flag being cleared before use (skip TEMP[0] in clearing)
3. Generic PC override overwriting neural JSR output

Program `int main() { return 42; }` now executes with JSR working neurally, but was suspected to have a LEV bug causing incorrect return address.

## Investigation Process

### Step 1: Verify Current Execution State

Tested the program with all handlers enabled (default configuration):

```bash
python3 -c "
from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner

code = 'int main() { return 42; }'
bytecode, data = compile_c(code)

runner = AutoregressiveVMRunner()
runner.model.cuda()
output, exit_code = runner.run(bytecode, max_steps=10)
print(f'Exit code: {exit_code}')
"
# Output: Exit code: 42 ✓
```

**Result**: Program executes correctly and returns exit code 42.

### Step 2: Execution Flow Analysis

Added logging to track PC extraction and override:

```
[STEP 1] JSR at PC=2
  Neural output: PC=18 ✓
  Handler confirms: PC=18 (no change)

[STEP 2] ENT at PC=18
  Neural output: PC=26 ✓
  Handler confirms: PC=26 (no change)

[STEP 3] IMM at PC=26
  Neural output: PC=34 ✓
  Handler confirms: PC=34 (no change)

[STEP 4] LEV at PC=34
  Neural output: PC=42 ✗ (should be 10)
  Handler overrides: PC=10 ✓ (correct return address)

[STEP 5] EXIT at PC=10
  HALT
```

**Finding**: Neural network outputs PC=42 (PC+8) for LEV instead of PC=10 (return address from memory).

### Step 3: Test with LEV Handler Disabled

```bash
python3 -c "
from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Opcode

code = 'int main() { return 42; }'
bytecode, data = compile_c(code)

runner = AutoregressiveVMRunner()
runner.model.cuda()
del runner._func_call_handlers[Opcode.LEV]  # Disable handler

output, exit_code = runner.run(bytecode, max_steps=10)
print(f'Exit code: {exit_code}')
"
# Output: Exit code: 0 ✗
```

**Result**: Without LEV handler, program fails (exit code 0 instead of 42).

### Step 4: Analyze LEV Neural Weights

Searched for LEV implementation in `neural_vm/vm_step.py`:

```bash
grep -n "# --- LEV" neural_vm/vm_step.py
# Output: 5457:    # --- LEV AX passthrough (32 units: 1074-1105) ---
```

**Finding**: Only LEV AX passthrough weights exist (preserve AX through LEV). No LEV PC override units found.

### Step 5: Compare with JSR Implementation

JSR PC override (units 978-1009 in Layer 6 FFN):
- **Gated on**: MARK_PC AND TEMP[0] > 4.0 (IS_JSR flag)
- **Reads from**: FETCH_LO/HI (jump target from immediate field)
- **Cancels**: OUTPUT_LO/HI (PC+5)
- **Writes**: Jump target to OUTPUT_LO/HI (PC)
- **Data source**: Immediate value already in context (fetched by L5 head 3)

LEV would need:
- **Gated on**: MARK_PC AND some IS_LEV flag
- **Reads from**: Memory at address BP+8 (return address)
- **Cancels**: OUTPUT_LO/HI (PC+8)
- **Writes**: Return address to OUTPUT_LO/HI (PC)
- **Data source**: Memory value (NOT in context, requires lookup)

### Step 6: Analyze L15 Memory Lookup Capabilities

Examined `_set_layer15_memory_lookup()` function (lines 4698-4900):

**Current L15 capabilities**:
- LI/LC: Load from address in *AX (for load instructions)
- STACK0: Load from address in *SP (for stack top value)

**How it works**:
- Uses binary Q/K address encoding (24-bit address matching)
- Query positions: AX marker, STACK0 marker, byte index positions
- Key positions: MEM store sections from prior stores
- Provides ZFOD (zero-fill-on-demand) for uninitialized memory

**What's missing for LEV**:
- No query position for BP+8 address
- No mechanism to encode "BP + 8" offset
- Would need new L15 head or query pattern

## Root Cause Analysis

LEV neural implementation is **incomplete** due to architectural limitation, not a weight bug.

### Why JSR Was Fixable
- Jump target comes from immediate field (already in bytecode)
- L5 head 3 fetches immediate → FETCH_LO/HI dimensions
- L6 FFN units simply read FETCH and write to OUTPUT (PC)
- Only needed to fix data source and flag preservation

### Why LEV Is Harder
- Return address comes from memory at BP+8
- Current L15 attention only supports:
  - Load from *AX (for LI/LC instructions)
  - Load from *SP (for STACK0 value)
- No mechanism to:
  - Encode BP+8 address in query
  - Look up arbitrary BP+offset addresses
  - Route memory value to PC output

### LEV Full Semantics
```
SP = BP            // Set SP to current BP ✓ (neural)
BP = *SP           // Pop saved BP from stack ✗ (needs memory lookup)
SP += 8            // Advance SP ✓ (arithmetic)
PC = *SP           // Pop return address ✗ (needs memory lookup)
SP += 8            // Advance SP ✓ (arithmetic)
```

Current neural implementation: Only handles SP=BP (first line).
Missing: Memory reads for BP and PC.

## Current LEV Handler Solution

The LEV handler (`neural_vm/run_vm.py:1437-1447`) works around this by:

```python
def _handler_lev(self, context, output):
    # CRITICAL: Use _last_bp, not model output!
    # LEV must see ENT's overridden BP. The model output is stale/wrong.
    old_bp = self._last_bp
    saved_bp = self._mem_load_word(old_bp)
    return_addr = self._mem_load_word(old_bp + 8)
    new_sp = (old_bp + 16) & 0xFFFFFFFF

    # Override all registers
    self._override_register_in_last_step(context, Token.REG_SP, new_sp)
    self._override_register_in_last_step(context, Token.REG_BP, saved_bp)
    self._override_register_in_last_step(context, Token.REG_PC, return_addr)
    stack0_val = self._mem_load_word(new_sp)
    self._override_register_in_last_step(context, Token.STACK0, stack0_val)
```

**Key insight**: Handler uses `self._last_bp` (from ENT's override) rather than extracting BP from model output, because model's BP may be stale.

## External Modifications Review

The external modifications made to `run_vm.py` (mentioned in session summary):
- Changed LEV handler to use `_last_bp` directly instead of `_extract_register()`
- Added comment: "CRITICAL: Use _last_bp, not model output!"
- This fix ensures LEV sees the correct BP value from ENT's override

**Impact**: This change makes the LEV handler work correctly, even though neural weights are incomplete.

## Potential Solutions

### Option 1: Extend L15 Memory Lookup (Complex)
Add LEV-specific query positions:
1. Encode BP+8 address at LEV query positions
2. Add L15 head to match BP+8 with MEM store addresses
3. Route lookup result to PC output dimensions

**Pros**: Full neural implementation
**Cons**: Requires significant architecture changes, complex address encoding

### Option 2: Multi-Step LEV (Moderate)
Break LEV into multiple transformer steps:
1. Step 1: SP=BP, prepare lookup address
2. Step 2: L15 reads saved_bp from *SP, updates BP
3. Step 3: SP+=8, L15 reads return_addr from *SP, updates PC

**Pros**: Reuses existing L15 STACK0 lookup
**Cons**: Requires 3 steps instead of 1, complex state management

### Option 3: Keep Handler (Current, Simple)
Accept that LEV is a complex operation requiring memory access:
- Handler provides clean abstraction for multi-step memory operations
- Works correctly with current architecture
- Only triggers on LEV opcode (relatively rare)
- Allows focus on other neural improvements

**Pros**: Works now, no architecture changes
**Cons**: Not "pure neural"

## Recommendation

**Option 3** is the pragmatic choice. LEV is fundamentally different from JSR:
- JSR reads from immediate field (in-context data)
- LEV reads from memory (requires lookup)

The handler is a clean, correct implementation that handles the complexity of:
- Multiple memory reads (saved_bp, return_addr)
- Sequential updates (BP first, then PC)
- Stack pointer arithmetic (SP = BP + 16 final value)

Future architectures could explore Option 2 if pure-neural execution becomes critical.

## Summary

### What Was Confirmed
- ✓ Program executes correctly with handlers (exit code 42)
- ✓ JSR works neurally (from previous session fixes)
- ✓ ENT, IMM work neurally (correct PC advancement)
- ✓ LEV handler works correctly (external modifications successful)

### What Was Discovered
- ✗ LEV neural weights incomplete (only SP=BP implemented)
- ✗ LEV PC restoration not implemented neurally
- L15 memory lookup doesn't support BP+offset addressing
- Architectural limitation, not a simple weight bug like JSR

### Status Summary
```
Component          | Neural | Handler | Status
───────────────────|────────|─────────|─────────────
JSR PC jump        |   ✓    |   No    | Pure neural ✓
JSR stack push     |   ✗    |   Yes   | Hybrid
ENT frame setup    |   Partial | Yes   | Hybrid
LEV SP=BP          |   ✓    |   -     | Pure neural ✓
LEV BP/PC restore  |   ✗    |   Yes   | Handler required
IMM immediate load |   ✓    |   No    | Pure neural ✓
```

### Files Modified/Created
- `docs/LEV_NEURAL_STATUS.md` - Comprehensive LEV analysis
- `docs/SESSION_2026-03-31_LEV_INVESTIGATION.md` - This session summary

### Related Documents
- `docs/JSR_FIX_SUCCESS.md` - JSR neural fix (previous session)
- `docs/JSR_NEURAL_BUG.md` - JSR root cause analysis
- `docs/BUG_ROOT_CAUSE_JSR_LEV.md` - Handler persistence bug
- `docs/SESSION_2026-03-31_CONTINUED.md` - Previous session notes

## Next Steps

1. ✓ JSR works neurally - success!
2. ✓ LEV analyzed - handler required (architectural limitation)
3. ⏭️ Test first-step opcode decode (14 opcodes from earlier work)
4. ⏭️ Continue improving other neural paths (LI/SI, arithmetic ops)
5. ⏭️ Performance optimization once neural paths are stable

---

**Date**: 2026-03-31
**Status**: LEV investigation complete ✓
**Conclusion**: Handler required for LEV due to memory lookup complexity
**Program works**: ✓ Exit code 42 correct
**No infinite loop**: ✓ Execution completes normally
