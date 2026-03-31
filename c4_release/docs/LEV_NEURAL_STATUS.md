# LEV Neural Implementation Status

## Summary

LEV (Leave Function) is **partially implemented** neurally. The handler in run_vm.py compensates for missing neural memory lookup capabilities.

## Current State

### What Works (with Handler)
- Program `int main() { return 42; }` executes correctly
- Exit code: 42 ✓
- Execution flow: JSR → ENT → IMM → LEV → EXIT

### What Works Neurally (without Handler)
- **SP = BP**: Register copy (first part of LEV semantics)
- **AX passthrough**: Preserves AX value through LEV (units 1074-1105 in L6 FFN)

### What Doesn't Work Neurally
- **BP restoration**: Reading saved_bp from memory[BP]
- **PC restoration**: Reading return_addr from memory[BP+8]
- Without handler: outputs PC=42 (PC+8) instead of PC=10 (return address)
- Without handler: exit_code=0 instead of 42

## Root Cause: L15 Memory Lookup Limitation

### LEV Requirements
LEV semantics (C4 spec):
```
SP = BP            // Set SP to current BP
BP = *SP           // Pop saved BP from stack
SP += 8            // Advance SP past saved BP
PC = *SP           // Pop return address from stack
SP += 8            // Advance SP past return address
```

To implement neurally, LEV needs to:
1. Read from memory address BP (to get saved_bp)
2. Read from memory address BP+8 (to get return_addr)
3. Set PC = return_addr

### Current L15 Capabilities
L15 attention (layer 15) provides memory lookup for:
- **LI/LC**: Load from address in *AX
- **STACK0**: Load from address in *SP

Implementation details (lines 4698-4900 in vm_step.py):
- Uses binary Q/K address encoding (24-bit)
- Matches address bits between query (LI/LC/STACK0 position) and key (MEM store positions)
- Provides ZFOD (zero-fill-on-demand) for uninitialized memory

### What's Missing
LEV needs memory lookup at **BP+8**, but:
1. No mechanism to encode "BP+8" address in the context
2. No L15 query position for LEV return address lookup
3. Would need:
   - BP value relay to LEV query position
   - +8 offset computation in queries
   - New L15 head or query pattern

## Current Handler Implementation

The LEV handler (run_vm.py:1437-1447) works around this by:

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

Key insight: Handler uses `self._last_bp` (ENT's overridden value) not model output, because model's BP may be stale or wrong.

## Test Results

### With Handler Enabled (default)
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

Execution trace:
```
[STEP 1] JSR at PC=2  → Neural PC=18 ✓ → Handler confirms PC=18
[STEP 2] ENT at PC=18 → Neural PC=26 ✓ → Handler confirms PC=26
[STEP 3] IMM at PC=26 → Neural PC=34 ✓ → Handler confirms PC=34
[STEP 4] LEV at PC=34 → Neural PC=42 ✗ → Handler overrides to PC=10 ✓
[STEP 5] EXIT at PC=10 → HALT
```

### With Handler Disabled
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

Without handler, LEV outputs PC=42 (PC+8) instead of reading return address from memory.

## Comparison with JSR

### JSR (Fixed in Previous Session)
- **Source**: Immediate value (jump target in instruction)
- **Mechanism**: L5 head 3 fetches immediate → FETCH_LO/HI
- **PC override**: L6 FFN units read FETCH, write to OUTPUT (PC)
- **Complexity**: Low (data already in context)
- **Status**: ✓ Works neurally without handler

### LEV (Current Issue)
- **Source**: Memory value (return address at *BP+8)
- **Mechanism**: Would need L15 attention lookup at BP+8 address
- **PC override**: Would need to route memory value to OUTPUT (PC)
- **Complexity**: High (requires new L15 query pattern, address computation)
- **Status**: ✗ Requires handler

## Path Forward

### Option 1: Extend L15 Memory Lookup (Architectural Change)
Add LEV-specific memory lookup to L15:
1. Add LEV query position markers to context
2. Encode BP+8 address at query positions
3. Add L15 head to match BP+8 with MEM store addresses
4. Route lookup result to PC output dimensions

**Pros**: Full neural implementation, no handler needed
**Cons**: Complex, requires redesigning L15 attention patterns

### Option 2: Use Existing STACK0 Mechanism (Simpler)
Insight: After `SP = BP`, the stack pointer points to saved_bp. After `SP += 8`, it points to return_addr.

Could potentially:
1. Have LEV first do `SP = BP` (already works)
2. Have L15 STACK0 lookup read from *SP (saved_bp)
3. Update BP with that value
4. Advance SP by 8
5. Have L15 STACK0 lookup read from *SP (return_addr)
6. Update PC with that value

**Pros**: Reuses existing L15 STACK0 lookup
**Cons**: Requires multiple steps or complex multi-value routing within one step

### Option 3: Accept Handler (Current Approach)
Keep LEV handler as permanent shim:
- Works correctly with current architecture
- Minimal code complexity
- Only triggers on LEV opcode (rare in practice)
- Other opcodes (LI, SI, etc.) use full neural paths

**Pros**: Works now, no architecture changes needed
**Cons**: Not "pure neural", depends on runner-side logic

## Recommendation

**Option 3** (keep handler) is pragmatic for now. LEV is a complex operation requiring multiple memory reads and register updates in a specific sequence. The handler is a clean abstraction that handles this complexity correctly.

Future work could explore Option 2 if pure-neural execution becomes a hard requirement.

## Related Issues

- **ENT handler**: Also needed for full frame setup (BP push, SP adjustment)
- **JSR handler**: Still used for stack operations (SP-=8, push return_addr) even though PC jump works neurally
- **Memory operations**: LI/SI work neurally via L15, but require MEM section retention

## Files

- `neural_vm/run_vm.py:1437-1447` - LEV handler implementation
- `neural_vm/vm_step.py:5457-5471` - LEV AX passthrough weights
- `neural_vm/vm_step.py:4698-4900` - L15 memory lookup (LI/LC/STACK0)

## Date

2026-03-31

---

**Status**: LEV works correctly with handler ✓
**Neural implementation**: Partial (SP=BP only)
**Handler required**: Yes (for BP and PC restoration)
**Exit code**: Correct (42) ✓
