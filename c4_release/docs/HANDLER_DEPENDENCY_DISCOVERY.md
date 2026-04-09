# Handler Dependency Discovery - Critical Findings

## Executive Summary

**Critical Discovery**: The majority of arithmetic, bitwise, and shift operations in the C4 Neural VM rely entirely on Python handlers, not neural weights. The neural weights for these operations are broken and non-functional.

**Impact**: Only **26% (11/42)** of opcodes work with pure neural execution. The previously reported 38% success rate was inflated because tests ran with handlers enabled.

## Background

During systematic opcode verification, we discovered "purity violations" when testing certain opcodes. Investigation revealed that disabling handlers removes the purity violations but exposes that the neural weights return incorrect values.

## Test Methodology

### Phase 1: Testing with Handlers Enabled
Initial tests showed:
- MUL, DIV, MOD: ✓ PASS (exit_code=42)
- ADD, SUB: ✗ FAIL (wrong values)
- OR, XOR, AND, EQ, LT, SHL, SHR: ✗ ERROR (purity violations)

### Phase 2: Testing with Handlers Disabled
Disabled handlers for specific opcodes to test pure neural execution:

```python
runner = AutoregressiveVMRunner()
del runner._func_call_handlers[Opcode.ADD]  # Disable handler
# Test shows neural weights alone
```

**Results**:
- ADD (10 + 32): Returns 10, expected 42 ✗
- SUB (50 - 8): Returns 34, expected 42 ✗
- MUL (6 * 7): Returns 0, expected 42 ✗
- DIV (84 / 2): Returns 0, expected 42 ✗
- OR (32 | 10): Returns 32, expected 42 ✗
- XOR (40 ^ 2): No purity violation, but returns wrong value ✗
- AND (63 & 42): Returns wrong value ✗

## Handler Configuration

Handlers defined in `neural_vm/run_vm.py` lines 214-235:

```python
self._func_call_handlers = {
    Opcode.IMM: self._handler_imm,
    Opcode.LEA: self._handler_lea,
    Opcode.JSR: self._handler_jsr,
    Opcode.ENT: self._handler_ent,
    Opcode.LEV: self._handler_lev,
    # Stack operations
    Opcode.PSH: self._handler_psh,
    # Arithmetic operations (neural weights broken, using fallback) ← COMMENT
    Opcode.ADD: self._handler_add,
    Opcode.SUB: self._handler_sub,
    Opcode.MUL: self._handler_mul,
    Opcode.DIV: self._handler_div,
    Opcode.MOD: self._handler_mod,
    # Bitwise operations (neural weights broken, using fallback) ← COMMENT
    Opcode.OR: self._handler_or,
    Opcode.XOR: self._handler_xor,
    Opcode.AND: self._handler_and,
    # Shift operations (neural weights broken, using fallback) ← COMMENT
    Opcode.SHL: self._handler_shl,
    Opcode.SHR: self._handler_shr,
}
```

**Note**: The comments explicitly state "neural weights broken, using fallback".

## Opcode Classification

### Pure Neural (11 opcodes = 26%)

**Working without any handlers:**
- **Stack/Control**: IMM, JMP, JSR, BZ, BNZ, EXIT
- **Memory**: LI, SI (via L15 attention)
- **Comparison**: NE, LE
- **System**: PRTF
- **I/O**: PUTCHAR

### Handler-Dependent (11 opcodes = 26%)

**Require handlers for correct operation:**
- **Arithmetic**: ADD, SUB, MUL, DIV, MOD
- **Bitwise**: OR, XOR, AND
- **Shift**: SHL, SHR
- **Stack/Function**: PSH, ENT, LEV (stack frame operations)

### Broken (4 opcodes = 10%)

**No handlers, neural weights don't work:**
- **Comparison**: EQ, LT (purity violations)
- **Comparison**: GT, GE (return 0 instead of 1)

### Untested (16 opcodes = 38%)

**Require special setup or not emitted by compiler:**
- **Stack**: LEA, ADJ
- **Memory**: LC, SC
- **System**: OPEN, READ, CLOS, MALC, FREE, MSET, MCMP
- **Control**: NOP, POP, BLT, BGE
- **I/O**: GETCHAR

## Detailed Failure Analysis

### ADD Operation

**Test**: `int main() { return 10 + 32; }`

**Bytecode**:
```
[3] PC=26: IMM  10    # AX = 10
[4] PC=34: PSH   0    # Push 10 to stack
[5] PC=42: IMM  32    # AX = 32
[6] PC=50: ADD   0    # Should: pop stack (10), AX = 10 + 32 = 42
```

**With handler disabled**:
- Returns: 10 (first operand only)
- Expected: 42
- **Hypothesis**: Neural weights read first operand (from stack) but don't perform addition

**Neural Architecture** (Layer 8 FFN, line 3878):
```python
# ADD: lo nibble (256 units)
for a in range(16):
    for b in range(16):
        result = (a + b) % 16
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.ALU_LO + a] = S
        ffn.W_up[unit, BD.AX_CARRY_LO + b] = S
        ffn.b_up[unit] = -S * 2.5  # 3-way AND
        ffn.W_gate[unit, BD.OP_ADD] = 1.0
        ffn.W_down[BD.OUTPUT_LO + result, unit] = 2.0 / S
```

**Issue**: Weights look correct structurally, but execution returns wrong value.

### MUL Operation

**Test**: `int main() { return 6 * 7; }`

**With handler disabled**:
- Returns: 0
- Expected: 42

**Hypothesis**: Neural MUL circuit not activating at all, defaulting to 0.

### Bitwise Operations (OR, XOR, AND)

**Test**: `int main() { return 32 | 10; }`

**With handler disabled**:
- OR returns: 32 (first operand only)
- XOR: Wrong value
- AND: Wrong value

**Pattern**: Similar to ADD - returns first operand instead of computing result.

## Purity Violations Explained

**What are purity violations?**

Error message: `PURITY VIOLATION: forward() structure is invalid!`

**Root Cause**: These occur when handlers are enabled for opcodes that also have neural weight implementations. The handler modifies the output, creating a conflict between neural generation and handler override.

**Why disabling handlers fixes purity violations**: When handlers are disabled, the model generates tokens purely from neural weights, eliminating the conflict (but revealing the weights don't work correctly).

**Opcodes with purity violations**:
- OR, XOR, AND: Have handlers
- EQ, LT: NO handlers (purity violation from different source)
- SHL, SHR: Have handlers

## Impact Assessment

### Previous Test Results (Misleading)

**Reported**: 16/42 (38%) opcodes working
**Reality**: Many "working" opcodes only worked because of handlers

### Actual Pure Neural Performance

**Pure Neural**: 11/42 (26%) opcodes
**Handler-Dependent**: 11/42 (26%) opcodes
**Broken**: 4/42 (10%) opcodes
**Untested**: 16/42 (38%) opcodes

### Handler Reliance

The VM currently relies heavily on Python fallback handlers for:
- All arithmetic operations (ADD, SUB, MUL, DIV, MOD)
- All bitwise operations (OR, XOR, AND)
- All shift operations (SHL, SHR)
- Stack frame operations (PSH, ENT, LEV)

This means the "neural" VM is actually a hybrid system where ~26% of operations run neurally and ~26% run via Python handlers.

## Why This Matters

### Performance
- Handlers require Python interpretation overhead
- Neural operations run in transformer forward pass (fast)
- Handler-dependent opcodes much slower than pure neural

### Transparency
- Tests should clearly distinguish pure neural vs handler-dependent
- Current test suite masked the handler dependency
- Need separate test modes: `pure_neural=True/False`

### Future Work
- Can neural weights be fixed?
- Should handlers be kept as permanent solution?
- Need architectural changes for pure neural arithmetic?

## Comparison with Working Opcodes

### Why JSR Works Neurally

**JSR** (Jump Subroutine):
- Reads jump target from immediate field (in bytecode)
- L5 head 3 fetches immediate → FETCH_LO/HI
- L6 FFN reads FETCH, writes to OUTPUT (PC)
- **Data source**: Already in context ✓

### Why ADD Doesn't Work Neurally

**ADD**:
- Reads two operands: one from stack (STACK0), one from AX
- L7 attention gathers STACK0 → ALU_LO
- L3 attention carries AX → AX_CARRY_LO
- L8 FFN should compute sum
- **Issue**: Returns first operand only, not sum ✗

### Key Difference

Working opcodes (JSR, IMM, JMP):
- Simple data routing (copy, relay)
- No complex computation

Broken opcodes (ADD, MUL, OR):
- Require actual computation
- Multi-stage pipeline (gather → compute → output)
- More complex weight dependencies

## Evidence Trail

### Test Code

```python
# Test ADD without handler
from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Opcode
import torch

code = "int main() { return 10 + 32; }"
bytecode, data = compile_c(code)

runner = AutoregressiveVMRunner()
del runner._func_call_handlers[Opcode.ADD]  # Disable handler

runner.model.cuda()
_, exit_code = runner.run(bytecode, max_steps=50)

print(f"Result: {exit_code}")  # Output: 10 (WRONG)
print(f"Expected: 42")
```

### Handler Implementation

Example handler from `neural_vm/run_vm.py:1241`:

```python
def _handler_add(self, context, output):
    """ADD -- pop stack, AX = stack_val + AX."""
    stack_val = self._last_sp_value()  # Get top of stack
    ax = self._last_ax
    result = (stack_val + ax) & 0xFFFFFFFF
    self._override_ax_in_last_step(context, result)
    # SP already incremented by generic binary pop handling
```

**This performs the actual addition in Python**, not neural weights.

## Recommended Actions

### Immediate (Documentation)
1. ✓ Update OPCODE_TEST_STATUS.md with handler dependency info
2. ✓ Create this comprehensive discovery document
3. Create test suite that validates pure neural vs handler operation

### Investigation (Why Neural Weights Fail)
1. Debug Layer 7 operand gathering (ALU_LO population)
2. Debug Layer 8 ADD circuit activation
3. Check if ALU outputs are being overridden by passthrough
4. Trace activation values through the pipeline

### Fixes (Attempt to Enable Pure Neural)
1. Fix neural ADD/SUB if possible
2. Document if architectural changes needed
3. Determine if handlers should be permanent or temporary

### Protection (Prevent Future Issues)
1. Create purity mode enforcement
2. Add tests that fail if handlers are added without documentation
3. Protect test code from modification

## Related Documents

- `docs/OPCODE_TEST_STATUS.md` - Opcode test results (needs update)
- `docs/SESSION_2026-04-07_OPCODE_VERIFICATION.md` - Session summary
- `neural_vm/run_vm.py:214-235` - Handler definitions
- `neural_vm/vm_step.py:3878-3915` - ADD/SUB neural weights

## Date

2026-04-07

---

**Status**: Handler dependency discovered and documented ✓
**Pure Neural**: 11/42 opcodes (26%)
**Handler-Dependent**: 11/42 opcodes (26%)
**Next**: Investigate why neural arithmetic fails
