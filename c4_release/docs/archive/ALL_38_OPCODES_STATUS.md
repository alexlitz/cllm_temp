# All 38 Opcodes - Neural Execution Status

**Goal**: 100% neural execution for all 38 C4 VM opcodes.

**Current Status**: 22/38 neural (58%), 16 require handlers

---

## Quick Summary

| Category | Neural | Handler | Total |
|----------|--------|---------|-------|
| Stack/Address (0-8) | 2 | 7 | 9 |
| Memory (9-13) | 0 | 5 | 5 |
| Bitwise (14-16) | 3 | 0 | 3 |
| Comparison (17-22) | 6 | 0 | 6 |
| Shift (23-24) | 2 | 0 | 2 |
| Arithmetic (25-29) | 5 | 0 | 5 |
| System Calls (30-38) | 1 | 7 | 8 |
| **TOTAL** | **19** | **19** | **38** |

*Note: Some syscalls (OPEN, READ, etc.) may always need handlers for I/O boundary.*

---

## Detailed Status by Opcode

### Stack/Address Operations (0-8)

| # | Op | Status | Handler | Root Cause | Fix Required |
|---|-----|--------|---------|------------|--------------|
| 0 | LEA | ✅ Neural | No | - | - |
| 1 | IMM | ⚠️ Handler | Yes | PRTF format string tracking | Remove `_last_ax` dependency |
| 2 | JMP | ❌ Handler | Yes | L5 head 3 fixed address lookup | Dynamic FETCH for step 2+ |
| 3 | JSR | ❌ Handler | Yes | PC increment broken step 2+ | Same as JMP + return addr |
| 4 | BZ | ❌ Handler | Yes | Same as JMP | Dynamic FETCH |
| 5 | BNZ | ❌ Handler | Yes | Same as JMP | Dynamic FETCH |
| 6 | ENT | ❌ Handler | Yes | OP_ENT=0 at embedding, BP wrong | Fix embedding timing |
| 7 | ADJ | ✅ Neural | No | - | - |
| 8 | LEV | ❌ Handler | Yes | L16 FFN SP/BP outputs 0 | Add L16 SP/BP routing units |

### Memory Operations (9-13)

| # | Op | Status | Handler | Root Cause | Fix Required |
|---|-----|--------|---------|------------|--------------|
| 9 | LI | ❌ Handler | Yes | L14/L15 all-zero with 2+ locals | Fix multi-MEM attention |
| 10 | LC | ❌ Handler | Yes | Same as LI | Same fix |
| 11 | SI | ❌ Handler | Yes | Shadow memory for PRTF | Neural memory tracking |
| 12 | SC | ❌ Handler | Yes | Same as SI | Same fix |
| 13 | PSH | ❌ Handler | Yes | STACK0 lookup broken, PRTF | Fix L8 STACK0 attention |

### Bitwise Operations (14-16) ✅ ALL NEURAL

| # | Op | Status | Handler | Root Cause | Fix Required |
|---|-----|--------|---------|------------|--------------|
| 14 | OR | ✅ Neural | No | - | - |
| 15 | XOR | ✅ Neural | No | - | - |
| 16 | AND | ✅ Neural | No | - | - |

### Comparison Operations (17-22) ✅ ALL NEURAL

| # | Op | Status | Handler | Root Cause | Fix Required |
|---|-----|--------|---------|------------|--------------|
| 17 | EQ | ✅ Neural | No | - | - |
| 18 | NE | ✅ Neural | No | - | - |
| 19 | LT | ✅ Neural | No | - | - |
| 20 | GT | ✅ Neural | No | - | - |
| 21 | LE | ✅ Neural | No | - | - |
| 22 | GE | ✅ Neural | No | - | - |

### Shift Operations (23-24) ✅ ALL NEURAL

| # | Op | Status | Handler | Root Cause | Fix Required |
|---|-----|--------|---------|------------|--------------|
| 23 | SHL | ✅ Neural | No | - | - |
| 24 | SHR | ✅ Neural | No | - | - |

### Arithmetic Operations (25-29) ✅ ALL NEURAL

| # | Op | Status | Handler | Root Cause | Fix Required |
|---|-----|--------|---------|------------|--------------|
| 25 | ADD | ✅ Neural | No | - | - |
| 26 | SUB | ✅ Neural | No | - | - |
| 27 | MUL | ✅ Neural | No | - | - |
| 28 | DIV | ✅ Neural | No | - | - |
| 29 | MOD | ✅ Neural | No | - | - |

### System Calls (30-38)

| # | Op | Status | Handler | Root Cause | Fix Required |
|---|-----|--------|---------|------------|--------------|
| 30 | OPEN | ❌ Handler | Yes | External I/O | I/O boundary (may keep) |
| 31 | READ | ❌ Handler | Yes | External I/O | I/O boundary (may keep) |
| 32 | CLOS | ❌ Handler | Yes | External I/O | I/O boundary (may keep) |
| 33 | PRTF | ❌ Handler | Yes | Format string parsing | Complex - may keep |
| 34 | MALC | ❌ Handler | Yes | Heap allocation | Neural bump allocator |
| 35 | FREE | ❌ Handler | Yes | Heap management | Neural tracking |
| 36 | MSET | ❌ Handler | Yes | Memory bulk ops | Neural implementation |
| 37 | MCMP | ❌ Handler | Yes | Memory comparison | Neural implementation |
| 38 | EXIT | ✅ Neural | No | - | - |

---

## Priority Fix Order

### Phase 1: Control Flow (JMP/BZ/BNZ/JSR)
**Impact**: 4 opcodes → 23/38 neural (61%)

**Root Cause**: L5 attention head 3 uses fixed ADDRESS bits from embedding, not updated dynamically.

**Fix Strategy**:
1. Add dynamic FETCH mechanism that works for step 2+
2. Update L5 to read PC from previous step's output, not embedding
3. Compute target address from immediate operand dynamically

**Files**: `neural_vm/vm_step.py` L5 attention, ~line 1500-1800

### Phase 2: Function Stack (ENT/LEV)
**Impact**: 2 opcodes → 25/38 neural (66%)

**ENT Root Cause**: OP_ENT=0 at embedding time (only non-zero after embedding layer)
**LEV Root Cause**: L16 FFN only routes AX, not SP/BP

**Fix Strategy**:
1. ENT: Propagate OP_ENT earlier OR compute BP in later layer
2. LEV: Add 64 FFN units to L16 for SP/BP byte routing

**Files**: `neural_vm/vm_step.py` L6 FFN (ENT), L16 FFN (LEV)

### Phase 3: Memory (LI/LC/SI/SC/PSH)
**Impact**: 5 opcodes → 30/38 neural (79%)

**Root Cause**: L14/L15 memory attention outputs all-zero when multiple MEM sections exist

**Fix Strategy**:
1. Debug L14 addr head attention scores
2. Fix MEM section generation to have unique addresses
3. Update L15 to handle multiple MEM sections

**Files**: `neural_vm/vm_step.py` L14/L15 attention

### Phase 4: IMM (Remove PRTF Dependency)
**Impact**: 1 opcode → 31/38 neural (82%)

**Root Cause**: `_last_ax` tracking needed for PRTF format string

**Fix Strategy**:
1. Track format string address in neural memory
2. Or: Keep IMM handler only when PRTF is used

### Phase 5: System Calls (MALC/FREE/MSET/MCMP)
**Impact**: 4 opcodes → 35/38 neural (92%)

**Strategy**:
- MALC: Neural bump allocator (update HEAP_PTR)
- FREE: Neural tracking (or no-op for bump allocator)
- MSET/MCMP: Neural memory iteration

### Phase 6: I/O Boundary (OPEN/READ/CLOS/PRTF)
**Impact**: 3-4 opcodes → 38/38 neural (100%)

**Decision Point**: These are external I/O. Options:
1. Keep handlers (not "pure neural" but acceptable)
2. Implement conversational I/O mode (already exists)
3. Tool-use mode for external calls

---

## Root Cause Summary

| Root Cause | Opcodes Affected | Complexity |
|------------|-----------------|------------|
| L5 fixed FETCH address | JMP, BZ, BNZ, JSR | Medium |
| L16 SP/BP routing | LEV | Low |
| ENT embedding timing | ENT | Medium |
| L14/L15 multi-MEM | LI, LC, SI, SC, PSH | High |
| PRTF dependency | IMM, PSH | Low (workaround) |
| External I/O | OPEN, READ, CLOS, PRTF | N/A (boundary) |

---

## Quick Wins (Low Effort, High Impact)

1. **LEV SP/BP** - Add 64 FFN units to L16 (~30 lines of code)
2. **IMM handler removal** - Track format string differently
3. **PSH handler removal** - Once STACK0 fixed

## Hard Problems

1. **L5 dynamic FETCH** - Requires architectural change to address lookup
2. **L14/L15 multi-MEM** - Complex attention pattern for multiple memory regions
3. **ENT embedding timing** - OP_ENT needs earlier propagation

---

## Test Commands

```bash
# Run smoke tests (< 30 sec)
pytest tests/test_smoke.py -v --tb=short

# Run neural parity tests
pytest tests/test_neural_handler_parity.py -v

# Check handler status
python -c "
from neural_vm.run_vm import AutoregressiveVMRunner
r = AutoregressiveVMRunner()
print('Func handlers:', list(r._func_call_handlers.keys()))
print('Syscall handlers:', list(r._syscall_handlers.keys()))
"
```

---

## Success Criteria

| Milestone | Opcodes | Percentage |
|-----------|---------|------------|
| Current | 19/38 | 50% |
| Phase 1 | 23/38 | 61% |
| Phase 2 | 25/38 | 66% |
| Phase 3 | 30/38 | 79% |
| Phase 4 | 31/38 | 82% |
| Phase 5 | 35/38 | 92% |
| Phase 6 | 38/38 | 100% |

---

*Last Updated: 2026-04-17*
