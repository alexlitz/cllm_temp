# Memory Test Coverage Analysis

**Date**: 2026-04-07

## Executive Summary

**Question 1**: Are the code length estimates accurate?
**Answer**: You're right - both are significantly longer than initially stated:
- Hand-set weights: **487 lines** (accurate, but part of 6003-line file)
- Compiler infrastructure: **3,704 lines** across 3 files (not 400)

**Question 2**: Do we have tests for memory retention and overwriting?
**Answer**: **Partial coverage** - we have basic tests but gaps exist:
- ✅ Basic overwrite test (1 test)
- ⚠️ Limited long-term retention tests
- ❌ No stress tests for arbitrary-length retention
- ❌ No tests for many overwrites to same location

---

## Part 1: Actual Code Lengths (CORRECTED)

### Hand-Set Weights Function

**File**: `neural_vm/vm_step.py`

```
Total file size:           6,003 lines
set_vm_weights function:     487 lines (lines 1281-1768)
Percentage of file:          8.1%
```

**Breakdown**:
- Function definition and docstring: ~40 lines
- Embedding setup: ~100 lines
- Layer 0-15 weight setting: ~300 lines
- Helper function calls: ~47 lines

**Note**: The function itself is 487 lines, but it calls many helper functions (`_set_threshold_attn`, `_set_phase_a_ffn`, etc.) that add significantly more code. Including helpers, the **total hand-set weight infrastructure is ~2,000+ lines**.

### Compiled Weight Infrastructure

**Files and Line Counts**:

```
graph_weight_compiler.py:    1,606 lines
weight_compiler.py:            787 lines
nibble_weight_compiler.py:   1,311 lines
────────────────────────────────────────
TOTAL:                       3,704 lines
```

**Status**: Development/experimental (not used in production)

### Comparison

| Metric | Hand-Set | Compiled | Ratio |
|--------|----------|----------|-------|
| Core function | 487 lines | N/A | - |
| Total infrastructure | ~2,000 lines | 3,704 lines | 1.9x |
| Status | Production ✅ | Development ⚠️ | - |

**Correction**: Both approaches are much larger than initial estimates. The compiler is nearly 2x the size of the hand-set infrastructure when including all supporting code.

---

## Part 2: Memory Test Coverage

### Tests Found

#### 1. **Basic Memory Operations** (test_opcodes.py)

**Line 1940**: `test_si_li_round_trip_values`
```python
"""SI+LI round-trip for various values including lo-nibble-0."""
for v in [0, 1, 15, 16, 127, 128, 255, 256, 1000, 65535, 0xFFFFFFFF]:
    bytecode = [
        Opcode.IMM | (0x100 << 8), # addr
        Opcode.PSH,
        Opcode.IMM | (v << 8),     # value
        Opcode.SI,                  # memory[addr] = v
        Opcode.IMM | (0x100 << 8),
        Opcode.LI,                  # AX = memory[addr]
        Opcode.EXIT,
    ]
    self.assertEqual(ec, v, f"SI+LI round-trip failed for v={v}")
```
**Coverage**: ✅ Various values, single write-read cycle

**Line 1958**: `test_si_li_round_trip`
```python
"""SI stores AX at *STACK0, then LI loads from the same address."""
```
**Coverage**: ✅ Single write-read at stack pointer

**Line 1978**: `test_sc_lc_round_trip`
```python
"""SC stores byte, LC loads byte."""
```
**Coverage**: ✅ Byte-level write-read

#### 2. **Zero-on-First-Read (ZFOD)** Tests

**Line 1996**: `test_zfod_li`
```python
"""LI from uninitialized address should return 0 (ZFOD)."""
```
**Coverage**: ✅ Reading uninitialized memory

**Line 2006**: `test_zfod_lc`
```python
"""LC from uninitialized address should return 0 (ZFOD)."""
```
**Coverage**: ✅ Byte read from uninitialized memory

#### 3. **Overwrite Test** ⭐ (Only one!)

**Line 2027**: `test_multiple_writes_latest_wins`
```python
"""Two SI to same address, then LI should read latest write."""
addr = 0x100
bytecode = [
    Opcode.IMM | (addr << 8),  # AX = addr
    Opcode.PSH,                # STACK0 = addr
    Opcode.IMM | (10 << 8),    # AX = 10
    Opcode.SI,                 # memory[addr] = 10
    Opcode.IMM | (20 << 8),    # AX = 20
    Opcode.SI,                 # memory[addr] = 20 (overwrite)
    Opcode.IMM | (addr << 8),  # AX = addr
    Opcode.LI,                 # AX = memory[addr] = 20 (latest)
    Opcode.EXIT,
]
self.assertEqual(ec, 20, "Latest write should win")
```
**Coverage**: ✅ **2 writes to same address**, verify latest wins

#### 4. **Stack Memory Tests**

**Lines 1340-1353**: PSH/STACK0 tests
```python
test_psh_stack0_set()
test_psh_stack0_zero()
test_psh_stack0_255()
```
**Coverage**: ✅ Stack writes with various values

**Line 1859**: `test_psh_generates_mem_section`
```python
"""PSH should generate MEM marker + 9 tokens."""
```
**Coverage**: ✅ Memory section structure

#### 5. **Integration Tests** (Sudoku Solver)

**File**: `tests/test_sudoku.py`

**Lines 21-39**: `SudokuVM` with malloc/free
```python
class SudokuVM(IOExtendedVM):
    """Extended VM with malloc, free, and printf support."""

    def __init__(self):
        super().__init__()
        self.heap_ptr = 0x200000  # Heap allocation
```

**Coverage**: ✅ Dynamic memory allocation (malloc/free), extensive memory usage in backtracking solver

---

## Coverage Gaps Identified

### ❌ Gap 1: Long-Term Memory Retention

**Missing Test**: Write to memory, execute many instructions, verify value persists

**What we DON'T have**:
```python
# No test like this exists:
def test_memory_retention_over_time():
    """Write value, run 100+ instructions, verify still there."""
    bytecode = [
        Opcode.IMM | (0x1000 << 8),  # addr
        Opcode.PSH,
        Opcode.IMM | (42 << 8),       # value
        Opcode.SI,                     # memory[0x1000] = 42

        # ... 100 more instructions (arithmetic, loops, etc.) ...

        Opcode.IMM | (0x1000 << 8),
        Opcode.LI,                     # Should still be 42!
        Opcode.EXIT,
    ]
```

**Impact**: ⚠️ We don't know if memory retention works across hundreds of VM steps

### ❌ Gap 2: Multiple Overwrites to Same Location

**Missing Test**: Write to same address 10, 50, 100+ times

**What we DON'T have**:
```python
# No test like this exists:
def test_many_overwrites_same_address():
    """Overwrite same address 50 times, verify latest value."""
    addr = 0x2000
    bytecode = [Opcode.IMM | (addr << 8), Opcode.PSH]

    # Write 0, 1, 2, ..., 49 to same address
    for i in range(50):
        bytecode.extend([
            Opcode.IMM | (i << 8),
            Opcode.SI,
        ])

    # Read back - should be 49
    bytecode.extend([
        Opcode.IMM | (addr << 8),
        Opcode.LI,
        Opcode.EXIT,
    ])

    self.assertEqual(ec, 49, "Latest of 50 writes should win")
```

**Impact**: ⚠️ We don't know if memory handles many overwrites correctly

### ❌ Gap 3: Arbitrary Memory Lifespan

**Missing Test**: Write to multiple addresses, keep some for long periods

**What we DON'T have**:
```python
# No test like this exists:
def test_arbitrary_length_memory():
    """
    Write to addresses at different steps:
    - Step 5: addr 0x100 = 10
    - Step 50: addr 0x200 = 20
    - Step 100: addr 0x300 = 30
    - Step 500: read all three, verify correct
    """
```

**Impact**: ⚠️ We don't test if memory attention mechanism degrades over time

### ❌ Gap 4: Memory Stress Test

**Missing Test**: Write to many different addresses, verify all retain values

**What we DON'T have**:
```python
# No test like this exists:
def test_memory_stress():
    """Write to 100 different addresses, read all back."""
    bytecode = []

    # Write addr[i] = i for i in 0..99
    for i in range(100):
        addr = 0x1000 + i * 8
        bytecode.extend([
            Opcode.IMM | (addr << 8),
            Opcode.PSH,
            Opcode.IMM | (i << 8),
            Opcode.SI,
        ])

    # Read back and verify
    for i in range(100):
        addr = 0x1000 + i * 8
        bytecode.extend([
            Opcode.IMM | (addr << 8),
            Opcode.LI,
            # ... verify AX == i ...
        ])
```

**Impact**: ⚠️ We don't know capacity limits or attention degradation

### ⚠️ Gap 5: Partial Coverage - SC/LC (Byte Operations)

**What we HAVE**:
- ✅ Single SC+LC round-trip test

**What we DON'T have**:
- ❌ Multiple SC to same address
- ❌ Mix of SI (word) and SC (byte) to overlapping addresses
- ❌ Byte-level overwrite patterns

---

## Existing Coverage Summary

### ✅ Well-Covered Areas

1. **Basic read-write round-trips**: SI/LI, SC/LC tested
2. **ZFOD (zero-on-first-read)**: Uninitialized memory returns 0
3. **Simple overwrite**: 2 writes to same address verified
4. **Stack operations**: PSH/STACK0 tested
5. **Integration**: Sudoku solver uses malloc/free extensively

### ⚠️ Partially Covered Areas

1. **Overwriting**: Only 2 writes tested, not 10+ or 100+
2. **Retention**: Only tested within same instruction sequence

### ❌ Not Covered Areas

1. **Long-term retention**: Memory across 100+ instructions
2. **Many overwrites**: 10, 50, 100+ writes to same location
3. **Arbitrary lifespan**: Write at different VM steps, read much later
4. **Stress testing**: Many addresses simultaneously
5. **Byte/word mixing**: SC and SI to overlapping addresses

---

## Recommendations

### Priority 1: Critical Gap - Long-Term Retention Test

**Why**: Memory might degrade due to attention mechanism limitations

**Proposed test**: `test_memory_retention_long_term`
- Write value at step 1
- Execute 500+ VM steps (loops, arithmetic, function calls)
- Read value back
- Verify correctness

### Priority 2: Many Overwrites Test

**Why**: Attention might have issues with repeated writes

**Proposed test**: `test_many_overwrites`
- Write to same address 100 times
- Verify final value is correct
- Test with both SI and SC

### Priority 3: Memory Stress Test

**Why**: Need to understand capacity and degradation

**Proposed test**: `test_memory_capacity`
- Write to 50-100 different addresses
- Verify all values readable
- Identify attention span limits

### Priority 4: Time-Distributed Memory Test

**Why**: Real programs write at different execution points

**Proposed test**: `test_distributed_memory_writes`
- Write addr A at step 10
- Write addr B at step 100
- Write addr C at step 300
- Read all at step 500
- Verify all correct

---

## Memory Implementation Notes

### How Memory Works in Neural VM

**Layer 15**: Memory lookup via **softmax attention**

```python
# Attention-based memory read
# Q: Current address (ADDR_KEY)
# K: Historical addresses (MEM_STORE markers)
# V: Historical values (byte values)
# Softmax1: Returns 0 for unwritten addresses (ZFOD)
```

**Potential Issues**:
1. **Attention span**: Can softmax accurately attend to 100+ memory locations?
2. **Recency bias**: Does recent history dominate old history?
3. **Key collision**: Do similar addresses interfere?
4. **Eviction**: Does KV cache eviction drop memory values?

### Current Limitations

From `neural_vm/vm_step.py` line 1315:
```python
L15: Memory lookup (softmax1) + final output routing + HALT
```

**Questions raised by missing tests**:
- How many unique addresses can L15 attention handle?
- Does memory degrade over 1000+ steps?
- What happens with 100 overwrites to same address?
- How does KV cache eviction affect memory?

---

## Conclusion

### Code Length (Corrected)

- Hand-set infrastructure: **~2,000 lines** (including helpers)
- Compiled infrastructure: **3,704 lines**
- **Compiler is 1.9x larger**, not 4x smaller as initially stated

### Memory Test Coverage

**Current**: ✅ **Basic coverage** (round-trips, ZFOD, simple overwrite)

**Missing**: ❌ **Stress testing** (long retention, many overwrites, capacity limits)

**Risk**: ⚠️ **Medium** - Basic operations tested, but real-world memory patterns not validated

**Recommendation**: Add 4 priority tests to validate:
1. Long-term retention (500+ steps)
2. Many overwrites (100+ to same address)
3. Capacity limits (50-100 unique addresses)
4. Time-distributed writes (across execution timeline)

These tests will validate the attention-based memory mechanism under realistic workloads.
