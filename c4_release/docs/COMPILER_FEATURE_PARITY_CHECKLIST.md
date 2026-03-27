# Graph Weight Compiler - Feature Parity Checklist

## Goal
Achieve feature parity with manual weight setting for all Neural VM operations.

## Status Legend
- ✅ **DONE**: Implemented and tested
- 🔧 **IN PROGRESS**: Partially implemented
- ⏳ **TODO**: Not started
- ❌ **BLOCKED**: Waiting on dependencies

---

## Phase 1: Core Architecture (DONE)
- ✅ Computation graph (DAG) construction
- ✅ Register allocation via graph coloring
- ✅ Weight emission to FFN matrices
- ✅ PureFFN integration (correct W_up/W_gate/W_down pattern)
- ✅ Sparsity preservation (>99% sparse like manual setting)

---

## Phase 2: Basic Arithmetic (2/5)

### ✅ ADD - `out = a + b`
- Implementation: `emit_add()`
- Pattern: W_up=±S (activation), W_gate=a+b (computation), W_down=1/S
- Units: 2
- **Status**: VERIFIED WORKING (5 + 10 = 15.0 ✓)

### ✅ SUB - `out = a - b`
- Implementation: `emit_sub()`
- Pattern: Same as ADD but W_gate=a-b
- Units: 2
- **Status**: IMPLEMENTED (needs test)

### ⏳ MUL - `out = a * b` (one-hot)
- Implementation: `emit_mul()`
- Pattern: Lookup table for all (i,j) where i*j=k mod base
- Units: 2*base (32 for base=16)
- **Dependencies**: One-hot encoding support
- **Status**: TODO

### ⏳ DIV - `out = a / b` (one-hot)
- Implementation: `emit_div()`
- Pattern: Lookup table for floor(a/b)
- Units: 2*base
- **Dependencies**: One-hot encoding support
- **Status**: TODO

### ⏳ MOD - `out = a mod b`
- Implementation: `emit_mod()`
- Pattern: Raw sum + conditional base subtraction
- Units: 6 (scalar), 2*base (one-hot)
- **Status**: TODO

---

## Phase 3: Comparison Operations (1/6)

### 🔧 CMP_EQ - `out = (a == b)`
- Implementation: `emit_cmp_eq()`
- Pattern: (a>=b) + (b>=a) - (a>b) - (b>a)
- Units: 4
- **Status**: IMPLEMENTED (needs pattern fix + test)

### 🔧 CMP_LT - `out = (a < b)`
- Implementation: `emit_cmp_lt()`
- Pattern: step(b-a >= 1)
- Units: 2
- **Status**: IMPLEMENTED (needs pattern fix + test)

### 🔧 CMP_GE - `out = (a >= b)`
- Implementation: `emit_cmp_ge()`
- Pattern: step(a-b >= 0)
- Units: 2
- **Status**: IMPLEMENTED (needs pattern fix + test)

### ⏳ CMP_LE - `out = (a <= b)`
- Implementation: `emit_cmp_le()`
- Pattern: step(b-a >= 0)
- Units: 2
- **Status**: TODO

### ⏳ CMP_GT - `out = (a > b)`
- Implementation: `emit_cmp_gt()`
- Pattern: step(a-b >= 1)
- Units: 2
- **Status**: TODO

### ⏳ CMP_NE - `out = (a != b)`
- Implementation: `emit_cmp_ne()`
- Pattern: NOT(CMP_EQ)
- Units: 2 (if fused), 6 (if composed)
- **Status**: TODO

---

## Phase 4: Logical Operations (0/4)

### 🔧 AND - `out = a && b`
- Implementation: `emit_logical_and()`
- Pattern: step(a+b >= 2)
- Units: 2
- **Status**: IMPLEMENTED (needs pattern fix + test)

### 🔧 OR - `out = a || b`
- Implementation: `emit_logical_or()`
- Pattern: step(a+b >= 1)
- Units: 2
- **Status**: IMPLEMENTED (needs pattern fix + test)

### 🔧 NOT - `out = !a`
- Implementation: `emit_logical_not()`
- Pattern: 1 - a
- Units: 2
- **Status**: IMPLEMENTED (needs pattern fix + test)

### ⏳ XOR - `out = a ^ b`
- Implementation: `emit_logical_xor()`
- Pattern: (a+b) - 2*(a&&b)
- Units: 4
- **Status**: TODO

---

## Phase 5: Bitwise Operations (0/5)

### ⏳ BIT_AND - `out = a & b` (one-hot)
- Implementation: `emit_bit_and()`
- Pattern: Lookup table for bitwise AND
- Units: 2*N (N=width)
- **Dependencies**: One-hot encoding
- **Status**: TODO

### ⏳ BIT_OR - `out = a | b` (one-hot)
- Implementation: `emit_bit_or()`
- Pattern: Lookup table for bitwise OR
- Units: 2*N
- **Dependencies**: One-hot encoding
- **Status**: TODO

### ⏳ BIT_XOR - `out = a ^ b` (one-hot)
- Implementation: `emit_bit_xor()`
- Pattern: Lookup table for bitwise XOR
- Units: 2*N
- **Dependencies**: One-hot encoding
- **Status**: TODO

### ⏳ SHL - `out = a << n` (one-hot)
- Implementation: `emit_shl()`
- Pattern: Lookup table for shifts
- Units: 2*N (constant shift), 2*N² (variable)
- **Dependencies**: One-hot encoding
- **Status**: TODO

### ⏳ SHR - `out = a >> n` (one-hot)
- Implementation: `emit_shr()`
- Pattern: Lookup table for shifts
- Units: 2*N (constant shift), 2*N² (variable)
- **Dependencies**: One-hot encoding
- **Status**: TODO

---

## Phase 6: Register Operations (1/4)

### ✅ CONST - `out = constant`
- Implementation: `emit_const()`
- Pattern: Bias-based (b_up = S*value)
- Units: 2
- **Status**: IMPLEMENTED (works with ADD test)

### 🔧 MOVE - `out = in` (relay)
- Implementation: `emit_move()`
- Pattern: Cancel pair relay
- Units: 2 (scalar), 2*width (vector)
- **Status**: IMPLEMENTED (needs pattern fix + test)

### 🔧 CLEAR - `reg = 0`
- Implementation: `emit_clear()`
- Pattern: Subtract from self
- Units: 2 (scalar), 2*width (vector)
- **Status**: IMPLEMENTED (needs pattern fix + test)

### ⏳ SWAP - `(a, b) = (b, a)`
- Implementation: Compose 3 MOVE operations
- Pattern: temp=a; a=b; b=temp
- Units: 6
- **Status**: TODO

---

## Phase 7: Conditional Operations (0/2)

### ⏳ SELECT - `out = cond ? a : b` (MUX)
- Implementation: `emit_select()`
- Pattern: cond*a + (1-cond)*b
- Units: 4-6
- **Dependencies**: NOT for (1-cond)
- **Status**: TODO

### ⏳ IF_THEN - `if cond: out = a`
- Implementation: `emit_if_then()`
- Pattern: Gated relay
- Units: 2
- **Status**: TODO

---

## Phase 8: Memory Operations (0/2)

### ⏳ LOOKUP - `out = memory[addr]`
- Implementation: **ATTENTION-BASED** (not FFN)
- Pattern: Attention query on address → retrieve value
- Units: 0 FFN units (uses attention layer)
- **Note**: Separate attention layer, not WeightEmitter
- **Status**: TODO

### ⏳ STORE - `memory[addr] = value`
- Implementation: **ATTENTION-BASED** (not FFN)
- Pattern: Add KV pair to cache
- Units: 0 FFN units (uses attention mechanism)
- **Note**: Token emission, not weights
- **Status**: TODO

---

## Phase 9: Pattern Fixes (CRITICAL)

All implemented operations need pattern fixes to match existing codebase:

### 🔧 Fix Pattern for All Ops
Current incorrect pattern:
```python
W_up[unit, input_reg] = S  # ❌ WRONG: silu zeros negatives
```

Correct pattern (from neural_vm/alu/ops/sub.py:48-57):
```python
# Activation via bias or opcode:
b_up[unit] = S  # or W_up[unit, opcode_reg] = S

# Computation in gate (linear, not silu'd):
W_gate[unit, a_reg] = 1.0
W_gate[unit, b_reg] = ±1.0

# Scale back:
W_down[out_reg, unit] = 1.0 / S
```

**Operations needing fix**:
- 🔧 CMP_EQ, CMP_LT, CMP_GE
- 🔧 AND, OR, NOT
- 🔧 MOVE, CLEAR

---

## Phase 10: Testing & Validation (0/4)

### ⏳ Unit Tests
- Test each primitive independently
- Verify numerical correctness
- Check against expected patterns
- **Status**: TODO

### ⏳ Integration Tests
- Test multi-operation graphs
- Verify register allocation efficiency
- Check weight sparsity
- **Status**: TODO

### ⏳ VM Test Suite
- Run on 1000+ existing VM tests
- Compare output with manual weights
- Measure match rate
- **Status**: TODO

### ⏳ Performance Benchmarks
- Compilation time
- Memory usage
- Runtime performance vs manual
- **Status**: TODO

---

## Phase 11: Advanced Features (0/5)

### ⏳ One-Hot Encoding Support
- Dimension layout for one-hot values
- Lookup table weight patterns
- Register allocation for vectors
- **Status**: TODO

### ⏳ Multi-Layer Operations
- Operations spanning multiple FFN layers
- Layer assignment algorithm
- Inter-layer dependencies
- **Status**: TODO

### ⏳ Optimization Passes
- Dead code elimination
- Constant folding
- Common subexpression elimination
- Strength reduction
- **Status**: TODO

### ⏳ High-Level API
- Composite operations (nibble_add, byte_mul, etc.)
- Control flow constructs
- Function/macro support
- **Status**: TODO

### ⏳ Code Generation
- Direct integration with ALU ops
- Automatic weight baking
- Replace manual setting entirely
- **Status**: TODO

---

## Implementation Priority

### **Sprint 1: Core Fixes (1-2 days)**
1. ✅ Fix ADD pattern (DONE)
2. ✅ Fix SUB pattern (DONE)
3. Fix CMP_GE, CMP_LT, CMP_EQ patterns
4. Fix AND, OR, NOT patterns
5. Fix MOVE, CLEAR patterns
6. Write unit tests for fixed operations

### **Sprint 2: Complete Comparisons & Logical (1 day)**
1. Implement CMP_LE, CMP_GT, CMP_NE
2. Implement XOR
3. Test all comparison operations
4. Test all logical operations

### **Sprint 3: Conditionals & Advanced Arithmetic (2 days)**
1. Implement SELECT, IF_THEN
2. Implement MOD
3. Implement SWAP
4. Test conditionals

### **Sprint 4: One-Hot Operations (2-3 days)**
1. Design one-hot encoding layout
2. Implement MUL, DIV (lookup tables)
3. Implement bitwise ops (BIT_AND, BIT_OR, BIT_XOR)
4. Implement shift ops (SHL, SHR)
5. Test one-hot operations

### **Sprint 5: Memory & Attention (2-3 days)**
1. Design attention-based LOOKUP
2. Design attention-based STORE
3. Implement memory operations
4. Test memory operations

### **Sprint 6: Integration & Validation (2-3 days)**
1. Run on VM test suite (1000+ tests)
2. Measure match rate vs manual weights
3. Performance benchmarks
4. Bug fixes and optimization

---

## Success Metrics

### **Minimum Viable Product (MVP)**
- ✅ Core architecture working
- ✅ ADD operation verified
- ⏳ All basic arithmetic (ADD, SUB, MUL, DIV, MOD)
- ⏳ All comparisons (6 operations)
- ⏳ All logical (4 operations)
- ⏳ All register ops (4 operations)
- ⏳ 90% match rate on VM test suite

### **Feature Complete**
- All 26 FFN-based primitives implemented
- 2 attention-based primitives (LOOKUP, STORE)
- 100% match rate on VM test suite
- Performance within 10% of manual setting

### **Production Ready**
- All tests passing
- Documentation complete
- Integration with existing ALU ops
- Code generation working
- Optimization passes enabled

---

## Current Status Summary

**Completed**: 2/28 primitives (7%)
- ✅ ADD
- ✅ SUB (needs test)

**In Progress**: 8/28 primitives (29%)
- 🔧 CMP_EQ, CMP_LT, CMP_GE
- 🔧 AND, OR, NOT
- 🔧 MOVE, CLEAR

**TODO**: 18/28 primitives (64%)
- Remaining comparisons, conditionals, bitwise, memory ops

**Estimated Time to MVP**: 5-7 days
**Estimated Time to Feature Complete**: 10-14 days
