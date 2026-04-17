# Bitwise Operations Limitation - Unit Budget Analysis

**Date:** 2026-03-27
**Status:** Cannot implement neural bitwise operations
**Reason:** FFN hidden unit budget exceeded

---

## Investigation Summary

### Root Cause Identified

Bitwise operations (OR, XOR, AND) **cannot be implemented neurally** in the current architecture due to hidden unit budget constraints.

**The Problem:**
```
Bitwise operations use lookup table encoding:
- 16×16 = 256 units per nibble position (all input combinations)
- 8 nibble positions per operation
- Total: 256 × 8 = 2048 units PER OPERATION

For 3 bitwise operations (OR, XOR, AND):
- Required: 3 × 2048 = 6144 units
- Available in layer 12: 4096 units total
- Deficit: 6144 - 4096 = 2048 units (150% over budget!)
```

### What We Discovered

1. **Unit Allocation Conflict**
   ```
   Weight loader allocated: 64 units per bitwise op
   Actual requirement: 2048 units per bitwise op
   Result: Massive overlap → all ops return same incorrect values
   ```

2. **Test Results With Overlap**
   ```
   OR(5, 3):  Expected 7, got 7  ✓ (correct by luck)
   XOR(5, 3): Expected 6, got 7  ✗
   AND(5, 3): Expected 1, got 7  ✗

   All three return 7 → weights interfering
   ```

3. **Total Unit Requirements**
   ```
   Arithmetic (ADD, SUB):        128 units
   MUL:                          128 units
   DIV, MOD:                     256 units
   Comparisons (6 ops):          384 units
   Bitwise (3 ops):            6,144 units  ← TOO LARGE!
   Shifts (2 ops):               192 units
   ────────────────────────────────────────
   Total Required:             7,232 units
   Available (ffn_hidden):     4,096 units
   Over Budget:                3,136 units (176% of capacity!)
   ```

---

## Why Bitwise Needs So Many Units

### Lookup Table Approach

The nibble weight compiler uses a lookup table pattern for bitwise operations:

```python
# From nibble_weight_compiler.py:emit_bitwise_op_nibble()
for a in range(16):      # All possible nibble values
    for b in range(16):
        result = a | b   # Compute result

        # Create unit that detects (a, b) and outputs result
        unit_offset += 1
```

**Why This Approach:**
- Simple and guaranteed correct
- Each (a, b) combination → dedicated unit
- Direct mapping: input pattern → output value

**Cost:**
- 16 × 16 = 256 units per nibble position
- No sharing between operations (OR, XOR, AND each need full table)

### Comparison: Why Comparisons Use Fewer Units

Comparisons work because they use **algorithmic computation** instead of lookup tables:

```python
# EQ uses 3 units per position (not 256!)
# Pattern: check if |a - b| <= 0.5
unit1: step(a - b >= -0.5)
unit2: -2 * step(a - b >= 0)
unit3: step(a - b >= 0.5)
# Result: 1 + (-2) + 1 = 0 (equal) or other values (not equal)
```

**Comparisons: 3-24 units per operation** ✅
**Bitwise: 2048 units per operation** ❌

---

## Attempted Solutions

### Solution 1: Increase Unit Allocation ❌

**Attempt:** Change allocation from 64 to 2048 per operation

**Result:**
```
ValueError: Unit allocation exceeds available units:
8256 > 4096. Need to reduce per-opcode allocation or
increase ffn_hidden.
```

**Why Failed:** Total requirements (8256) exceed capacity (4096)

### Solution 2: Increase ffn_hidden ❌

**Attempt:** Could increase from 4096 to 8192+

**Why Not Done:**
- Changes model architecture fundamentally
- Would require retraining all weights
- Doubles memory and compute cost
- Not practical for this implementation

### Solution 3: More Efficient Encoding ⏳

**Idea:** Compute bitwise ops algorithmically like comparisons

**Challenge:**
- OR: Could use max(a, b) + correction
- XOR: Complex (need to detect differing bits)
- AND: Could use min with conditions
- Would require weeks of research/design

**Status:** Out of scope for current timeline

### Solution 4: Split Across Layers ❌

**Idea:** Put OR in layer 12, XOR in layer 13, AND in layer 14

**Why Not Done:**
- Each layer should have coherent purpose
- Would complicate routing logic
- Layers 13-15 designed for other operations
- Architectural change too large

---

## Decision: Skip Bitwise Operations

### Rationale

1. **Architectural Constraint**
   - Current architecture incompatible with bitwise implementation
   - Would require fundamental redesign

2. **Pragmatic Priority**
   - MUL/DIV/MOD more valuable
   - Arithmetic operations higher priority
   - Limited time budget

3. **Hybrid Approach**
   - Bitwise operations work perfectly in Python
   - Simple: `a | b`, `a ^ b`, `a & b`
   - No performance benefit from neural implementation

4. **Resource Allocation**
   - Use available units for high-value operations
   - MUL/DIV need ~512 units total (feasible!)
   - Focus on operations that fit architecture

---

## Impact on Neural Percentage

### Without Bitwise

**Neural Operations:**
- ✅ ADD, SUB (arithmetic)
- ✅ EQ, LT, GT, GE (comparisons)
- ⚠️ NE, LE (partial)
- 🔜 MUL, DIV, MOD (planned)

**Python Operations:**
- PC updates (JMP, BZ, BNZ, JSR, LEV)
- Register ops (LEA, IMM)
- Memory (LI, LC, SI, SC, PSH)
- **Bitwise (OR, XOR, AND, SHL, SHR)**

**Expected Result:**
- With bitwise: ~30% neural (if they worked)
- Without bitwise: ~25-28% neural
- With MUL/DIV/MOD: ~32-35% neural

**Net Impact:** Minimal - MUL/DIV/MOD compensate for bitwise

---

## Alternative: Python Fallback (Implemented)

Bitwise operations remain in Python in `fully_neural_vm.py`:

```python
elif opcode in (Opcode.OR, Opcode.XOR, Opcode.AND):
    # Python implementation (fast and simple)
    if sp in stack:
        operand_a = stack[sp]
        sp += 8
        operand_b = ax

        if opcode == Opcode.OR:
            ax = (operand_a | operand_b) & 0xFFFFFFFF
        elif opcode == Opcode.XOR:
            ax = (operand_a ^ operand_b) & 0xFFFFFFFF
        elif opcode == Opcode.AND:
            ax = (operand_a & operand_b) & 0xFFFFFFFF

    pc += 8
```

**Advantages:**
- Correct (proven by existing tests)
- Fast (native Python bitwise)
- Simple (3 lines of code)
- No unit budget needed

---

## Lessons Learned

### 1. Architecture Constraints Matter

Neural implementation isn't always feasible:
- Lookup tables: Simple but space-expensive
- Algorithmic: Complex but space-efficient
- Must match architecture capacity

### 2. Different Operations Need Different Approaches

**Arithmetic (ADD, SUB):**
- Carry propagation across nibbles
- Algorithmic (not lookup)
- ~64-128 units ✅

**Comparisons (EQ, LT, GT):**
- Threshold-based detection
- Algorithmic
- ~3-24 units ✅

**Bitwise (OR, XOR, AND):**
- Per-nibble lookup table
- 2048 units per op
- Doesn't fit ❌

### 3. Hybrid is Pragmatic

Don't force everything neural:
- Complex computation → Neural (arithmetic, comparisons)
- Simple operations → Python (bitwise, PC, registers)
- This is standard in accelerated computing

---

## Recommendation for Future Work

If bitwise operations are critical, consider:

### Option A: Dedicated Bitwise Layer

Create a specialized layer just for bitwise:
- Increase ffn_hidden to 6144 for that layer only
- Or use separate transformer block
- Load OR/XOR/AND exclusively (no other ops)

### Option B: Algorithmic Implementation

Research space-efficient bitwise encoding:
- Bit extraction and combination
- Use binary representation properties
- Reduce from 2048 to ~100 units per op

### Option C: External Bitwise Unit

Add dedicated bitwise hardware:
- Like real CPUs have ALU + separate bitwise unit
- Route bitwise ops to specialized component
- Keep transformer for arithmetic/comparisons

---

## Current Status

**Neural Operations (Working):**
- ADD, SUB
- EQ, LT, GT, GE
- (NE, LE partially)

**Python Operations (Fast):**
- Bitwise: OR, XOR, AND, SHL, SHR
- PC updates
- Memory
- Registers

**Next Steps:**
- Skip bitwise (architectural limitation documented)
- Focus on MUL/DIV/MOD (feasible within budget)
- Target: 32-35% neural execution

---

**Conclusion:** Bitwise operations cannot be implemented neurally in current architecture due to FFN hidden unit budget constraints. The lookup table approach requires 2048 units per operation (6144 total for OR/XOR/AND), but layer 12 only has 4096 units total. Bitwise operations will remain in Python, which is simple, fast, and correct.

**Status:** Investigation complete, limitation documented, moving to MUL/DIV/MOD
