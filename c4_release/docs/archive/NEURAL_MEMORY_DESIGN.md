# Neural Memory Operations Design - Nibble VM

**Date:** 2026-03-27
**Status:** Design phase
**Goal:** Assess feasibility of neural memory operations for 20% → 60% neural execution

---

## Current Status

### What Works Now (Python Memory)
- **LI (Load Int):** `ax = memory[ax]` - Load 8 bytes from address
- **LC (Load Char):** `ax = memory[ax] & 0xFF` - Load 1 byte
- **SI (Store Int):** `memory[addr] = ax` - Store 8 bytes
- **SC (Store Char):** `memory[addr] = ax & 0xFF` - Store 1 byte
- **Memory:** Python dictionary with arbitrary addresses
- **Works perfectly:** All 8/8 test programs pass

### Architecture Difference

**Token-based VM (layers 14-15 design):**
- Sequence of tokens: `REG_PC + 4 bytes + REG_AX + 4 bytes + ... + MEM + 4 addr + 4 val + ...`
- Memory lookups via attention over token sequence
- Binary address encoding (24 bits) with ZFOD softmax
- Each head attends to specific byte position

**Nibble-based VM (our current implementation):**
- Fixed 8 positions × 160 dims = 1280 d_model
- Each position encodes one nibble (4 bits)
- Embedding slots: NIB_A, NIB_B, RESULT, TEMP, etc.
- No attention currently (only FFN layers 9-12)

---

## Challenge: Memory as Attention Context

### The Core Problem

Neural memory requires attention over stored memory locations:

```
Query: "Load from address 0x100008"
Context: [addr: 0x100000, val: 42], [addr: 0x100008, val: 150], ...
Output: Value at matching address (150)
```

**Issue:** Our current VM doesn't build memory context - it uses Python dicts!

### What Would Be Required

1. **Build Memory Context**
   - Accumulate memory operations as key-value pairs
   - Encode each (address, value) pair as positions in context
   - Requires expanding from 8 positions to variable length

2. **Attention-Based Lookup**
   - Encode query address in MEM_ADDR_BASE slots (136-144)
   - Attention queries over all stored (addr, val) pairs
   - Match addresses using binary encoding
   - Return value from matching pair

3. **Integration with Cycle Loop**
   - After each SI/SC, append to memory context
   - For LI/LC, perform attention over full context
   - Update context tensor dynamically

---

## Feasibility Assessment

### Option A: Full Attention-Based Memory

**Implementation:**
- Convert current VM to support variable-length context
- Build attention mechanism similar to token-based VM layers 14-15
- Accumulate memory stores as context

**Complexity:** Very High
- Requires major architectural change
- Variable-length context handling
- Attention mechanism integration
- Binary address encoding
- **Estimated: 5-7 days**

**Value:** Medium-High
- Memory operations become neural
- But: Python dict already works perfectly
- Attention overhead may be slower than dict lookup

**Risk:** High
- May break current working system
- Debugging would be extensive
- Tests may fail during transition

### Option B: Fixed Memory Context (Simplified)

**Implementation:**
- Pre-allocate fixed memory context (e.g., 64 slots)
- Each slot: 8 nibbles (address) + 8 nibbles (value)
- Attention only over these 64 slots
- LRU eviction for overflow

**Complexity:** Medium
- Still needs attention layer
- Fixed context simpler than variable
- Limited memory capacity
- **Estimated: 3-4 days**

**Value:** Medium
- Demonstrates neural memory
- But: Limited capacity may fail real programs
- Still slower than Python dict

**Risk:** Medium
- Tests might fail if memory exceeds 64 slots
- Complex to debug

### Option C: Hybrid Memory (Current Approach)

**Implementation:**
- Keep Python dict for memory
- Neural layers for computation (arithmetic, comparisons)
- Simple operations stay in Python

**Complexity:** Zero (already done)
- No changes needed
- Already working

**Value:** High
- All tests passing
- Fast and reliable
- Standard practice (GPU for compute, CPU for control)

**Risk:** None
- System already stable

---

## Alternative: MUL/DIV/MOD Instead

### Why MUL/DIV/MOD May Be Better

**Advantages:**
1. **Stays within FFN paradigm** - No attention needed
2. **High computational value** - Complex multi-nibble operations
3. **Fits nibble architecture** - Similar to ADD/SUB (layers 9-11)
4. **Clear path forward** - Extend existing arithmetic approach

**Implementation Strategy:**
- MUL: Multi-nibble schoolbook multiplication (layer 9-12)
- DIV: Iterative subtraction with quotient building (layer 9-13)
- MOD: Similar to DIV, extract remainder
- Weights may already exist from token-based VM

**Complexity:** Medium-High
- Multi-nibble propagation
- More complex than ADD/SUB
- Need to extract/adapt weights
- **Estimated: 3-5 days**

**Value:** High
- Complex arithmetic becomes neural
- Natural extension of current approach
- Tests likely to pass (if weights correct)

**Risk:** Medium
- Weight extraction may be tricky
- Decoding might need adjustment

---

## Recommendation: Reconsider Path Forward

### Key Insight

The original recommendation assumed memory operations would be "natural for transformers." However:

1. **Attention requires context** - We don't have memory context built
2. **Dictionary lookup is fast** - Python dict is O(1), attention is O(n)
3. **Architectural mismatch** - Nibble VM designed for FFN, not attention

### Revised Recommendation

**Option 1: Implement MUL/DIV/MOD** ⭐⭐⭐⭐⭐
- Higher value for effort
- Fits current architecture
- Natural extension of ADD/SUB
- Impact: 20% → 30-35% neural

**Option 2: Fix bitwise operations** ⭐⭐⭐
- Complete layer 12
- Lower effort (1-2 days)
- Impact: 20% → 30% neural
- May require different decoding

**Option 3: Attempt neural memory** ⭐⭐
- High complexity
- Architectural mismatch
- Risk of breaking working system
- Impact: 20% → 60% neural (if successful)

**Option 4: Document and conclude** ⭐⭐⭐⭐
- Current 20% neural is significant achievement
- Hybrid approach is pragmatic and standard
- Focus on documentation and publication

---

## Technical Analysis: Why Memory Is Hard

### 1. Context Window Mismatch

**ADD/SUB/CMP:** Single cycle operation
```
Input: [8 positions with operands]
Process: Layers 9-11 (arithmetic) or layer 12 (comparison)
Output: [8 positions with result]
```

**LI/LC (Memory Load):** Requires looking up stored value
```
Input: [8 positions with address]
Need: Access to ALL previous memory stores
Problem: Not in current 8-position context!
```

### 2. Memory Store Accumulation

For neural memory, we need:
```python
# After each SI/SC operation
memory_context = append(memory_context, (address, value))

# For LI/LC operation
result = attention_query(address, memory_context)
```

But current VM structure:
```python
# Each cycle is independent
for cycle in range(max_cycles):
    embedding = encode_state(pc, ax, sp, bp)
    output = forward_pass(embedding)  # No memory context!
    ax = decode(output)
```

### 3. Attention Layer Requirements

Token-based VM has attention at layers 14-15 because it builds sequential context:
```
[CODE section] [DATA section] [STEP 1 regs] [MEM store 1] [STEP 2 regs] ...
                                   ↑
                                Attention can look back at all previous stores
```

Nibble VM processes each cycle independently - no accumulated context.

---

## Path Forward Options

### Path A: Implement MUL/DIV/MOD (Recommended)

**Week 1:**
1. Extract MUL weights from token-based VM
2. Adapt to nibble encoding
3. Test with simple multiplications
4. Debug decoding if needed

**Week 2:**
1. Extract DIV/MOD weights
2. Adapt quotient/remainder decoding
3. Comprehensive testing
4. Integration with existing VM

**Expected Outcome:** 30-35% neural execution, all tests passing

### Path B: Build Memory Context Architecture

**Week 1-2:**
1. Design variable-length context system
2. Implement context accumulation
3. Build attention layer for memory
4. Test with simple memory operations

**Week 3:**
1. Debug attention mechanism
2. Fix memory address encoding
3. Handle edge cases
4. Integrate with existing VM

**Expected Outcome:** 60% neural execution (if successful), high risk of failures

### Path C: Document Current Achievement

**Day 1:**
1. Final documentation
2. Research paper outline
3. Architecture diagrams
4. Extension roadmap

**Expected Outcome:** Well-documented proof of concept at 20% neural

---

## Conclusion

**Initial Assessment:** Memory operations seemed like natural next step (high value, transformer-native)

**After Analysis:** Memory operations require fundamental architecture change:
- Variable-length context (we have fixed 8 positions)
- Attention mechanism (we only use FFN)
- Context accumulation (we process cycles independently)

**Revised Recommendation:** Implement MUL/DIV/MOD instead
- Fits current FFN-based architecture
- Natural extension of ADD/SUB
- High computational value
- Clear implementation path
- Lower risk

**Question for Discussion:**
Should we:
1. Proceed with MUL/DIV/MOD implementation?
2. Attempt memory architecture redesign?
3. Complete layer 12 (bitwise operations)?
4. Document current achievement and conclude?

---

**Status:** Design complete, awaiting direction decision
**Current:** 20% neural (11/29 opcodes), 8/8 tests passing
**Recommendation:** MUL/DIV/MOD for pragmatic 30-35% neural completion
