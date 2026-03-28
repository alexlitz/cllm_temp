# Fully Neural Autoregressive VM - Design Document

## Current Status (Hybrid Autoregressive)

✅ **Working:**
- Autoregressive execution framework
- Neural arithmetic (ADD, SUB) via layers 9-11
- 7/7 test programs passing
- Python control flow (PC updates, branching)
- Python memory operations

## Goal: Fully Neural Execution

All operations (fetch, decode, execute, memory, control flow) happen through transformer layers. NO Python arithmetic or control flow in forward passes.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│ Input: VM State Embedding [1, 1, 1280]                  │
│                                                           │
│   Current PC, AX, SP, BP (as nibbles)                   │
│   + Bytecode memory (in KV cache)                        │
│   + Stack/data memory (in KV cache)                      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Layers 0-8: Instruction Fetch & Decode                  │
│                                                           │
│  • Attend to bytecode memory at current PC              │
│  • Extract opcode and immediate value                    │
│  • Encode instruction info in embedding                  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Layers 9-11: Arithmetic ✅ WORKING                       │
│                                                           │
│  • ADD: 3-layer carry propagation                       │
│  • SUB: 3-layer borrow propagation                      │
│  • Result in RESULT nibbles                              │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Layer 12: Comparisons & Bitwise                          │
│                                                           │
│  • EQ, NE, LT, GT, LE, GE                               │
│  • OR, XOR, AND, SHL, SHR                                │
│  • Weights loaded ✅                                     │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Layer 13: Control Flow & PC Update                       │
│                                                           │
│  • Compute next PC based on opcode:                      │
│    - Most ops: PC += 8                                   │
│    - JMP: PC = immediate                                 │
│    - BZ: PC = immediate if AX==0, else PC += 8          │
│    - BNZ: PC = immediate if AX!=0, else PC += 8         │
│  • Weights loaded ✅ (JMP, BZ, BNZ, ADJ)                │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Layers 14-15: Memory Operations                          │
│                                                           │
│  • Attention-based memory lookup                         │
│  • LI, LC: Attend to memory at address AX               │
│  • SI, SC: Write to memory                               │
│  • PSH: Write to stack, update SP                        │
│  • Weights loaded ✅                                     │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Output: Next VM State Embedding [1, 1, 1280]            │
│                                                           │
│   Next PC, AX, SP, BP (as nibbles)                      │
│   Updated memory state (written to KV cache)             │
└─────────────────────────────────────────────────────────┘
                          ↓
                   Feed back as input
                   (Autoregressive loop)
```

## Key Components to Implement

### 1. Extended Nibble Embedding

**Current:** 8 nibbles for operands (NIB_A, NIB_B) + opcode

**Needed:**
```python
class ExtendedNibbleEmbedding:
    # Current nibbles (positions 0-7)
    NIB_A[0-7]  # Operand A (AX or result)
    NIB_B[0-7]  # Operand B (stack top)

    # NEW: PC nibbles (use TEMP or dedicated slots)
    PC[0-7]     # Current PC (8 nibbles = 32 bits)

    # NEW: Immediate value nibbles
    IMM[0-7]    # Immediate value from instruction

    # NEW: Next PC nibbles (output)
    NEXT_PC[0-7]  # Computed by layer 13

    # Opcode (already have)
    OPCODE      # One-hot encoding
```

### 2. Instruction Fetch (Layers 0-8)

**Challenge:** Need to attend to bytecode memory at current PC

**Options:**

**Option A: Python fetch with neural decode**
- Python: Fetch instruction from code[PC//8]
- Neural: Decode opcode and immediate in layers 0-8
- Hybrid approach (simpler)

**Option B: Full neural fetch**
- Embed entire bytecode in KV cache
- Attention: Query = current PC, Keys = bytecode addresses
- Extract instruction via attention weights
- More complex but fully neural

**Recommendation:** Start with Option A (Python fetch, neural decode)

### 3. Neural PC Update (Layer 13)

Layer 13 needs to implement:
```
if opcode == JMP:
    next_PC = immediate
elif opcode == BZ:
    next_PC = immediate if AX == 0 else PC + 8
elif opcode == BNZ:
    next_PC = immediate if AX != 0 else PC + 8
elif opcode == JSR:
    next_PC = immediate  (also push PC to stack)
elif opcode == LEV:
    next_PC = pop from stack
else:
    next_PC = PC + 8
```

**Neural Implementation:**
- Use SwiGLU step functions to implement conditionals
- W_gate: Activate based on opcode one-hot
- W_up: Compute conditions (AX == 0, etc.)
- W_down: Select between PC+8 or immediate

### 4. Memory Operations (Layers 14-15)

**Current:** Python dict for stack/memory

**Needed:** Attention-based memory
- KV cache stores (address, value) pairs
- Query = address in AX
- Attention retrieves value at address
- Write: Append (address, value) to KV cache

**Challenge:** Need to maintain memory consistency
- Latest write to an address should be retrieved
- Use ALiBi bias for recency

## Implementation Plan

### Phase 1: Neural PC Updates ⏳ IN PROGRESS
1. ✅ Extend NibbleVMEmbedding to encode PC and immediate
2. ⏳ Layer 13 computes next_PC
3. ⏳ Decode next_PC from output
4. ⏳ Test on programs with branches (if/while)

**Estimated:** 2-3 days

### Phase 2: Neural Instruction Fetch
1. Hybrid approach: Python fetch, neural decode
2. Layers 0-8 extract opcode and immediate
3. Pass to execution layers

**Estimated:** 2-3 days

### Phase 3: Neural Memory
1. Attention-based memory lookup
2. LI/LC: Attend to memory at address AX
3. SI/SC: Write to memory (append to KV cache)
4. PSH/ENT/LEV: Stack operations

**Estimated:** 3-4 days

### Phase 4: Full Integration
1. Connect all layers in single forward pass
2. State propagation (output[t] → input[t+1])
3. Run full test suite

**Estimated:** 2-3 days

**Total:** ~9-13 days

## What We Have Now

✅ **Fully working hybrid autoregressive VM:**
- Framework for state propagation
- Neural arithmetic (ADD, SUB) working
- 7/7 test programs passing
- Clear architecture for extension

✅ **All weights loaded:**
- 29/29 opcodes have weights in correct layers
- Can access these weights for neural execution

✅ **Clear path forward:**
- Each phase builds on previous
- Can test incrementally
- Maintain backward compatibility

## Testing Strategy

For each phase, test programs that exercise the new feature:

### Phase 1 Tests (Neural PC):
```c
int main() { if (1) return 42; else return 0; }  // Test BZ
int main() { if (0) return 0; else return 42; }  // Test BZ false
int main() { int i = 5; while (i > 0) i--; return i; }  // Test BNZ
```

### Phase 2 Tests (Instruction Fetch):
- Same as Phase 1, but fetch instruction neurally
- Verify opcode/immediate extracted correctly

### Phase 3 Tests (Memory):
```c
int main() { int x = 42; return x; }  // Test LI
int main() { int x[3]; x[0] = 1; x[1] = 2; return x[0] + x[1]; }  // Test SI/LI
```

### Phase 4 Tests (Full Integration):
- Run existing 1000+ test suite
- Compare against FastLogicalVM (reference)

## Key Insights

1. **Nibble-based is simpler** than token-based
   - Fixed-size embedding (1280 dims)
   - No variable-length sequences per step
   - Direct register encoding

2. **Hybrid approach is pragmatic**
   - Validate each component works
   - Incremental neural migration
   - Python fallback during development

3. **Weights are ready**
   - All 29 opcodes loaded
   - Just need to connect them
   - Extraction logic proven (arithmetic works)

4. **Architecture is modular**
   - Each layer has clear responsibility
   - Can test layers independently
   - Easy to debug failures

## Success Criteria

**Minimum (Phase 1):**
- [ ] Neural PC updates working
- [ ] Programs with if/else execute correctly
- [ ] Programs with while loops execute correctly

**Stretch (Phase 4):**
- [ ] Full neural execution (all layers)
- [ ] No Python control flow
- [ ] Pass 100+ programs from test suite

## Current Achievement

We have successfully demonstrated:
- ✅ 3-layer arithmetic with carry/borrow propagation
- ✅ Nibble-based encoding/decoding
- ✅ Autoregressive execution framework
- ✅ Integration with C4 compiler
- ✅ 7/7 test programs passing (100%)

This is ~40% of the full neural VM. The remaining 60% is primarily integration work (wiring up layers), not new algorithmic challenges.

---

**Date:** 2026-03-27
**Status:** Phase 1 (Neural PC) in progress
**Next Milestone:** Programs with branching executing neurally
