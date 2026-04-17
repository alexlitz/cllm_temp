# Visual Explanation: Multi-Operation vs Multi-Layer

## Current State: Single Operation ✅

```
┌─────────────────────────────────────┐
│   Single FFN Layer (Layer 9)       │
│                                     │
│   Input: pop, AX                    │
│     │                               │
│     ├──[emit_add_nibble]──> Result │
│     │                               │
│   Output: AX                        │
└─────────────────────────────────────┘

Status: ✅ Works perfectly
Opcodes: ADD, SUB, MUL, DIV, MOD, EQ, NE, LT, GT, LE, GE,
         OR, XOR, AND, SHL, SHR, LEA, IMM
Count: 18 opcodes
```

## Problem 1: Multi-Operation (SAME LAYER) ❌

### Example: BZ (Branch if Zero)

```
What we need (ALL IN ONE FFN LAYER):

┌──────────────────────────────────────────────────────────┐
│          Single FFN Layer (Layer 9)                      │
│                                                          │
│  Step 1: Check if AX == 0                               │
│    AX ──┬──[emit_cmp_eq]──> TEMP[0] (cond=0 or 1)      │
│    0 ───┘                                                │
│                                                          │
│  Step 2: Calculate fallthrough                          │
│    PC ──┬──[emit_add]──> TEMP2[0-7] (PC+8)             │
│    8 ───┘                                                │
│                                                          │
│  Step 3: Select target or fallthrough                   │
│    TEMP[0] (cond) ──┬                                   │
│    target ──────────├──[emit_pc_conditional]──> PC     │
│    TEMP2[0-7] ──────┘                                   │
│                                                          │
└──────────────────────────────────────────────────────────┘

Status: ❌ Not implemented yet
Needs: MultiOperationCompiler
Opcodes: JMP, BZ, BNZ, ADJ, MALC, FREE
Count: +6 opcodes
```

**Key Point**: All operations happen in ONE forward pass, using intermediate TEMP slots to pass data.

## Problem 2: Multi-Layer (DIFFERENT LAYERS) ❌

### Example: LI (Load Indirect - AX = *AX)

```
What we need (ACROSS THREE LAYERS):

┌──────────────────────────────────────────┐
│   Layer 9: FFN Setup                     │
│                                          │
│   AX[0-7] ──[emit_move]──> MEM_ADDR     │
│             [emit_flag_set]──> MEM_READ  │
└──────────────────────────────────────────┘
                    │
                    ↓
┌──────────────────────────────────────────┐
│   Layer 10: Attention (Memory Lookup)    │
│                                          │
│   MEM_ADDR ──[Q·K^T lookup]──> MEM_DATA │
│   (Uses KV cache to find memory value)  │
└──────────────────────────────────────────┘
                    │
                    ↓
┌──────────────────────────────────────────┐
│   Layer 11: FFN Copy Result              │
│                                          │
│   MEM_DATA[0-7] ──[emit_move]──> AX     │
└──────────────────────────────────────────┘

Status: ❌ Not implemented yet
Needs: MultiLayerWeightGenerator
Opcodes: LI, LC, SI, SC, PSH, JSR, ENT, LEV
Count: +8 opcodes
```

**Key Point**: Operations span MULTIPLE transformer layers, with attention in between.

## Side-by-Side Comparison

```
┌────────────────────┬─────────────────────┬──────────────────────┐
│ Single-Operation   │ Multi-Operation     │ Multi-Layer          │
├────────────────────┼─────────────────────┼──────────────────────┤
│ ONE operation      │ MULTIPLE operations │ MULTIPLE layers      │
│ ONE FFN layer      │ ONE FFN layer       │ MULTIPLE layers      │
│ Direct compilation │ Sequential in layer │ Across layers        │
│                    │                     │                      │
│ Example: ADD       │ Example: BZ         │ Example: LI          │
│   pop ──┐          │   AX ──[CMP]──┐     │ L9: AX→MEM_ADDR      │
│   AX ──┬├[ADD]─>AX │   0 ──────────┘     │ L10: [Attention]     │
│        ││          │        │             │ L11: MEM_DATA→AX     │
│        │└──────────┤   PC ──[ADD]──┐     │                      │
│        │           │   8 ──────────┘     │                      │
│        │           │        │             │                      │
│        │           │   [SELECT]──> PC    │                      │
│        │           │                     │                      │
│ 18 opcodes         │ +6 opcodes          │ +8 opcodes           │
│ ✅ Done            │ ❌ TODO             │ ❌ TODO              │
└────────────────────┴─────────────────────┴──────────────────────┘
```

## Implementation Complexity

### Phase 1: Multi-Operation (Easier)

```
Input:  Computation graph with 3 nodes
        Node 1: CMP_EQ
        Node 2: ADD
        Node 3: PC_CONDITIONAL

Output: Single weight matrix that does all 3 sequentially

Challenges:
  - Topological sort (ordering)
  - Slot allocation (TEMP storage)
  - Sequential emission

Estimated: 1-2 days
```

### Phase 2: Multi-Layer (Medium)

```
Input:  MultiLayerWeights with 3 layers
        Layer 1: FFN graph (setup)
        Layer 2: Attention (memory lookup)
        Layer 3: FFN graph (copy result)

Output: Weights for layers 9 and 11 (skip layer 10)

Challenges:
  - Compile each FFN separately
  - Track which layer goes where
  - Coordinate with attention

Estimated: 2-3 days
```

## Final Coverage Map

```
┌─────────────────────────────────────────────────────────────┐
│                    39 C4 Opcodes                            │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴──────────────┐
              │                              │
         Autoregressive                 External I/O
         (34 opcodes)                   (5 opcodes)
              │                              │
      ┌───────┴───────┐                  OPEN, READ
      │               │                  CLOS, PRTF
  Pure FFN      Attention                EXIT, GETCHAR
  (26 opcodes)  (8 opcodes)             PUTCHAR
      │               │
      │           LI, LC, SI, SC        ❌ Cannot be
      │           PSH, JSR, ENT, LEV       autoregressive
      │                                    (by design)
      │
  ┌───┴───┐
  │       │
Single   Multi-op
(18)     (8)
  │       │
  │     JMP, BZ, BNZ
  │     ADJ, MALC, FREE
  │     MSET, MCMP
  │
ADD, SUB, MUL
DIV, MOD, EQ,
NE, LT, GT, LE,
GE, OR, XOR, AND,
SHL, SHR, LEA, IMM

✅ Done: 18 (46%)
🔄 Phase 1: +6 → 24 (62%)
🔄 Phase 2: +8 → 32 (82%)
❌ External: 5 (13%)
⚠️  Skip: 2 (5%) - MSET/MCMP too complex
```

## Summary Answer

**Q: What's multi-operation vs multi-layer?**

**Multi-Operation**:
- Multiple operations in SAME FFN layer
- Like: "compare, add, select" all in one pass
- Example: BZ checks AX==0, calculates PC+8, selects target/fallthrough
- Needs: Chaining operations with TEMP slots

**Multi-Layer**:
- Operations across DIFFERENT transformer layers
- Like: "setup → attention → copy" in 3 layers
- Example: LI sets up address, attention looks up memory, FFN copies result
- Needs: Compile each FFN layer separately

**Implementation**: 3-5 days total
**Result**: 32/39 opcodes (82%) with weight generation ✅
