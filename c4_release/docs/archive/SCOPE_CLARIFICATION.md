# Graph Weight Compiler - Scope Clarification

## TL;DR

**Our graph weight compiler compiles OPERATIONS (like ALU primitives), NOT a full autoregressive VM.**

- ✅ We compile: Individual operations (ADD, MUL, CMP_GT, etc.)
- ❌ We don't compile: Full instruction fetch/decode/execute cycle
- ❌ We don't compile: Autoregressive token generation
- ❌ We don't compile: Memory management, PC updates, register forwarding

---

## Two Different Things

### 1. Full Autoregressive Neural VM (Existing - vm_step.py)

**What it is:**
- 16-layer decoder-only transformer
- Generates VM execution traces autoregressively (token by token)
- Each "step" = 35 tokens: `REG_PC(5) + REG_AX(5) + REG_SP(5) + REG_BP(5) + STACK0(5) + MEM(9) + STEP_END(1)`

**Architecture:**
```
Layer 0:  Step structure (recognize 35-token boundaries)
Layer 1:  Fine thresholds + step-end detection
Layer 2:  Step distance encoding
Layer 3:  PC increment + branch override
Layer 4:  PC relay + PC+1 synthesis
Layer 5:  Opcode/immediate fetch via attention (memory lookup)
Layer 6:  Register carry-forward (AX, SP, BP)
Layer 7:  STACK0 address computation
Layer 8:  Stack memory read
Layer 9:  ← ALU START: Operand gather + simple ops (ADD/SUB/bitwise/CMP)
Layer 10: ← ALU: Carry propagation + comparisons + shift prep
Layer 11: ← ALU: MUL computation + DIV step A
Layer 12: ← ALU: MUL carry + DIV step B + SHL/SHR
Layer 13: Branch condition + SP/BP writeback + control flow
Layer 14: Memory address computation
Layer 15: Memory lookup (attention) + output routing + HALT
```

**Parameters:**
- d_model: 512
- num_heads: 8
- ffn_dim: 4096
- num_layers: 16
- **Total: ~50M parameters** (typical transformer)

**What it does:**
- Fetches bytecode instruction from memory (via attention)
- Decodes opcode
- Reads operands from registers/memory
- Executes ALU operation
- Writes result back to registers/memory
- Updates PC
- Generates next step autoregressively

**Tested with:**
- 3000+ opcode-level tests
- 1000+ full C programs (factorial, fibonacci, loops, recursion)

---

### 2. Graph Weight Compiler (Our Work - graph_weight_compiler.py)

**What it is:**
- Compiles individual operations to sparse FFN weight matrices
- Each operation = one forward pass through weights
- Operations can be composed into computation graphs

**What it compiles:**
```python
# Example: Compile ADD operation
graph = ComputationGraph()
a = graph.add_node(CONST, value=42)
b = graph.add_node(CONST, value=58)
result = graph.add_node(ADD, inputs=[a, b])

compiler = WeightCompiler()
weights = compiler.compile(graph)  # → W_up, W_gate, W_down matrices

# Use weights in forward pass
hidden = silu(W_up @ x + b_up) * (W_gate @ x + b_gate)
output = x + W_down @ hidden + b_down
# Result: output[result_reg] = 100
```

**Parameters:**
- Per operation: 10-1,536 non-zero weights
- 22 operations total: **4,712 non-zero parameters**
- Overall: **98.44% sparse**

**What it does:**
- Compiles primitive operations (ADD, MUL, CMP_GT, SELECT, etc.)
- Generates sparse weight matrices
- Operations execute in one forward pass (non-autoregressive)
- Can be composed into multi-op graphs

**Tested with:**
- 48 integration tests (100% passing)
- Direct operation correctness testing
- Composition testing (max, abs functions)

---

## Comparison Table

| Aspect | Full Autoregressive VM | Graph Weight Compiler |
|--------|------------------------|----------------------|
| **Architecture** | 16-layer transformer | Single FFN layer per operation |
| **Generation** | Autoregressive (token-by-token) | Single forward pass |
| **Scope** | Full instruction cycle | Individual operations |
| **Parameters** | ~50M (typical transformer) | 4.7K non-zero (all ops) |
| **What it implements** | 256 opcodes (full C4 ISA) | 22 primitive operations |
| **Layers** | Fetch → Decode → ALU → Memory → Write | Just computation (ALU-like) |
| **Input format** | Token sequence with markers | Register values |
| **Output format** | Next token prediction | Register values |
| **Testing** | 3000+ opcodes, 1000+ C programs | 48 operation tests |

---

## Where Our Compiler Fits

### Current Use Case: Building Block Library

Our compiler provides **primitive operations** that could be used:

1. **As ALU components** (Layers 9-12 of the full VM)
   - Replace hand-crafted ALU weights with compiler-generated
   - Example: CMP_GT, ADD, MUL operations in layer 9-12

2. **For meta-learning / task adaptation**
   - Generate task-specific operations on-the-fly
   - Compose primitives into custom functions

3. **For neural program synthesis**
   - Compile programs to differentiable operations
   - Optimize via backpropagation

### What Would Be Needed for Full VM Compilation

To compile a **full autoregressive VM** from scratch:

```
1. ✅ Primitive operations (DONE - our compiler)
   └─ 22 ops: ADD, MUL, CMP_GT, SELECT, etc.

2. 🔲 Instruction fetch/decode layer
   └─ Attention-based memory lookup
   └─ Opcode decoding (256 opcodes)

3. 🔲 Register management layers
   └─ Carry-forward (maintain AX, SP, BP across steps)
   └─ Register writeback

4. 🔲 Memory management layers
   └─ Stack operations (PUSH/POP)
   └─ Memory read/write via attention

5. 🔲 Control flow layers
   └─ PC increment
   └─ Branch condition evaluation
   └─ Jump target computation

6. 🔲 Autoregressive structure
   └─ Step boundary detection (35 tokens)
   └─ Token-by-token generation
   └─ STEP_END / HALT emission

7. 🔲 Full integration
   └─ 16-layer architecture
   └─ Layer-specific weight generation
   └─ Training/initialization strategy
```

**Current status:** Step 1 complete (22 primitive operations)

---

## Analogy

Think of it like building a CPU:

### Full Autoregressive VM = Complete CPU
```
┌─────────────────────────────────────┐
│          Complete CPU               │
│  ┌─────────────────────────────┐   │
│  │  Instruction Fetch Unit     │   │ ← Layers 0-5
│  │  Instruction Decode         │   │ ← Layer 5
│  │  Register File              │   │ ← Layers 6-7
│  │  ALU (ADD, MUL, CMP, etc.)  │   │ ← Layers 9-12 ✓ OUR SCOPE
│  │  Memory Management Unit     │   │ ← Layers 8, 14-15
│  │  Control Unit (PC, branches)│   │ ← Layers 3-4, 13
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

### Our Graph Compiler = ALU Component Library
```
┌─────────────────────────────────────┐
│       ALU Component Library         │
│  ┌─────────────────────────────┐   │
│  │  ADD  unit                  │   │ ✓
│  │  SUB  unit                  │   │ ✓
│  │  MUL  unit                  │   │ ✓
│  │  DIV  unit                  │   │ ✓
│  │  CMP_GT, CMP_LT units       │   │ ✓
│  │  AND, OR, NOT units         │   │ ✓
│  │  SELECT (mux) unit          │   │ ✓
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

We've built the **ALU components**, not the full CPU.

---

## Summary: Just the ALU

**Question:** "Is it compiling to a full autoregressive model or just the ALU?"

**Answer:** **Just the ALU (well, ALU-like primitive operations).**

More precisely:

- ✅ We compile **primitive operations** (arithmetic, logic, comparisons, conditionals)
- ✅ These are **building blocks** that could replace layers 9-12 in the full VM
- ✅ Single forward pass per operation (non-autoregressive)
- ✅ 4,712 non-zero parameters total (extremely sparse)

- ❌ We do **NOT** compile the full instruction cycle (fetch, decode, execute, writeback)
- ❌ We do **NOT** generate autoregressive token sequences
- ❌ We do **NOT** handle PC management, memory, registers, control flow
- ❌ We do **NOT** implement the 16-layer transformer architecture

**Scope:** ALU-like primitive operations, not full VM compilation.

**Use case:** Building blocks that could be:
1. Composed into higher-level functions
2. Integrated into the ALU layers of the full VM
3. Used for meta-learning and task adaptation
4. Foundation for eventual full VM compilation

**Status:** Foundation complete, proven correct, ready for next level of integration.
