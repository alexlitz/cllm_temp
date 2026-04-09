# Weight Setting Approaches: Hand-Set vs Compiled

**Date**: 2026-04-07
**Status**: Active - Both approaches coexist in the codebase

## Overview

The Neural VM supports **two fundamentally different approaches** to configuring the transformer's weights for VM execution. Both produce fully neural implementations, but differ in **how the weights are created**.

---

## Approach 1: Hand-Set Weights (`set_vm_weights()`)

**File**: `neural_vm/vm_step.py` (lines 1280-2700+)

### What It Is

A **manually crafted** weight setting function that directly sets individual matrix elements with precise numerical values. This is like writing assembly code - every weight value is explicitly specified.

### Architecture

```python
def set_vm_weights(model, enable_tool_calling=False,
                   enable_conversational_io=False, alu_mode='lookup'):
    """Set weights into AutoregressiveVM for true neural VM execution.

    Architecture (16 layers, d_model=512, 8 heads, FFN=4096):
      L0:  Step structure (threshold attention for 39-token step)
      L1:  Fine thresholds + HAS_SE + CS distance
      L2:  CS distance refinement → thermometer encoding
      L3:  PC carry-forward + PC increment + branch override
      L4:  PC relay + PC+1 synthesis
      L5:  Opcode/immediate fetch via memory keys
      L6:  Register carry-forward (AX, SP, BP) + SP dispatch
      L7:  STACK0 address build (intra-step SP gather)
      L8:  STACK0 memory read (softmax1 at SP address)
      L9:  Operand gather + simple ALU (ADD/SUB raw, bitwise, comparisons)
      L10: Carry apply + comparison cascade + shift precompute
      L11: MUL operand gather + cross-term products + DIV step A
      L12: MUL carry propagation + DIV step B + SHL/SHR select
      L13: Branch condition broadcast + SP/BP writeback + control flow
      L14: MEM address gather
      L15: Memory lookup (softmax1) + final output routing + HALT
    """
```

### Example: Manual Weight Setting

```python
# Layer 0: Step structure via threshold attention
attn0 = model.blocks[0].attn
attn0.alibi_slopes.fill_(ALIBI_S)
_set_threshold_attn(
    attn0,
    [3.5, 4.5, 7.5, 8.5, 9.5, 14.5, 19.5, 24.5],
    [BD.H0, BD.H1, BD.H2, BD.H3, BD.H4, BD.H5, BD.H6, BD.H7],
    ALIBI_S,
    HD,
)

ffn0 = model.blocks[0].ffn
_set_phase_a_ffn(ffn0, S, BD)

# Embedding: Manual one-hot assignments
embed = model.embed.embed.weight
embed.zero_()

for tok in range(V):
    embed[tok, BD.CONST] = 1.0

for tok, dim in [
    (Token.REG_PC, BD.MARK_PC),
    (Token.REG_AX, BD.MARK_AX),
    (Token.REG_SP, BD.MARK_SP),
    (Token.REG_BP, BD.MARK_BP),
]:
    embed[tok, dim] = 1.0
    embed[tok, BD.IS_MARK] = 1.0

# Layer 9: ADD operation (manual matrix manipulation)
# W_up[unit, input_dim] = value for SwiGLU activation
W_up[0, BD.NIB_A] = S
W_up[0, BD.NIB_B] = S
W_gate[0, BD.OPCODE_FLAGS + some_offset] = S
W_down[BD.RAW_SUM, 0] = 1.0 / (S * S)
```

### Characteristics

✅ **Advantages**:
- **Complete control**: Every single weight value is explicitly set
- **Optimized**: Hand-tuned for maximum efficiency
- **Production-ready**: Currently powers the working Neural VM
- **Well-tested**: 2000+ tests verify correctness
- **Debuggable**: Can trace exact weight values

⚠️ **Disadvantages**:
- **Verbose**: ~1500 lines of weight setting code
- **Error-prone**: Easy to get indices wrong
- **Hard to modify**: Changing architecture requires rewriting many lines
- **Not reusable**: Patterns aren't abstracted
- **Maintenance burden**: Manual tracking of dimension indices, scaling factors

### When to Use

- ✅ Production deployment (current working implementation)
- ✅ Performance-critical applications (hand-optimized)
- ✅ When you need exact control over every weight
- ✅ When architecture is stable and well-understood

---

## Approach 2: Compiled Weights (Weight Compiler Infrastructure)

**Files**:
- `neural_vm/graph_weight_compiler.py` - Graph-based compiler with register allocation
- `neural_vm/weight_compiler.py` - High-level abstractions
- `neural_vm/nibble_weight_compiler.py` - Nibble-specific compiler

### What It Is

A **compiler infrastructure** that generates weights from high-level specifications. This is like writing in a high-level language (C, Python) instead of assembly.

### Architecture

```
Source Specification → IR Graph → Register Allocation → Weight Emission → PyTorch Weights
```

#### Pipeline Stages

**Stage 1: High-Level Specification**
```python
# Declare WHAT you want, not HOW to implement it
add_op = Operation("ADD")

with add_op.layer(9):
    # Raw sum using cancel pair primitive
    add_op.cancel_pair(
        inputs=["NIB_A", "NIB_B"],
        output="RAW_SUM",
        gate="OPCODE_ADD"
    )

    # Carry generation using step primitive
    add_op.step(
        condition="NIB_A + NIB_B >= base",
        output="CARRY_OUT",
        gate="OPCODE_ADD"
    )
```

**Stage 2: Intermediate Representation (IR)**
```python
IR = [
    IRNode(
        id=0,
        op=OpType.ADD,
        inputs=[vreg_a, vreg_b],
        output_reg="RAW_SUM",
        params={'method': 'cancel_pair'},
        gate="OPCODE_ADD"
    ),
    IRNode(
        id=1,
        op=OpType.CMP_GE,
        inputs=[vreg_sum],
        output_reg="CARRY_OUT",
        params={'threshold': 16, 'method': 'step_pair'}
    ),
]
```

**Stage 3: Register Allocation (Graph Coloring)**
```python
# Allocate virtual registers to physical dimensions
allocation = {
    "RAW_SUM": 100,      # Physical dimension 100
    "CARRY_OUT": 102,    # Physical dimension 102
    "NIB_A": 0,          # Physical dimension 0
    "NIB_B": 16,         # Physical dimension 16
}
```

**Stage 4: Weight Emission**
```python
# Compiler automatically generates weight matrices
class CancelPairCompiler:
    def compile(self, node, unit_offset, registers, S):
        weights = WeightValues()

        # Unit 0: Positive sum
        for inp in node.inputs:
            weights.W_up[unit_offset, registers[inp]] = S
        weights.W_gate[unit_offset, registers[node.gate]] = S
        weights.W_down[registers[node.output], unit_offset] = 1.0 / (S * S)

        # Unit 1: Negative sum (for cancel pair)
        for inp in node.inputs:
            weights.W_up[unit_offset + 1, registers[inp]] = -S
        weights.W_gate[unit_offset + 1, registers[node.gate]] = -S
        weights.W_down[registers[node.output], unit_offset + 1] = 1.0 / (S * S)

        return weights
```

### Primitive Operations (Building Blocks)

The compiler provides **reusable primitives**:

```python
# Arithmetic primitives
OpType.ADD           # a + b
OpType.SUB           # a - b
OpType.MUL           # a * b (lookup table)
OpType.DIV           # a / b (lookup table)

# Comparison primitives
OpType.CMP_EQ        # a == b
OpType.CMP_LT        # a < b
OpType.CMP_GE        # a >= b

# Logical primitives
OpType.AND           # a && b
OpType.OR            # a || b
OpType.SELECT        # cond ? a : b

# SwiGLU patterns
cancel_pair()        # (+sum) + (-sum) for signed addition
step_pair()          # step(x >= threshold) using ReLU
clear_pair()         # Clears values
equality_step()      # step(x == value)

# Memory primitives
OpType.LOOKUP        # Attention-based memory lookup
OpType.STORE         # Memory write request
```

### Example: Compiled vs Hand-Set

**Hand-Set Approach** (40 lines):
```python
# Manual weight matrix manipulation
W_up = model.blocks[9].ffn.W_up
W_gate = model.blocks[9].ffn.W_gate
W_down = model.blocks[9].ffn.W_down
b_up = model.blocks[9].ffn.b_up

# Unit 0: Positive sum component
W_up[0, BD.NIB_A] = S
W_up[0, BD.NIB_B] = S
W_gate[0, BD.OPCODE_FLAGS + ADD_OFFSET] = S
W_down[BD.RAW_SUM, 0] = 1.0 / (S * S)

# Unit 1: Negative sum component
W_up[1, BD.NIB_A] = -S
W_up[1, BD.NIB_B] = -S
W_gate[1, BD.OPCODE_FLAGS + ADD_OFFSET] = -S
W_down[BD.RAW_SUM, 1] = 1.0 / (S * S)

# Unit 2: Carry generate (threshold >= base)
W_up[2, BD.NIB_A] = S
W_up[2, BD.NIB_B] = S
b_up[2] = -S * (base - 1.0)
W_gate[2, BD.OPCODE_FLAGS + ADD_OFFSET] = 1.0
W_down[BD.CARRY_OUT, 2] = 1.0 / S

# Unit 3: -step(>= base+1)
W_up[3, BD.NIB_A] = S
W_up[3, BD.NIB_B] = S
b_up[3] = -S * base
W_gate[3, BD.OPCODE_FLAGS + ADD_OFFSET] = 1.0
W_down[BD.CARRY_OUT, 3] = -1.0 / S

# ... (20 more lines for propagate flag, etc.)
```

**Compiled Approach** (10 lines):
```python
# High-level operation specification
compiler = WeightCompiler(config)
add_op = compiler.operation("ADD")

with add_op.layer(9):
    # Declare intent, compiler handles details
    add_op.cancel_pair(
        inputs=["NIB_A", "NIB_B"],
        output="RAW_SUM"
    )

    add_op.step(
        condition="NIB_A + NIB_B >= base",
        output="CARRY_OUT"
    )

# Compiler automatically:
# - Allocates hidden units (4 units for these operations)
# - Sets W_up, W_gate, W_down, b_up with correct scaling
# - Manages dimension indices
# - Verifies correctness
```

### Characteristics

✅ **Advantages**:
- **Concise**: 4x less code for typical operations
- **Type-safe**: Register names validated at compile time
- **Reusable**: cancel_pair, step_pair used across all ops
- **Maintainable**: Change implementation once, affects all uses
- **Composable**: Build complex ops from simple primitives
- **Self-documenting**: Code reads like specification
- **Optimizable**: Compiler can apply optimization passes

⚠️ **Disadvantages**:
- **Abstraction overhead**: May not expose all low-level controls
- **Development status**: Infrastructure partially implemented
- **Not production**: Hand-set weights currently in use
- **Complexity**: Requires understanding compiler pipeline

### When to Use

- ✅ Experimenting with new operations
- ✅ Rapid prototyping of VM architectures
- ✅ When maintainability matters more than control
- ✅ Building reusable operation libraries
- ✅ Future: When compiler is production-ready

---

## Comparison Summary

| Aspect | Hand-Set (`set_vm_weights`) | Compiled (Weight Compiler) |
|--------|----------------------------|---------------------------|
| **Abstraction Level** | Assembly-level | High-level language |
| **Lines of Code** | ~2,000 lines (incl. helpers) | 3,704 lines |
| **Core Function** | 487 lines | N/A |
| **Control** | Complete, explicit | Through abstractions |
| **Readability** | Low (indices, scaling) | High (intent-driven) |
| **Maintainability** | Difficult | Easy |
| **Reusability** | None (copy-paste) | High (primitives) |
| **Error Proneness** | High | Low (type-checked) |
| **Production Status** | ✅ Active, tested | ⚠️ Development |
| **Performance** | Hand-optimized | Compiler-optimized |
| **Learning Curve** | Steep | Moderate |
| **Flexibility** | Maximum | Through extensions |

**Note**: Compiler infrastructure is actually **1.9x larger** than hand-set code when including all supporting functions. However, it provides higher-level abstractions that make individual operation definitions much shorter.

---

## Code Examples

### Example 1: Embedding Setup

**Hand-Set**:
```python
embed = model.embed.embed.weight
embed.zero_()

for tok in range(V):
    embed[tok, BD.CONST] = 1.0

for b in range(256):
    embed[b, BD.IS_BYTE] = 1.0
    embed[b, BD.EMBED_LO + (b & 0xF)] = 1.0
    embed[b, BD.EMBED_HI + ((b >> 4) & 0xF)] = 1.0
    embed[b, BD.CLEAN_EMBED_LO + (b & 0xF)] = 1.0
    embed[b, BD.CLEAN_EMBED_HI + ((b >> 4) & 0xF)] = 1.0
```

**Compiled**:
```python
compiler.embedding()
    .set_all(Token.CONST, 1.0)
    .set_bytes_one_hot("EMBED_LO", "EMBED_HI")
    .set_bytes_one_hot("CLEAN_EMBED_LO", "CLEAN_EMBED_HI")
    .compile()
```

### Example 2: Threshold Attention

**Hand-Set**:
```python
attn0 = model.blocks[0].attn
attn0.alibi_slopes.fill_(ALIBI_S)

for h_idx, (threshold, output_dim) in enumerate([
    (3.5, BD.H0), (4.5, BD.H1), (7.5, BD.H2), (8.5, BD.H3),
    (9.5, BD.H4), (14.5, BD.H5), (19.5, BD.H6), (24.5, BD.H7)
]):
    base = h_idx * HD
    attn0.W_q[base, BD.CONST] = 10.0
    attn0.W_k[base, BD.IS_MARK] = 10.0
    attn0.W_v[base + 1, BD.IS_MARK] = 1.0
    attn0.W_o[output_dim, base + 1] = 1.0
    attn0.b_k[base] = -10.0 * threshold
```

**Compiled**:
```python
compiler.layer(0).attention()
    .threshold_heads(
        thresholds=[3.5, 4.5, 7.5, 8.5, 9.5, 14.5, 19.5, 24.5],
        outputs=["H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7"],
        query_key="IS_MARK",
        alibi_slope=ALIBI_S
    )
    .compile()
```

### Example 3: ALU Operation (SUB)

**Hand-Set** (60 lines):
```python
def _set_layer9_sub_ffn(ffn, S, BD):
    W_up = ffn.W_up
    W_gate = ffn.W_gate
    W_down = ffn.W_down
    b_up = ffn.b_up

    # Manual weight setting for subtraction...
    # Units 0-1: RAW_DIFF = A - B (cancel pair with negated B)
    W_up[0, BD.NIB_A] = S
    W_up[0, BD.NIB_B] = -S
    W_gate[0, BD.OPCODE_FLAGS + SUB_OFFSET] = S
    W_down[BD.RAW_DIFF, 0] = 1.0 / (S * S)

    W_up[1, BD.NIB_A] = -S
    W_up[1, BD.NIB_B] = S
    W_gate[1, BD.OPCODE_FLAGS + SUB_OFFSET] = -S
    W_down[BD.RAW_DIFF, 1] = 1.0 / (S * S)

    # Units 2-3: BORROW_OUT = step(A < B)
    W_up[2, BD.NIB_A] = -S
    W_up[2, BD.NIB_B] = S
    b_up[2] = -S * (-1.0)  # threshold for A < B
    W_gate[2, BD.OPCODE_FLAGS + SUB_OFFSET] = 1.0
    W_down[BD.BORROW_OUT, 2] = 1.0 / S

    # ... (40 more lines)
```

**Compiled** (15 lines):
```python
compiler = WeightCompiler(config)
sub_op = compiler.operation("SUB")

with sub_op.layer(9):
    # Raw difference (A - B via cancel pair with negation)
    sub_op.cancel_pair(
        inputs=["NIB_A", "~NIB_B"],  # ~ means negate
        output="RAW_DIFF"
    )

    # Borrow flag (A < B)
    sub_op.step(
        condition="NIB_A < NIB_B",
        output="BORROW_OUT"
    )

    # Propagate flag (A == B)
    sub_op.equality_step(
        left="NIB_A",
        right="NIB_B",
        output="BORROW_PROP"
    )
```

---

## Current Status

### Hand-Set Weights
- ✅ **Production**: Active, powers working Neural VM
- ✅ **Complete**: All 16 layers implemented
- ✅ **Tested**: 2000+ tests passing
- ✅ **Supports**: 41/43 opcodes (95%)

### Compiled Weights
- ⚠️ **Development**: Infrastructure partially implemented
- ⚠️ **Experimental**: Not yet used in production
- ⚠️ **In Progress**: Graph compiler, register allocation complete
- ⚠️ **Future**: Plan to port hand-set implementation

---

## Migration Path

### Phase 1: Coexistence (Current)
- Hand-set weights power production
- Compiler infrastructure being developed
- Both approaches maintained separately

### Phase 2: Validation
- Port individual operations to compiled approach
- Verify weight outputs match hand-set exactly
- Run full test suite on compiled weights

### Phase 3: Transition
- Compiled weights become primary
- Hand-set weights as reference/fallback
- Performance comparison and optimization

### Phase 4: Deprecation
- Compiled weights fully replace hand-set
- Hand-set code archived for reference
- Compiler becomes standard approach

---

## Recommendations

### Use Hand-Set Weights When:
1. You need **production-ready** execution (current requirement)
2. You want **maximum control** over every weight value
3. You need **proven, tested** implementation
4. Performance is critical and hand-optimization matters

### Use Compiled Weights When:
1. You're **prototyping** new operations or architectures
2. **Maintainability** is more important than raw control
3. You want **reusable** operation primitives
4. You're building a **new VM** from compiler abstractions
5. The compiler infrastructure is production-ready (future)

### For New Contributors:
- Start by **reading** hand-set weights to understand architecture
- Use **compiler abstractions** for new operations
- Contribute to **compiler development** for long-term maintainability

---

## Key Files Reference

### Hand-Set Approach
- `neural_vm/vm_step.py` (lines 1280+) - Main `set_vm_weights()` function
- `neural_vm/embedding.py` - Embedding dimension definitions
- `neural_vm/vm_step.py` (lines 600-1280) - Helper functions for weight setting

### Compiled Approach
- `neural_vm/graph_weight_compiler.py` - Graph-based IR and register allocation
- `neural_vm/weight_compiler.py` - High-level operation abstractions
- `neural_vm/nibble_weight_compiler.py` - Nibble-specific compiler
- `docs/WEIGHT_COMPILER_DESIGN.md` - Design documentation

### Related
- `src/baked_c4.py` - "Baked" transformer (compiler embedded in weights)
- `neural_vm/alu/` - ALU operation implementations (candidates for compilation)
- `AUTOREGRESSIVE_PURITY_AUDIT.md` - Purity requirements for both approaches

---

## Conclusion

Both approaches are **valid and coexist** in the codebase:

- **Hand-Set Weights**: Production-ready, assembly-level control, currently active
- **Compiled Weights**: High-level, maintainable, future direction

The hand-set approach built the working Neural VM. The compiled approach will make it maintainable and extensible.

**Current recommendation**: Use hand-set for production, develop compiler for future.
