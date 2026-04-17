# Multi-Operation vs Multi-Layer: What's the Difference?

## Current State: Single-Operation Only

**What works now**: Graphs with ONE operation

```
ADD:
  Input: pop, AX
  Operation: ADD(pop, AX)
  Output: AX

Graph: pop ──┐
             ├──[ADD]──> AX
        AX ──┘

Compiles to: ONE FFN layer with emit_add_nibble()
```

## Problem 1: Multi-Operation Graphs (SAME LAYER)

**What doesn't work**: Graphs with MULTIPLE operations that execute sequentially IN ONE FFN PASS

### Example: BZ (Branch if Zero)

```
BZ (if AX == 0, jump to target):

  AX ──────┬──[CMP_EQ]──> cond (0 or 1)
  zero ────┘              │
                          ├──[SELECT]──> PC
  target ─────────────────┤
  PC+8 ───────────────────┘

Operations in sequence:
1. CMP_EQ: Check if AX == 0
2. ADD: Calculate fallthrough = PC + 8
3. PC_CONDITIONAL: Select target or fallthrough based on condition
```

**Why it fails**:
```python
# Current code only handles ONE operation
ops = [n for n in graph.nodes.values() if n.op != OpType.CONST]
if len(ops) == 1:
    return self.compile_operation(ops[0].op, opcode)  # ✅ Works
else:
    raise NotImplementedError("Multi-operation not supported")  # ❌ BZ fails here
```

**What we need**: Chain operations together in ONE FFN layer
- Operation 1 writes to intermediate slots
- Operation 2 reads from those slots and writes result
- All happens in ONE forward pass

### Example: ADJ (Adjust Stack Pointer)

```
ADJ (SP += imm):

  SP ──────┐
           ├──[ADD]──> SP
  imm ─────┘

Operations:
1. ADD: Add immediate to SP
```

Simple, but currently fails because it's not recognized as FFN_DIRECT.

## Problem 2: Multi-Layer Operations (MULTIPLE LAYERS)

**What doesn't work**: Operations that need ATTENTION in between FFN operations

### Example: LI (Load Indirect - AX = *AX)

```
LI needs 3 transformer layers:

Layer 9 (FFN):
  AX ──[MOVE]──> MEM_ADDR mailbox
      [FLAG_SET]──> MEM_READ flag

Layer 10 (Attention):
  MEM_ADDR ──[Attention Lookup]──> MEM_DATA
  (Attention uses KV cache to find value at address)

Layer 11 (FFN):
  MEM_DATA ──[MOVE]──> AX
```

**Why it's different**:
- ATTENTION can't be compiled to FFN weights
- Attention is a separate mechanism (Q·K^T)
- We need to compile the FFN parts only

### Example: PSH (Push AX onto stack)

```
PSH needs 2 transformer layers:

Layer 9 (FFN):
  SP ──[SUB 8]──> SP_new
  AX ──[MOVE]──> MEM_DATA mailbox
  SP_new ──[MOVE]──> MEM_ADDR mailbox
  [FLAG_SET]──> MEM_WRITE flag

Layer 10 (Attention):
  MEM_ADDR, MEM_DATA ──[Attention Write]──> Memory KV cache
  (Attention stores value at address)
```

## Implementation Plan

### Phase 1: Multi-Operation Graph Compiler (1-2 days)

**Goal**: Compile graphs with multiple operations into ONE FFN layer

**Approach**: Topological sort + sequential emission

```python
class MultiOperationCompiler:
    """Compile multi-operation graphs to single FFN layer."""

    def compile_graph(self, graph: ComputationGraph, opcode: int):
        """
        1. Topological sort: Order operations by dependencies
        2. Allocate intermediate slots in E class
        3. Emit each operation sequentially
        4. Chain outputs → inputs
        """

        # Example for BZ:
        # 1. emit_cmp_eq_nibble(pos) → writes to TEMP[pos]
        # 2. emit_add_nibble(pos)    → PC + 8 to TEMP2[pos]
        # 3. emit_pc_conditional()   → reads TEMP, writes PC
```

**What this unlocks**:
- ✅ JMP, BZ, BNZ, ADJ (4 opcodes)
- ✅ MALC, FREE (2 opcodes)
- Total: +6 opcodes → 24/39 (62%)

### Phase 2: Multi-Layer Weight Generator (2-3 days)

**Goal**: Generate weights for each FFN layer in multi-layer operations

**Approach**: Compile each layer's FFN graph separately

```python
class MultiLayerWeightGenerator:
    """Generate weights for multi-layer operations."""

    def generate_multilayer_weights(self, opcode: int):
        """
        For LI (3 layers):

        Layer 9 FFN:
          - emit_move_nibble: AX → MEM_ADDR
          - emit_flag_set: MEM_READ flag
          → Returns: weights_layer9

        Layer 10: Attention (no compilation needed)

        Layer 11 FFN:
          - emit_move_nibble: MEM_DATA → AX
          → Returns: weights_layer11

        Return: {9: weights_layer9, 11: weights_layer11}
        """
```

**What this unlocks**:
- ✅ LI, LC, SI, SC (4 memory opcodes)
- ✅ PSH (1 stack opcode)
- ✅ JSR, ENT, LEV (3 function opcodes)
- Total: +8 opcodes → 32/39 (82%)

## Detailed Implementation Plan

### Step 1: Extend E Class with Intermediate Slots (30 min)

```python
# In neural_vm/embedding.py
class E:
    # ... existing slots ...

    # Intermediate computation slots for multi-operation graphs
    TEMP2 = 157       # Second temp slot
    TEMP3 = 158       # Third temp slot
    COND_FLAG = 159   # Condition flag for branches
```

### Step 2: Implement Topological Sort (1 hour)

```python
# In neural_vm/nibble_weight_compiler.py

def topological_sort(graph: ComputationGraph) -> List[NodeID]:
    """
    Sort nodes by dependencies (inputs before outputs).

    Returns list of node IDs in execution order.
    """
    # Standard topological sort algorithm
    visited = set()
    result = []

    def visit(node_id):
        if node_id in visited:
            return
        visited.add(node_id)
        node = graph.nodes[node_id]
        for input_id in node.inputs:
            visit(input_id)
        result.append(node_id)

    for node_id in graph.nodes:
        visit(node_id)

    return result
```

### Step 3: Implement Register Allocator (2 hours)

```python
def allocate_registers(graph: ComputationGraph) -> Dict[NodeID, SlotMapping]:
    """
    Allocate E class slots for each node's output.

    Returns mapping: node_id → (position, slot)
    """
    allocations = {}

    for node_id, node in graph.nodes.items():
        if node.output_name == "AX":
            allocations[node_id] = [(pos, E.RESULT) for pos in range(8)]
        elif node.output_name == "PC":
            allocations[node_id] = [(pos, E.PC_BASE + pos) for pos in range(8)]
        elif node.output_name.startswith("temp"):
            # Use TEMP/TEMP2/TEMP3 for intermediate values
            temp_idx = int(node.output_name[4:]) if len(node.output_name) > 4 else 0
            slot = E.TEMP + temp_idx
            allocations[node_id] = [(pos, slot) for pos in range(8)]

    return allocations
```

### Step 4: Implement Sequential Emitter (4 hours)

```python
def emit_sequential_operations(
    graph: ComputationGraph,
    opcode: int,
    emitter: NibbleWeightEmitter
) -> Dict[str, torch.Tensor]:
    """
    Emit operations sequentially in topological order.
    """

    # 1. Sort operations
    sorted_nodes = topological_sort(graph)

    # 2. Allocate slots
    slot_map = allocate_registers(graph)

    # 3. Emit each operation
    for node_id in sorted_nodes:
        node = graph.nodes[node_id]

        if node.op == OpType.CONST:
            # Emit constant directly to slot
            emit_constant(emitter, node.value, slot_map[node_id])

        elif node.op == OpType.ADD:
            # Read inputs from their slots
            input1_slots = slot_map[node.inputs[0]]
            input2_slots = slot_map[node.inputs[1]]
            output_slots = slot_map[node_id]

            # Setup input nibbles
            for pos in range(8):
                # Copy to NIB_A, NIB_B
                emit_move(emitter, input1_slots[pos], (pos, E.NIB_A))
                emit_move(emitter, input2_slots[pos], (pos, E.NIB_B))

            # Perform ADD
            for pos in range(8):
                emitter.emit_add_nibble(pos)

            # Copy result to output slot
            for pos in range(8):
                emit_move(emitter, (pos, E.RESULT), output_slots[pos])

        elif node.op == OpType.CMP_EQ:
            # Similar pattern...
            pass

        # ... handle all OpTypes

    return emitter.get_weights()
```

### Step 5: Implement Multi-Layer Generator (4 hours)

```python
class MultiLayerWeightGenerator:
    def generate_weights(
        self,
        multilayer: MultiLayerWeights,
        opcode: int
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Generate weights for each FFN layer.

        Returns: {layer_idx: weights}
        """
        layer_weights = {}

        for i, (layer_type, graph) in enumerate(multilayer.layers):
            if layer_type == "ffn":
                # Compile this layer's FFN graph
                weights = emit_sequential_operations(graph, opcode)
                layer_weights[i] = weights
            elif layer_type == "attention":
                # Skip - attention uses existing mechanism
                layer_weights[i] = None

        return layer_weights
```

### Step 6: Integration & Testing (2 hours)

```python
# Update OpcodeNibbleCompiler to use new compilers

def compile_opcode(self, opcode: int):
    if self.is_single_operation(opcode):
        # Use existing single-operation compiler
        return self.compile_operation(...)
    elif self.is_multi_operation(opcode):
        # Use new multi-operation compiler
        graph = self.mapper.map_opcode(opcode)
        return self.multi_op_compiler.compile_graph(graph, opcode)
    elif self.is_multi_layer(opcode):
        # Use new multi-layer generator
        multilayer = self.mapper.map_opcode_multilayer(opcode)
        return self.multilayer_gen.generate_weights(multilayer, opcode)
```

## Testing Strategy

### Phase 1 Tests (Multi-Operation)

```python
def test_multi_operation():
    # Test ADJ (simple: just ADD)
    weights = compiler.compile_opcode(Opcode.ADJ)
    assert weights is not None

    # Test BZ (complex: CMP + ADD + SELECT)
    weights = compiler.compile_opcode(Opcode.BZ)
    assert weights is not None

    # Test MALC (very complex: ADD + CMP + SELECT × 3)
    weights = compiler.compile_opcode(Opcode.MALC)
    assert weights is not None
```

### Phase 2 Tests (Multi-Layer)

```python
def test_multi_layer():
    # Test LI (3 layers)
    weights = compiler.compile_multilayer(Opcode.LI)
    assert 9 in weights  # FFN setup layer
    assert 11 in weights  # FFN copy layer

    # Test PSH (2 layers)
    weights = compiler.compile_multilayer(Opcode.PSH)
    assert 9 in weights  # FFN setup
```

## Timeline

**Week 1 (Phase 1 - Multi-Operation)**:
- Day 1: Topological sort + register allocation (3 hours)
- Day 2: Sequential emitter core (4 hours)
- Day 3: Test & debug ADJ, JMP (2 hours)
- Day 4: Test & debug BZ, BNZ (2 hours)
- Day 5: Test & debug MALC, FREE (2 hours)

**Result**: 24/39 opcodes (62%) with weight generation ✅

**Week 2 (Phase 2 - Multi-Layer)**:
- Day 1-2: Multi-layer weight generator (6 hours)
- Day 3: Test LI, SI (3 hours)
- Day 4: Test PSH, JSR (3 hours)
- Day 5: Test ENT, LEV (3 hours)

**Result**: 32/39 opcodes (82%) with weight generation ✅

## What You Get

After both phases:
- ✅ **32/39 opcodes** compiled to sparse neural weights
- ✅ **82% coverage** of all C4 operations
- ✅ **100% autoregressive** (except 7 external syscalls by design)
- ✅ **Production-ready** compilation system

The 7 remaining (OPEN, READ, CLOS, PRTF, EXIT, GETCHAR, PUTCHAR) are external syscalls that MUST remain external - they're the tool-use boundary by design.
