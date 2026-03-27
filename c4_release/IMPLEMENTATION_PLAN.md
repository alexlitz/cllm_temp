# Concrete Implementation Plan: Multi-Operation & Multi-Layer Compilation

## Summary

**Goal**: Increase from 18/39 (46%) to 32/39 (82%) opcodes with weight generation

**Approach**:
1. **Phase 1** (1-2 days): Multi-operation compiler → +6 opcodes → 24/39 (62%)
2. **Phase 2** (2-3 days): Multi-layer generator → +8 opcodes → 32/39 (82%)

## Phase 1: Multi-Operation Compiler

### What It Does

Compiles graphs with MULTIPLE operations into a SINGLE FFN layer by:
1. Sorting operations by dependencies (topological sort)
2. Allocating intermediate slots in E class
3. Emitting each operation sequentially
4. Chaining outputs to inputs

### File Structure

```
neural_vm/
├── nibble_weight_compiler.py          (existing)
├── multi_operation_compiler.py        (NEW - 300 lines)
│   ├── topological_sort()
│   ├── allocate_intermediate_slots()
│   ├── emit_operation_sequence()
│   └── MultiOperationCompiler class
└── opcode_nibble_integration.py      (UPDATE)
    └── Use MultiOperationCompiler for multi-op graphs
```

### Implementation Steps

#### Step 1.1: Create multi_operation_compiler.py (2 hours)

```python
"""
Multi-Operation Graph Compiler

Compiles computation graphs with multiple operations to single FFN layer.
"""

import torch
from typing import Dict, List, Tuple
from .graph_weight_compiler import ComputationGraph, OpType, NodeID
from .nibble_weight_compiler import NibbleWeightEmitter, NibbleRegisterMap
from .embedding import E


class IntermediateSlotAllocator:
    """Allocates E class slots for intermediate values."""

    def __init__(self):
        self.next_temp = E.TEMP  # Start at TEMP (slot 6)
        self.allocations = {}

    def allocate(self, node_id: NodeID, output_name: str) -> List[Tuple[int, int]]:
        """
        Allocate slots for a node's output.

        Returns: List of (position, slot) for 8 nibbles
        """
        # Named outputs get fixed slots
        if output_name == "AX":
            return [(pos, E.RESULT) for pos in range(8)]
        elif output_name == "SP":
            # SP updates go to RESULT, then copied to SP register
            return [(pos, E.RESULT) for pos in range(8)]
        elif output_name == "PC":
            # PC updates go to RESULT, then copied to PC register
            return [(pos, E.RESULT) for pos in range(8)]
        elif output_name == "cond":
            # Condition flags (0 or 1) use single slot
            return [(0, self.next_temp)]  # Single value, not per-nibble
        else:
            # Intermediate temps use sequential TEMP slots
            slots = [(pos, self.next_temp) for pos in range(8)]
            self.next_temp += 1
            if self.next_temp >= E.OP_START:
                raise RuntimeError("Ran out of temp slots")
            return slots


def topological_sort(graph: ComputationGraph) -> List[NodeID]:
    """
    Sort nodes by dependencies (topologically).

    Returns: List of node IDs in execution order
    """
    visited = set()
    result = []

    def visit(node_id):
        if node_id in visited:
            return
        visited.add(node_id)

        node = graph.nodes[node_id]
        # Visit inputs first (dependencies)
        for input_id in node.inputs:
            if input_id in graph.nodes:
                visit(input_id)

        result.append(node_id)

    # Visit all nodes
    for node_id in graph.nodes.keys():
        visit(node_id)

    return result


class MultiOperationCompiler:
    """Compile multi-operation graphs to single FFN layer."""

    def __init__(self, num_positions: int = 8):
        self.num_positions = num_positions
        self.reg_map = NibbleRegisterMap()

    def compile_graph(
        self,
        graph: ComputationGraph,
        opcode: int
    ) -> Dict[str, torch.Tensor]:
        """
        Compile multi-operation graph to weights.

        Strategy:
        1. Topological sort to get execution order
        2. Allocate slots for all intermediate values
        3. Emit operations sequentially
        4. Use TEMP slots to pass data between operations
        """

        # Create emitter
        emitter = NibbleWeightEmitter(opcode, self.num_positions)

        # Sort operations
        sorted_nodes = topological_sort(graph)

        # Allocate slots for all outputs
        allocator = IntermediateSlotAllocator()
        slot_map = {}
        for node_id in sorted_nodes:
            node = graph.nodes[node_id]
            if node.op != OpType.CONST:
                slot_map[node_id] = allocator.allocate(node_id, node.output_name)

        # Emit each operation
        for node_id in sorted_nodes:
            node = graph.nodes[node_id]

            if node.op == OpType.CONST:
                # Constants are embedded in other operations
                continue

            elif node.op == OpType.ADD:
                self._emit_add(emitter, node, graph, slot_map)

            elif node.op == OpType.CMP_EQ:
                self._emit_cmp_eq(emitter, node, graph, slot_map)

            elif node.op == OpType.PC_CONDITIONAL:
                self._emit_pc_conditional(emitter, node, graph, slot_map)

            # ... handle all OpTypes

        return emitter.get_weights()

    def _emit_add(self, emitter, node, graph, slot_map):
        """Emit ADD operation with input/output slot mapping."""
        # Get input values
        input1_val = self._get_input_value(node.inputs[0], graph, slot_map)
        input2_val = self._get_input_value(node.inputs[1], graph, slot_map)

        # For each nibble position
        for pos in range(self.num_positions):
            # Copy inputs to NIB_A, NIB_B
            self._emit_copy(emitter, input1_val[pos], (pos, E.NIB_A))
            self._emit_copy(emitter, input2_val[pos], (pos, E.NIB_B))

        # Emit ADD for all positions
        for pos in range(self.num_positions):
            emitter.emit_add_nibble(pos)

        # Copy results to output slots
        output_slots = slot_map[node.id]
        for pos in range(self.num_positions):
            self._emit_copy(emitter, (pos, E.RESULT), output_slots[pos])

    def _get_input_value(self, input_id, graph, slot_map):
        """Get the slot locations for an input value."""
        if input_id not in graph.nodes:
            raise ValueError(f"Input node {input_id} not found")

        input_node = graph.nodes[input_id]

        if input_node.op == OpType.CONST:
            # Constant value - return constant directly
            return input_node.value  # Single value

        else:
            # Variable - return its allocated slots
            return slot_map[input_id]

    def _emit_copy(self, emitter, src, dest):
        """Emit a nibble copy operation."""
        # Use MOVE pattern (cancel pair)
        # This is a helper - actual implementation uses W_up/W_down directly
        pass
```

#### Step 1.2: Implement Core Emitters (3 hours)

```python
# Continue in multi_operation_compiler.py

    def _emit_cmp_eq(self, emitter, node, graph, slot_map):
        """Emit CMP_EQ with slot mapping."""
        # Similar to _emit_add but calls emit_cmp_eq_nibble
        pass

    def _emit_pc_conditional(self, emitter, node, graph, slot_map):
        """
        Emit PC_CONDITIONAL: PC = cond ? target : fallthrough

        This is the key for BZ/BNZ compilation!
        """
        # Get inputs
        cond_slot = self._get_input_value(node.inputs[0], graph, slot_map)
        target_slots = self._get_input_value(node.inputs[1], graph, slot_map)
        fallthrough_slots = self._get_input_value(node.inputs[2], graph, slot_map)

        # Use existing emit_pc_conditional
        emitter.emit_pc_conditional(
            cond_slot=cond_slot[0][1],  # Condition is single value
            target_nibbles=target_slots,
            pc_nibbles=fallthrough_slots
        )
```

#### Step 1.3: Update OpcodeNibbleCompiler (1 hour)

```python
# In neural_vm/opcode_nibble_integration.py

from .multi_operation_compiler import MultiOperationCompiler

class OpcodeNibbleCompiler:
    def __init__(self, num_positions: int = 8):
        self.num_positions = num_positions
        self.nibble_compiler = NibbleWeightCompiler(num_positions)
        self.multi_op_compiler = MultiOperationCompiler(num_positions)  # NEW
        self.reg_map = NibbleRegisterMap()

    def compile_opcode(self, opcode: int) -> Dict[str, torch.Tensor]:
        """Compile with multi-operation support."""
        from .full_opcode_mapper import FullOpcodeMapper

        mapper = FullOpcodeMapper()

        # Get graph
        if opcode in [Opcode.LEA, Opcode.IMM, ...]:
            graph = mapper.map_opcode(opcode, imm=0)
        else:
            graph = mapper.map_opcode(opcode)

        # Count operations
        ops = [n for n in graph.nodes.values() if n.op != OpType.CONST]

        if len(ops) == 1:
            # Single operation - use existing compiler
            return self.nibble_compiler.compile_operation(ops[0].op, opcode)
        else:
            # Multi-operation - use new compiler
            return self.multi_op_compiler.compile_graph(graph, opcode)
```

#### Step 1.4: Mark Opcodes as Compilable (10 min)

```python
# In neural_vm/opcode_nibble_integration.py

_NIBBLE_OPCODE_SUPPORT = {
    # ... existing ...

    # Control flow (NOW compilable with multi-op)
    Opcode.JMP: (NibbleOpcodeSupport.FFN_DIRECT, None),  # Changed from NOT_SUPPORTED
    Opcode.BZ: (NibbleOpcodeSupport.FFN_DIRECT, None),
    Opcode.BNZ: (NibbleOpcodeSupport.FFN_DIRECT, None),
    Opcode.ADJ: (NibbleOpcodeSupport.FFN_DIRECT, None),

    # Heap (NOW compilable with multi-op)
    Opcode.MALC: (NibbleOpcodeSupport.FFN_DIRECT, None),
    Opcode.FREE: (NibbleOpcodeSupport.FFN_DIRECT, None),
}
```

#### Step 1.5: Test (2 hours)

```python
# test_multi_operation.py

def test_adj():
    """ADJ: SP += imm (simple: just ADD)"""
    compiler = OpcodeNibbleCompiler()
    weights = compiler.compile_opcode(Opcode.ADJ)
    assert weights is not None
    print("✅ ADJ compiled")

def test_bz():
    """BZ: if (AX==0) PC = imm (complex: CMP + ADD + SELECT)"""
    compiler = OpcodeNibbleCompiler()
    weights = compiler.compile_opcode(Opcode.BZ)
    assert weights is not None
    print("✅ BZ compiled")

def test_malc():
    """MALC: malloc with bump allocator (very complex)"""
    compiler = OpcodeNibbleCompiler()
    weights = compiler.compile_opcode(Opcode.MALC)
    assert weights is not None
    print("✅ MALC compiled")
```

**Phase 1 Result**: 24/39 opcodes (62%) ✅

## Phase 2: Multi-Layer Weight Generator

### What It Does

Generates weights for operations that span MULTIPLE transformer layers:
- Layer N (FFN): Setup operation
- Layer N+1 (Attention): Memory/stack access
- Layer N+2 (FFN): Copy result

### File Structure

```
neural_vm/
├── multi_layer_generator.py          (NEW - 200 lines)
│   └── MultiLayerWeightGenerator class
└── opcode_nibble_integration.py      (UPDATE)
    └── Add compile_multilayer() method
```

### Implementation Steps

#### Step 2.1: Create multi_layer_generator.py (3 hours)

```python
"""
Multi-Layer Weight Generator

Generates weights for multi-layer operations (FFN → Attention → FFN).
"""

from typing import Dict
import torch
from .multi_operation_compiler import MultiOperationCompiler
from .full_opcode_mapper import MultiLayerWeights


class MultiLayerWeightGenerator:
    """Generate weights for multi-layer operations."""

    def __init__(self, num_positions: int = 8):
        self.num_positions = num_positions
        self.multi_op_compiler = MultiOperationCompiler(num_positions)

    def generate_weights(
        self,
        multilayer: MultiLayerWeights,
        opcode: int
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Generate weights for each FFN layer.

        Returns: {layer_idx: weights_dict}
                 {layer_idx: None} for attention layers
        """
        layer_weights = {}
        layer_idx = 9  # Start at layer 9 (ALU layer)

        for layer_type, graph in multilayer.layers:
            if layer_type == "ffn":
                # Compile this FFN layer
                if graph is None:
                    # Empty layer (no-op)
                    layer_weights[layer_idx] = None
                else:
                    weights = self.multi_op_compiler.compile_graph(graph, opcode)
                    layer_weights[layer_idx] = weights

                layer_idx += 1

            elif layer_type == "attention":
                # Attention layer - no compilation needed
                # Mark as None so we know to skip it
                layer_weights[layer_idx] = None
                layer_idx += 1

        return layer_weights
```

#### Step 2.2: Update Integration (1 hour)

```python
# In neural_vm/opcode_nibble_integration.py

from .multi_layer_generator import MultiLayerWeightGenerator

class OpcodeNibbleCompiler:
    def __init__(self, num_positions: int = 8):
        self.num_positions = num_positions
        self.nibble_compiler = NibbleWeightCompiler(num_positions)
        self.multi_op_compiler = MultiOperationCompiler(num_positions)
        self.multilayer_gen = MultiLayerWeightGenerator(num_positions)  # NEW

    def compile_multilayer(
        self,
        opcode: int
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Compile multi-layer operation.

        Returns: {layer_idx: weights}
        """
        from .full_opcode_mapper import FullOpcodeMapper

        mapper = FullOpcodeMapper()

        # Get multi-layer graph
        if opcode in [Opcode.JSR, Opcode.ENT]:
            multilayer = mapper.map_opcode_multilayer(opcode, imm=0)
        else:
            multilayer = mapper.map_opcode_multilayer(opcode)

        # Generate weights for each layer
        return self.multilayer_gen.generate_weights(multilayer, opcode)

    def load_multilayer_into_vm(
        self,
        vm,
        weights: Dict[int, Dict[str, torch.Tensor]]
    ):
        """Load multi-layer weights into VM."""
        for layer_idx, layer_weights in weights.items():
            if layer_weights is not None:
                self.load_weights_into_vm(vm, layer_idx, layer_weights)
```

#### Step 2.3: Test (2 hours)

```python
# test_multi_layer.py

def test_li():
    """LI: AX = *AX (3 layers)"""
    compiler = OpcodeNibbleCompiler()
    weights = compiler.compile_multilayer(Opcode.LI)

    assert 9 in weights   # FFN setup layer
    assert 10 in weights  # Attention layer (None)
    assert 11 in weights  # FFN copy layer

    assert weights[9] is not None
    assert weights[10] is None  # Attention
    assert weights[11] is not None

    print("✅ LI compiled (3 layers)")

def test_psh():
    """PSH: push AX (2 layers)"""
    compiler = OpcodeNibbleCompiler()
    weights = compiler.compile_multilayer(Opcode.PSH)

    assert 9 in weights   # FFN setup
    assert 10 in weights  # Attention write

    print("✅ PSH compiled (2 layers)")
```

**Phase 2 Result**: 32/39 opcodes (82%) ✅

## Final Integration

### Update Support Table

```python
# Mark multi-layer opcodes as compilable
_NIBBLE_OPCODE_SUPPORT = {
    # ... existing ...

    # Memory (NOW compilable with multi-layer)
    Opcode.LI: (NibbleOpcodeSupport.MULTILAYER, None),
    Opcode.LC: (NibbleOpcodeSupport.MULTILAYER, None),
    Opcode.SI: (NibbleOpcodeSupport.MULTILAYER, None),
    Opcode.SC: (NibbleOpcodeSupport.MULTILAYER, None),
    Opcode.PSH: (NibbleOpcodeSupport.MULTILAYER, None),
    Opcode.JSR: (NibbleOpcodeSupport.MULTILAYER, None),
    Opcode.ENT: (NibbleOpcodeSupport.MULTILAYER, None),
    Opcode.LEV: (NibbleOpcodeSupport.MULTILAYER, None),
}
```

### Unified Compile Method

```python
def compile_any_opcode(self, opcode: int):
    """Compile any opcode (single, multi-op, or multi-layer)."""
    support = self.get_support_level(opcode)

    if support == "ffn_direct":
        # Single or multi-operation
        return self.compile_opcode(opcode)
    elif support == "multilayer":
        # Multi-layer
        return self.compile_multilayer(opcode)
    else:
        raise ValueError(f"Opcode {opcode} not compilable")
```

## Testing & Validation

### Comprehensive Test Suite

```bash
# Test all 32 autoregressive opcodes
python test_all_32_autoregressive.py

# Expected output:
# ✅ 18 pure FFN (already working)
# ✅ 6 multi-operation (JMP, BZ, BNZ, ADJ, MALC, FREE)
# ✅ 8 multi-layer (LI, LC, SI, SC, PSH, JSR, ENT, LEV)
# Total: 32/39 (82%)
```

## Timeline Summary

| Phase | Days | Opcodes Added | Total Coverage |
|-------|------|---------------|----------------|
| Start | - | 18 | 46% |
| Phase 1 | 1-2 | +6 | 62% |
| Phase 2 | 2-3 | +8 | 82% |
| **Total** | **3-5 days** | **+14** | **82%** |

## What You Get

After implementation:
- ✅ 32/39 opcodes with weight generation (82%)
- ✅ All 34 autoregressive opcodes covered
- ✅ Production-ready compilation system
- ✅ Full C4 VM capability (except external I/O)

The 7 remaining opcodes (OPEN, READ, CLOS, PRTF, EXIT, GETCHAR, PUTCHAR) are external syscalls by design and cannot be autoregressive.
