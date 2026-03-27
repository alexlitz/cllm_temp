# Weight Compiler Design: High-Level Abstractions for Neural VM

## Executive Summary

Current weight-setting code manually manipulates individual matrix elements:
```python
W_up[0, ge.NIB_A] = S
W_up[0, ge.NIB_B] = S
b_up[2] = -S * (base - 1.0)
W_down[ge.RAW_SUM, 0] = 1.0 / (S * S)
```

This is like writing assembly. We need a **compiler** with high-level abstractions:
- **Lookup tables** for attention-based selection
- **Registers** for state storage and relay
- **Operations** for computation (add, step, gate, etc.)
- **Control flow** for conditional execution

---

## Current State: Manual Weight Setting

### Example: ADD Operation (Layer 1)

```python
# Layer 1 of ADD: Compute RAW_SUM = A + B
with torch.no_grad():
    W_up = self.ffn.W_up
    b_up = self.ffn.b_up
    W_gate = self.ffn.W_gate
    W_down = self.ffn.W_down

    # Units 0-1: RAW_SUM = A + B (cancel pair)
    W_up[0, ge.NIB_A] = S
    W_up[0, ge.NIB_B] = S
    W_gate[0, ge.OP_START + opcode] = S
    W_down[ge.RAW_SUM, 0] = 1.0 / (S * S)

    W_up[1, ge.NIB_A] = -S
    W_up[1, ge.NIB_B] = -S
    W_gate[1, ge.OP_START + opcode] = -S
    W_down[ge.RAW_SUM, 1] = 1.0 / (S * S)

    # Units 2-3: G = step(A+B >= base) → CARRY_OUT
    W_up[2, ge.NIB_A] = S
    W_up[2, ge.NIB_B] = S
    b_up[2] = -S * (base - 1.0)
    W_gate[2, ge.OP_START + opcode] = 1.0
    W_down[ge.CARRY_OUT, 2] = 1.0 / S
    # ... 20 more lines ...
```

### Problems

1. **Low-level**: Directly manipulating weight matrices
2. **Error-prone**: Easy to get indices wrong
3. **Hard to read**: Intent obscured by details
4. **Not composable**: Can't easily reuse patterns
5. **No type safety**: No compile-time checks
6. **Manual scaling**: Have to track S factors everywhere

---

## Proposed: Weight Compiler with High-Level Abstractions

### Vision: Declare What, Not How

```python
# High-level description of ADD operation
add_op = Operation("ADD")

# Layer 1: Compute raw sum and carry
with add_op.layer(1):
    # RAW_SUM = A + B (cancel pair for signed addition)
    add_op.compute(
        output="RAW_SUM",
        formula="A + B",
        method="cancel_pair",
        gate="OPCODE_ADD"
    )

    # CARRY_OUT = step(A + B >= base)
    add_op.compute(
        output="CARRY_OUT",
        formula="step(A + B >= base)",
        method="step_pair",
        gate="OPCODE_ADD"
    )

    # PROPAGATE = step(A + B == base-1)
    add_op.compute(
        output="TEMP",
        formula="step(A + B == base-1)",
        method="equality_step",
        gate="OPCODE_ADD"
    )

# Layer 2: Carry lookahead
with add_op.layer(2):
    add_op.carry_lookahead(
        propagate="TEMP",
        generate="CARRY_OUT",
        output="CARRY_IN"
    )

# Layer 3: Final result with mod
with add_op.layer(3):
    add_op.compute(
        output="RESULT",
        formula="(RAW_SUM + CARRY_IN) mod base",
        method="modular_add",
        gate="OPCODE_ADD"
    )
```

---

## Core Abstractions

### 1. Registers (State Variables)

**Purpose**: Named dimensions in the embedding space

```python
class Register:
    """High-level register abstraction."""
    def __init__(self, name: str, index: int, width: int = 1):
        self.name = name
        self.index = index  # Base dimension index
        self.width = width  # Number of dimensions

# Define registers
registers = RegisterSet({
    "NIB_A": Register("NIB_A", 0, width=16),  # One-hot nibble
    "NIB_B": Register("NIB_B", 16, width=16),
    "RAW_SUM": Register("RAW_SUM", 100, width=1),  # Scalar
    "CARRY_IN": Register("CARRY_IN", 101, width=1),
    "CARRY_OUT": Register("CARRY_OUT", 102, width=1),
    "RESULT": Register("RESULT", 110, width=16),
})
```

**Usage**:
```python
# Instead of: W_up[0, 16] = S
compiler.read("NIB_B", into_unit=0)

# Instead of: W_down[100, 5] = 1.0
compiler.write(from_unit=5, to="RAW_SUM")
```

### 2. Operations (Computation Patterns)

**Purpose**: Reusable computation patterns with automatic weight setting

```python
class Operation:
    """High-level operation abstraction."""

    def cancel_pair(self, inputs: List[str], output: str, gate: str):
        """Cancel pair: output = (+input_sum) + (-input_sum)."""
        # Automatically allocates 2 hidden units
        # Sets W_up, W_gate, W_down appropriately
        pass

    def step_pair(self, formula: str, output: str, gate: str):
        """Step pair: output = step(formula >= threshold)."""
        # Automatically allocates 2 hidden units
        # Computes threshold from formula
        # Sets biases correctly
        pass

    def modular_reduce(self, input: str, modulus: int, output: str):
        """Modular reduction: output = input mod modulus."""
        pass

    def lookup(self, query: str, table: str, output: str):
        """Attention-based lookup: output = table[query]."""
        pass
```

**Usage**:
```python
# Instead of 10 lines of weight setting:
op.cancel_pair(
    inputs=["NIB_A", "NIB_B"],
    output="RAW_SUM",
    gate="OPCODE_ADD"
)
```

### 3. Lookup Tables (Attention Patterns)

**Purpose**: Attention-based memory and selection

```python
class LookupTable:
    """Attention-based lookup table."""

    def __init__(self, name: str, key_dim: int, value_dim: int):
        self.name = name
        self.key_dim = key_dim
        self.value_dim = value_dim

    def compile_query(self, query_expr: str) -> AttentionPattern:
        """Compile query expression to Q·K pattern."""
        pass

    def compile_softmax(self, temperature: float, anchor: float) -> SoftmaxConfig:
        """Configure softmax parameters."""
        pass

# Define lookup tables
tables = {
    "CODE_MEMORY": LookupTable(
        name="CODE_MEMORY",
        key_dim=48,  # ADDR_KEY
        value_dim=8   # Byte value
    ),

    "REGISTER_FILE": LookupTable(
        name="REGISTER_FILE",
        key_dim=4,   # Register ID
        value_dim=32  # 32-bit value
    ),
}
```

**Usage**:
```python
# L5: Instruction fetch
compiler.lookup(
    table="CODE_MEMORY",
    query="PC",  # Query with program counter
    output="FETCHED_BYTE",
    softmax=SoftmaxConfig(temperature=20.0)
)

# L15: Memory load
compiler.lookup(
    table="MEMORY",
    query="ADDRESS",
    output="MEM_VALUE",
    softmax=SoftmaxConfig(temperature=15.0, anchor=0.0)
)
```

### 4. Control Flow (Conditional Execution)

**Purpose**: Opcode-based and flag-based gating

```python
class ControlFlow:
    """Control flow abstraction."""

    def when(self, condition: str):
        """Execute when condition is true."""
        return ConditionalContext(condition)

    def switch(self, selector: str, cases: Dict[str, Callable]):
        """Switch statement on selector value."""
        pass

# Usage
with compiler.when("OPCODE == ADD"):
    compiler.compute(...)

with compiler.when("OPCODE in [ADD, SUB, MUL]"):
    compiler.compute(...)

compiler.switch("OPCODE", {
    "ADD": lambda: compiler.add_operation(),
    "SUB": lambda: compiler.sub_operation(),
    "MUL": lambda: compiler.mul_operation(),
})
```

---

## Complete Example: ADD Operation

### Current Low-Level Code (60 lines)

```python
class AddRawAndGenFFN(nn.Module):
    def __init__(self, ge: GenericE, opcode: int):
        super().__init__()
        S = ge.SCALE
        base = ge.BASE
        dim = ge.DIM

        self.ffn = GenericPureFFN(dim, hidden_dim=6)

        with torch.no_grad():
            W_up = self.ffn.W_up
            b_up = self.ffn.b_up
            W_gate = self.ffn.W_gate
            W_down = self.ffn.W_down

            # 40 lines of manual weight setting...
            W_up[0, ge.NIB_A] = S
            W_up[0, ge.NIB_B] = S
            # ...
```

### Proposed High-Level Code (15 lines)

```python
class AddRawAndGenFFN(CompiledFFN):
    def compile(self):
        with self.when("OPCODE == ADD"):
            # Raw sum with cancel pair
            self.cancel_pair(
                inputs=["NIB_A", "NIB_B"],
                output="RAW_SUM"
            )

            # Generate flag: A + B >= base
            self.step(
                condition="NIB_A + NIB_B >= base",
                output="CARRY_OUT"
            )

            # Propagate flag: A + B == base-1
            self.equality_step(
                left="NIB_A + NIB_B",
                right="base - 1",
                output="TEMP"
            )
```

**Benefits**:
- ✅ 4x less code
- ✅ Clearer intent
- ✅ Type-safe register names
- ✅ Automatic scaling
- ✅ Reusable patterns
- ✅ Easier to maintain

---

## Architecture

### Compiler Pipeline

```
High-Level Spec → IR → Weight Allocation → Weight Values → PyTorch Model
```

#### Stage 1: High-Level Specification

```python
spec = OperationSpec("ADD")

spec.layer(1).compute(
    output="RAW_SUM",
    formula="A + B",
    method="cancel_pair"
)
```

#### Stage 2: Intermediate Representation (IR)

```python
IR = [
    ComputeNode(
        layer=1,
        unit_ids=[0, 1],
        inputs=["NIB_A", "NIB_B"],
        output="RAW_SUM",
        method=CancelPairMethod(),
        gate="OPCODE_ADD"
    ),
    # ...
]
```

#### Stage 3: Weight Allocation

```python
allocation = {
    "layer_1": {
        "hidden_units": 6,
        "unit_0": ComputeNode(...),
        "unit_1": ComputeNode(...),
        # ...
    }
}
```

#### Stage 4: Weight Values

```python
weights = {
    "W_up": torch.tensor([[S, S, 0, ...], ...]),
    "b_up": torch.tensor([0, 0, -S*(base-1), ...]),
    "W_gate": torch.tensor([[S, 0, ...], ...]),
    "W_down": torch.tensor([[1/(S*S), 1/(S*S), ...], ...]),
}
```

#### Stage 5: PyTorch Model

```python
model = nn.Module()
model.W_up = nn.Parameter(weights["W_up"])
model.b_up = nn.Parameter(weights["b_up"])
# ...
```

---

## Implementation

### Core Classes

```python
class WeightCompiler:
    """Main compiler class."""

    def __init__(self, config: CompilerConfig):
        self.config = config
        self.registers = RegisterSet()
        self.layers = []
        self.operations = {}

    def operation(self, name: str) -> Operation:
        """Define a new operation."""
        op = Operation(name, self)
        self.operations[name] = op
        return op

    def compile(self) -> nn.Module:
        """Compile to PyTorch model."""
        # Stage 2: Build IR
        ir = self._build_ir()

        # Stage 3: Allocate units
        allocation = self._allocate_units(ir)

        # Stage 4: Compute weight values
        weights = self._compute_weights(allocation)

        # Stage 5: Build PyTorch model
        model = self._build_model(weights)

        return model

class Operation:
    """Operation builder."""

    def __init__(self, name: str, compiler: WeightCompiler):
        self.name = name
        self.compiler = compiler
        self.current_layer = None
        self.nodes = []

    def layer(self, index: int):
        """Enter layer context."""
        return LayerContext(self, index)

    def compute(self, output: str, formula: str, method: str, gate: str):
        """Add computation node."""
        node = ComputeNode(
            layer=self.current_layer,
            output=output,
            formula=formula,
            method=method,
            gate=gate
        )
        self.nodes.append(node)

class ComputeNode:
    """IR node representing a computation."""

    def __init__(self, layer: int, output: str, formula: str,
                 method: str, gate: str):
        self.layer = layer
        self.output = output
        self.formula = formula
        self.method = method
        self.gate = gate

    def allocate_units(self) -> int:
        """Return number of hidden units needed."""
        method_units = {
            "cancel_pair": 2,
            "step_pair": 2,
            "clear_pair": 2,
            "equality_step": 4,
            "modular_add": 6,
        }
        return method_units.get(self.method, 1)

    def compile_weights(self, unit_offset: int, registers: RegisterSet,
                       scale: float) -> WeightValues:
        """Compile this node to weight values."""
        # Delegate to method-specific compiler
        compiler = METHOD_COMPILERS[self.method]
        return compiler.compile(self, unit_offset, registers, scale)
```

### Method Compilers

```python
class CancelPairCompiler:
    """Compiles cancel pair: y = (+sum) + (-sum)."""

    def compile(self, node: ComputeNode, unit_offset: int,
                registers: RegisterSet, S: float) -> WeightValues:
        # Parse formula to get input registers
        inputs = self._parse_inputs(node.formula)
        output_reg = registers[node.output]
        gate_reg = registers[node.gate]

        weights = WeightValues()

        # Unit 0: Positive sum
        for inp_name in inputs:
            inp_reg = registers[inp_name]
            weights.W_up[unit_offset, inp_reg.index] = S
        weights.W_gate[unit_offset, gate_reg.index] = S
        weights.W_down[output_reg.index, unit_offset] = 1.0 / (S * S)

        # Unit 1: Negative sum
        for inp_name in inputs:
            inp_reg = registers[inp_name]
            weights.W_up[unit_offset + 1, inp_reg.index] = -S
        weights.W_gate[unit_offset + 1, gate_reg.index] = -S
        weights.W_down[output_reg.index, unit_offset + 1] = 1.0 / (S * S)

        return weights

class StepPairCompiler:
    """Compiles step pair: y = step(x >= threshold)."""

    def compile(self, node: ComputeNode, unit_offset: int,
                registers: RegisterSet, S: float) -> WeightValues:
        # Parse formula: "A + B >= base" → inputs=[A,B], threshold=base
        inputs, threshold = self._parse_step(node.formula)
        output_reg = registers[node.output]
        gate_reg = registers[node.gate]

        weights = WeightValues()

        # Unit 0: step(sum >= threshold)
        for inp_name in inputs:
            inp_reg = registers[inp_name]
            weights.W_up[unit_offset, inp_reg.index] = S
        weights.b_up[unit_offset] = -S * (threshold - 1.0)
        weights.W_gate[unit_offset, gate_reg.index] = 1.0
        weights.W_down[output_reg.index, unit_offset] = 1.0 / S

        # Unit 1: -step(sum >= threshold+1)
        for inp_name in inputs:
            inp_reg = registers[inp_name]
            weights.W_up[unit_offset + 1, inp_reg.index] = S
        weights.b_up[unit_offset + 1] = -S * threshold
        weights.W_gate[unit_offset + 1, gate_reg.index] = 1.0
        weights.W_down[output_reg.index, unit_offset + 1] = -1.0 / S

        return weights

# Registry of method compilers
METHOD_COMPILERS = {
    "cancel_pair": CancelPairCompiler(),
    "step_pair": StepPairCompiler(),
    "clear_pair": ClearPairCompiler(),
    "equality_step": EqualityStepCompiler(),
    "modular_add": ModularAddCompiler(),
}
```

---

## Usage Examples

### Example 1: Simple Operation

```python
# Create compiler
compiler = WeightCompiler(config)

# Define ADD operation
add = compiler.operation("ADD")

with add.layer(1):
    add.cancel_pair(["NIB_A", "NIB_B"], output="RAW_SUM", gate="OPCODE_ADD")
    add.step("NIB_A + NIB_B >= base", output="CARRY_OUT", gate="OPCODE_ADD")

with add.layer(2):
    add.carry_lookahead("TEMP", "CARRY_OUT", output="CARRY_IN")

with add.layer(3):
    add.modular_add("RAW_SUM", "CARRY_IN", modulus="base", output="RESULT")

# Compile to PyTorch model
model = compiler.compile()
```

### Example 2: Attention-Based Lookup

```python
# Define memory lookup operation
load = compiler.operation("LOAD")

with load.layer(15):
    load.lookup(
        table="MEMORY",
        query={
            "address": "ADDR_KEY",
            "validity": "MEM_STORE"
        },
        output="MEM_VALUE",
        softmax=SoftmaxConfig(
            temperature=15.0,
            anchor=0.0,  # softmax1 for ZFOD
            bias="zfod_offset"
        )
    )
```

### Example 3: Register Relay

```python
# Define PC relay
relay = compiler.operation("PC_RELAY")

with relay.layer(4):
    relay.relay(
        from_marker="REG_PC",
        to_marker="REG_AX",
        value="PC_VALUE",
        condition="NEEDS_FETCH"
    )
```

### Example 4: Conditional Execution

```python
# Different behavior per opcode
alu = compiler.operation("ALU")

with alu.layer(8):
    with alu.when("OPCODE == ADD"):
        alu.cancel_pair(["A", "B"], output="RESULT")

    with alu.when("OPCODE == SUB"):
        alu.cancel_pair(["A", "-B"], output="RESULT")

    with alu.when("OPCODE == MUL"):
        alu.swiglu_multiply("A", "B", output="RESULT")
```

---

## Benefits

### 1. Readability

**Before** (Low-level):
```python
W_up[2, ge.NIB_A] = S
W_up[2, ge.NIB_B] = S
b_up[2] = -S * (base - 1.0)
W_gate[2, ge.OP_START + opcode] = 1.0
W_down[ge.CARRY_OUT, 2] = 1.0 / S
```

**After** (High-level):
```python
self.step("NIB_A + NIB_B >= base", output="CARRY_OUT")
```

### 2. Correctness

- **Type safety**: Register names checked at compile time
- **Automatic scaling**: No manual S factor tracking
- **Verified patterns**: Method compilers tested once
- **Dimension checking**: Catch index errors early

### 3. Maintainability

- **Reusable patterns**: cancel_pair, step_pair used everywhere
- **Clear intent**: What, not how
- **Easy refactoring**: Change method implementation once
- **Self-documenting**: Code reads like specification

### 4. Composability

```python
# Build complex operations from primitives
def multi_byte_add(self, num_bytes: int):
    for i in range(num_bytes):
        self.cancel_pair([f"A_{i}", f"B_{i}"], output=f"SUM_{i}")
        self.step(f"A_{i} + B_{i} + CARRY_{i} >= 256",
                 output=f"CARRY_{i+1}")
```

### 5. Optimization

```python
# Compiler can optimize automatically
compiler.optimize(
    merge_cancel_pairs=True,
    share_units=True,
    sparsify=True,
    compact_moe=True
)
```

---

## Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1-2)

- [ ] `WeightCompiler` base class
- [ ] `RegisterSet` and `Register` classes
- [ ] `Operation` and `LayerContext` builders
- [ ] `ComputeNode` IR representation
- [ ] Weight allocation algorithm

### Phase 2: Method Compilers (Week 3-4)

- [ ] `CancelPairCompiler`
- [ ] `StepPairCompiler`
- [ ] `ClearPairCompiler`
- [ ] `EqualityStepCompiler`
- [ ] `ModularAddCompiler`
- [ ] `RelayCompiler`

### Phase 3: Attention Patterns (Week 5-6)

- [ ] `LookupTable` abstraction
- [ ] `AttentionPattern` compiler
- [ ] `SoftmaxConfig` for softmax1
- [ ] Query/Key/Value weight setting
- [ ] Bias and masking

### Phase 4: Control Flow (Week 7-8)

- [ ] `when()` conditional context
- [ ] `switch()` opcode multiplexing
- [ ] Gate weight management
- [ ] Opcode flag handling

### Phase 5: Integration (Week 9-10)

- [ ] Port ADD operation to new system
- [ ] Port SUB operation
- [ ] Port comparison operations
- [ ] Verify against existing tests

### Phase 6: Advanced Features (Week 11-12)

- [ ] Automatic optimization passes
- [ ] Sparsification
- [ ] MoE partitioning
- [ ] Weight verification
- [ ] Documentation generation

---

## Related Work

### Similar Systems

1. **Halide** - Image processing DSL with schedules
   - Separates algorithm from execution strategy
   - Our system: Separates computation from weight values

2. **TVM/Relay** - Neural network compiler IR
   - High-level ops → optimized kernels
   - Our system: High-level ops → transformer weights

3. **Triton** - GPU programming language
   - Python-like syntax → PTX assembly
   - Our system: Operation specs → weight matrices

4. **MLIR** - Multi-level intermediate representation
   - Progressive lowering through dialects
   - Our system: Spec → IR → Allocation → Weights

---

## Conclusion

### Current State
- ❌ Manual weight setting (assembly-level)
- ❌ Error-prone and hard to maintain
- ❌ Not reusable or composable
- ❌ Obscures intent

### Proposed State
- ✅ High-level abstractions (compiler-level)
- ✅ Type-safe and verified
- ✅ Reusable patterns
- ✅ Clear intent

### Impact
- **4x less code** for typical operations
- **Easier onboarding** for new contributors
- **Fewer bugs** from type safety
- **Faster development** from reusable patterns
- **Better optimization** from compiler passes

The weight compiler transforms neural VM development from low-level assembly programming to high-level systems programming with registers, lookup tables, and operations as first-class abstractions.

---

**Status**: Design proposal
**Next Steps**: Implement Phase 1 (Core Infrastructure)
**Estimated Effort**: 12 weeks for full implementation

