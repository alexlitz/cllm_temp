# Making Neural VM Easier to Debug

**Date**: 2026-04-07
**Context**: Debugging arithmetic handlers revealed major tooling gaps

## Problems Encountered During Investigation

### 1. No Visibility Into Execution
- Had to write custom hooks for every layer
- No way to see intermediate values without manual instrumentation
- Couldn't tell which forward pass corresponded to which operation
- 280 forward passes during execution - which one is ADD?

### 2. Methodology Traps
- Initially examined only first forward pass (prefix), missed generated tokens
- No clear way to distinguish prefix vs generation
- No markers for "this is the ADD step"
- Hard to map token positions to step structure

### 3. Missing Diagnostic Tools
- No dimension value tracker (follow AX_CARRY through all layers)
- No diff tool (compare working handler vs neural execution)
- No contract violation prominently displayed
- No automated dataflow tracing

### 4. Architecture Opacity
- Unclear which layer is responsible for what
- No explicit dimension reservations documented
- Can't easily validate "does Layer X write to dimension Y?"
- No self-diagnosing weight configurations

## Recommended Improvements

### Priority 1: Execution Tracer (CRITICAL)

**Create `neural_vm/debugger.py`** with:

```python
class VMExecutionTracer:
    """Trace neural VM execution with full introspection."""

    def __init__(self, runner, track_dims=None):
        """
        Args:
            runner: AutoregressiveVMRunner instance
            track_dims: List of dimension names to track (e.g., ['AX_CARRY_LO', 'ALU_LO'])
        """
        self.runner = runner
        self.track_dims = track_dims or []
        self.layer_outputs = {i: [] for i in range(16)}
        self.token_metadata = []  # (token_idx, token_id, step_num, is_marker)

    def trace_execution(self, bytecode, max_steps=10):
        """Run VM with full tracing enabled."""
        # Hook ALL layers
        # Record token metadata (which step, which opcode)
        # Track specified dimensions through all layers
        # Return structured trace

    def get_dimension_history(self, dim_name, step=None):
        """Get value history for a dimension across all layers."""
        # Return: [(layer_idx, position, value), ...]

    def find_operation(self, opcode):
        """Find which forward passes executed a specific opcode."""
        # Return: [(token_idx, step_num, layer_outputs), ...]

    def compare_with_handler(self, opcode):
        """Compare neural execution with handler execution."""
        # Run with handler, run without, show differences

    def visualize_dataflow(self, start_dim, end_dim):
        """Show how a value flows from one dimension to another."""
        # Trace AX_CARRY_LO from Layer 3 → Layer 8
```

**Usage**:
```python
tracer = VMExecutionTracer(runner, track_dims=['AX_CARRY_LO', 'ALU_LO'])
trace = tracer.trace_execution(bytecode)

# Find ADD operation
add_trace = tracer.find_operation(Opcode.ADD)

# Check if AX_CARRY reaches ADD
ax_carry_history = tracer.get_dimension_history('AX_CARRY_LO', step=4)
print(f"AX_CARRY at Layer 8 during ADD: {ax_carry_history[8]}")

# Compare with handler
diff = tracer.compare_with_handler(Opcode.ADD)
print(f"Neural result: {diff['neural']}, Handler result: {diff['handler']}")
```

### Priority 2: Dimension Contract System

**Create `neural_vm/contracts.py`**:

```python
class DimensionContract:
    """Track which layers read/write which dimensions."""

    # Define contracts
    CONTRACTS = {
        'AX_CARRY_LO': {
            'writers': ['Layer3_Attn_Head1'],  # Only Layer 3 should write
            'readers': ['Layer8_FFN', 'Layer9_FFN'],  # Who needs to read
            'reserved': True,  # No other layer should touch
        },
        'ALU_LO': {
            'writers': ['Layer7_Attn_Head0', 'Layer7_Attn_Head1'],
            'readers': ['Layer8_FFN'],
            'reserved': False,  # Can be overwritten
        },
    }

    @classmethod
    def validate_model(cls, model):
        """Check if model weights violate contracts."""
        violations = []

        for dim_name, contract in cls.CONTRACTS.items():
            dim_range = getattr(_SetDim, dim_name)

            # Check for unauthorized writers
            for layer_idx in range(16):
                layer = model.blocks[layer_idx]

                # Check attention W_o
                writes_attn = (layer.attn.W_o[dim_range:dim_range+16, :].abs() > 1e-6).any()
                if writes_attn and f"Layer{layer_idx}_Attn" not in contract['writers']:
                    violations.append(f"Layer {layer_idx} attn writes to {dim_name} (unauthorized)")

                # Check FFN W_down
                writes_ffn = (layer.ffn.W_down[dim_range:dim_range+16, :].abs() > 1e-6).any()
                if writes_ffn and f"Layer{layer_idx}_FFN" not in contract['writers']:
                    violations.append(f"Layer {layer_idx} FFN writes to {dim_name} (unauthorized)")

        return violations

    @classmethod
    def print_dimension_map(cls, model):
        """Print which layers read/write each dimension."""
        # Generate comprehensive map of dimension usage
```

**Usage**:
```python
# At model initialization
violations = DimensionContract.validate_model(model)
if violations:
    print("⚠️ CONTRACT VIOLATIONS:")
    for v in violations:
        print(f"  - {v}")

# During debugging
DimensionContract.print_dimension_map(model)
# Output:
# AX_CARRY_LO (dims 272-287):
#   Written by: Layer 3 Attn Head 1
#   Read by: Layer 8 FFN (ADD/SUB), Layer 9 FFN
#   ❌ Also written by: Layer 7 FFN (VIOLATION)
```

### Priority 3: Step-Level Debugger

**Create `neural_vm/step_debugger.py`**:

```python
class StepDebugger:
    """Interactive debugger that stops at each VM step."""

    def __init__(self, runner):
        self.runner = runner
        self.breakpoints = []  # [(step_num, condition), ...]

    def add_breakpoint(self, step=None, opcode=None, condition=None):
        """Add breakpoint at specific step or when opcode executes."""
        self.breakpoints.append({'step': step, 'opcode': opcode, 'condition': condition})

    def run_until_break(self, bytecode):
        """Run until breakpoint hit, then drop into interactive mode."""
        # At each step:
        #   - Check breakpoints
        #   - If hit, show current state and enter interactive mode

    def inspect(self, layer_idx, dim_name):
        """Inspect dimension value at current step."""

    def continue_execution(self):
        """Resume execution."""

    def compare_layers(self, layer1, layer2, dim_name):
        """Compare dimension value between two layers."""
```

**Usage**:
```python
debugger = StepDebugger(runner)
debugger.add_breakpoint(opcode=Opcode.ADD)
debugger.run_until_break(bytecode)

# Interactive mode:
>>> debugger.inspect(layer=3, dim='AX_CARRY_LO')
AX_CARRY_LO at Layer 3: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0]
>>> debugger.inspect(layer=8, dim='AX_CARRY_LO')
AX_CARRY_LO at Layer 8: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
>>> # Aha! Lost between Layer 3 and Layer 8
```

### Priority 4: Automated Test Suite

**Create `neural_vm/tests/test_dimensions.py`**:

```python
class TestDimensionDataflow:
    """Test that dimensions propagate correctly."""

    def test_ax_carry_propagation(self):
        """AX_CARRY should propagate from Layer 3 to Layer 8."""
        runner = AutoregressiveVMRunner()
        bytecode = compile_c("int main() { return 10 + 32; }")

        tracer = VMExecutionTracer(runner, track_dims=['AX_CARRY_LO'])
        trace = tracer.trace_execution(bytecode)

        # At ADD step
        add_step = tracer.find_operation(Opcode.ADD)[0]

        # Check Layer 3 sets AX_CARRY
        l3_ax_carry = trace.get_value(layer=3, step=add_step, dim='AX_CARRY_LO')
        assert l3_ax_carry.max() > 0.5, "Layer 3 should set AX_CARRY"

        # Check Layer 8 receives AX_CARRY
        l8_ax_carry = trace.get_value(layer=8, step=add_step, dim='AX_CARRY_LO')
        assert l8_ax_carry.max() > 0.5, "Layer 8 should receive AX_CARRY"

    def test_no_unauthorized_writes(self):
        """No layer should write to reserved dimensions."""
        model = AutoregressiveVM()
        set_vm_weights(model)

        violations = DimensionContract.validate_model(model)
        assert len(violations) == 0, f"Contract violations: {violations}"
```

### Priority 5: Visual Debugging Tools

**Create `neural_vm/visualization.py`**:

```python
def plot_dimension_flow(trace, dim_name, step):
    """Plot how a dimension's value changes across layers."""
    import matplotlib.pyplot as plt

    values = [trace.get_value(layer=i, step=step, dim=dim_name).max()
              for i in range(16)]

    plt.plot(values, marker='o')
    plt.xlabel('Layer')
    plt.ylabel(f'{dim_name} max value')
    plt.title(f'{dim_name} flow during step {step}')
    plt.grid(True)
    plt.show()

def plot_attention_pattern(trace, layer, head, step):
    """Visualize attention pattern for a specific head."""
    # Show which positions each query attends to

def plot_step_timeline(trace):
    """Show timeline of all steps with opcodes."""
    # Visual representation of execution:
    # Step 0: ENT
    # Step 1: IMM (10)
    # Step 2: PUSH
    # Step 3: IMM (32)
    # Step 4: ADD ← Click to inspect
```

### Priority 6: Better Error Messages

**Enhance contract violation warnings**:

```python
# Instead of:
# CONTRACT: READ-BEFORE-WRITE: L6_ffn_io reads 'AX_CARRY_LO' but no prior layer writes it

# Show:
╔══════════════════════════════════════════════════════════════╗
║ ⚠️  CONTRACT VIOLATION DETECTED                              ║
╠══════════════════════════════════════════════════════════════╣
║ Layer 6 FFN attempts to read AX_CARRY_LO                    ║
║ But no layer before Layer 6 writes to this dimension!       ║
║                                                              ║
║ Expected writer: Layer 3 Attention Head 1                   ║
║ Actual writers: (none found)                                ║
║                                                              ║
║ This will cause: Binary operations to fail (missing operand)║
║ Suggested fix: Check Layer 3 attention head 1 configuration ║
╚══════════════════════════════════════════════════════════════╝
```

### Priority 7: Architecture Documentation

**Create `neural_vm/docs/ARCHITECTURE_MAP.md`**:

```markdown
# Neural VM Architecture Map

## Layer Responsibilities

### Layer 0-1: Positional Encoding
- **Layer 0**: Step boundary detection, position thresholds
- **Layer 1**: Byte index encoding (L1H0/L1H1/L1H2)
- **Outputs**: Threshold distances for each marker type

### Layer 3: Register Carry-Forward
- **Head 0**: PC carry (prev PC → current PC)
- **Head 1**: AX carry (prev AX → AX_CARRY staging) ← CRITICAL FOR BINARY OPS
- **Head 2**: SP carry
- **Head 3**: BP carry
- **Head 4**: STACK0 carry
- **Outputs**: AX_CARRY_LO/HI (used by Layer 8 for ADD/SUB)

### Layer 7: Operand Gathering
- **Head 0**: Stack → ALU (first operand)
- **Head 1**: BP → ALU (for LEA)
- **Requires**: AX_CARRY from Layer 3 to still be present
- **Outputs**: ALU_LO/HI

### Layer 8: ALU Operations
- **FFN Units**: Implement ADD, SUB, bitwise ops
- **Requires**: ALU_LO (from Layer 7) + AX_CARRY_LO (from Layer 3)
- **Outputs**: OUTPUT_LO/HI (result)

## Dimension Reservations

| Dimension | Reserved | Writer | Readers | Purpose |
|-----------|----------|--------|---------|---------|
| AX_CARRY_LO | ✓ | L3 Attn H1 | L8 FFN, L9 FFN | Binary op second operand |
| ALU_LO | ✗ | L7 Attn H0/H1 | L8 FFN | Binary op first operand |
| OUTPUT_LO | ✗ | L8 FFN, L9 FFN | L15 routing | Operation result |

## Critical Data Flows

### Binary Operation (ADD/SUB/etc.)
```
Step N-1: IMM 10
  → Layer 3: AX_CARRY = (nothing yet)

Step N: IMM 32
  → Layer 3 Head 1: Copies prev AX (10) → AX_CARRY_LO/HI
  → AX_CARRY now holds 10

Step N+1: ADD
  → Layer 7 Head 0: Copies STACK0 (32) → ALU_LO/HI
  → Layer 8 FFN: Reads ALU_LO (32) + AX_CARRY_LO (10) → OUTPUT = 42

CRITICAL: AX_CARRY must survive Layers 3→4→5→6→7 to reach Layer 8!
```
```

## Implementation Priority

1. **Week 1**: Execution Tracer + Dimension Contract System
   - Most critical for debugging
   - Catches issues automatically

2. **Week 2**: Step Debugger + Test Suite
   - Makes investigation interactive
   - Prevents regressions

3. **Week 3**: Visualization + Documentation
   - Makes understanding easier
   - Onboards new developers

## Quick Wins (Can Implement Today)

### 1. Add dimension tracking to existing code

```python
# In vm_step.py, add after each layer:
if self._debug_mode:
    self._track_dimension('AX_CARRY_LO', layer_idx, x)
```

### 2. Better contract warnings

```python
# In weight_setter.py:
def validate_contracts_verbose(model):
    violations = DimensionContract.validate_model(model)
    if violations:
        print("\n" + "="*70)
        print("⚠️  DIMENSION CONTRACT VIOLATIONS DETECTED")
        print("="*70)
        for v in violations:
            print(f"  ❌ {v}")
        print("="*70 + "\n")
```

### 3. Execution summary

```python
# In run_vm.py, after execution:
if self._verbose:
    print(f"\nExecution Summary:")
    print(f"  Steps executed: {step_count}")
    print(f"  Forward passes: {len(layer_outputs)}")
    print(f"  Last opcode: {last_opcode_name}")
    print(f"  Contract violations: {len(violations)}")
```

## Long-term Vision

### Neural VM Studio (GUI Tool)

- **Timeline View**: See all steps, click to inspect
- **Layer Inspector**: Real-time dimension values
- **Dataflow Graph**: Visual representation of how values flow
- **Breakpoint System**: Stop at specific operations
- **Diff View**: Compare working vs broken execution side-by-side
- **Hot Reload**: Modify weights and see immediate effect

Would make debugging feel like using Chrome DevTools instead of `print()` statements!

---

## Conclusion

The core issue is **lack of introspection**. We're flying blind through 16 layers, 280 forward passes, 512 dimensions. The solutions above provide:

1. **Visibility**: See what's happening at each layer
2. **Validation**: Automatically catch configuration errors
3. **Debugging**: Interactive tools to investigate issues
4. **Testing**: Prevent regressions
5. **Documentation**: Understand the architecture

**Recommended first step**: Implement the Execution Tracer. It would have caught the AX_CARRY issue immediately:

```python
tracer = VMExecutionTracer(runner, track_dims=['AX_CARRY_LO'])
trace = tracer.trace_execution(bytecode)
trace.validate_flow('AX_CARRY_LO', from_layer=3, to_layer=8)
# Output: ❌ AX_CARRY_LO value drops to 0 at Layer X
```
