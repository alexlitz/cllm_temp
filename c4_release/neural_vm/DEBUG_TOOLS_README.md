# Neural VM Debugging Tools

## Overview

Two new debugging tools to make neural VM development easier:

1. **Execution Tracer** (`debugger.py`) - Track dimension values through all layers
2. **Contract Validator** (`contracts.py`) - Validate dimension reservations

## Quick Start

### Execution Tracer

Track how dimension values flow through layers:

```python
from neural_vm.debugger import VMExecutionTracer, quick_debug
from src.compiler import compile_c

code = "int main() { return 10 + 32; }"
bytecode, data = compile_c(code)

# Quick debug - shows dimension history across layers
trace = quick_debug(bytecode, dim_to_track='AX_CARRY_LO')

# Or use full API
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode

runner = AutoregressiveVMRunner()
tracer = VMExecutionTracer(runner, track_dims=['AX_CARRY_LO', 'ALU_LO'])
trace = tracer.trace_execution(bytecode, max_steps=10)

# Show how AX_CARRY changes across layers
trace.show_dimension_history('AX_CARRY_LO', step=4)

# Find where ADD executes
add_ops = trace.find_operation(Opcode.ADD)
print(f"ADD at step {add_ops[0]['step']}")

# Get specific value
val = trace.get_dimension_value(layer_idx=8, token_idx=120, dim_name='AX_CARRY_LO')
```

**Output Example:**
```
======================================================================
Dimension History: AX_CARRY_LO
Step: 4
======================================================================

  Layer  0: max=0.000, max=0.000, max=0.000
  Layer  1: max=0.000, max=0.000, max=0.000
  Layer  2: max=0.000, max=0.000, max=0.000
  Layer  3: max=1.000, max=1.000, max=1.000  ← Set here
  Layer  4: max=1.000, max=1.000, max=1.000
  Layer  5: max=1.000, max=1.000, max=1.000
  Layer  6: max=1.000, max=1.000, max=1.000
  Layer  7: max=1.000, max=1.000, max=1.000
  Layer  8: max=0.000, max=0.000, max=0.000  ← Lost here!
```

### Contract Validator

Check if layers respect dimension reservations:

```python
from neural_vm.contracts import DimensionContract, validate_and_print
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights

model = AutoregressiveVM()
set_vm_weights(model)

# Validate and print results
violations = validate_and_print(model)

# Or just validate
violations = DimensionContract.validate_model(model)
if violations:
    print(f"Found {len(violations)} violations!")
    DimensionContract.print_violations_verbose(violations)

# Or just show dimension map
DimensionContract.print_dimension_map(model)
```

**Output Example:**
```
================================================================================
❌ DIMENSION CONTRACT VIOLATIONS DETECTED
================================================================================

ERRORS (1):
  ❌ Layer 7 FFN writes to reserved dimension AX_CARRY_LO

================================================================================

IMPACT:
  AX_CARRY_LO: Previous AX value for binary operations
    → This may cause: Binary operations to fail

================================================================================
DIMENSION USAGE MAP
================================================================================

AX_CARRY_LO (dims 272-287):
  Description: Previous AX value for binary operations
  Reserved: Yes
  Expected writers: Layer3_Attn_Head1
  Actual writers: Layer3_Attn_Head1, Layer7_FFN
  ❌ UNAUTHORIZED: Layer7_FFN
```

## What These Tools Solve

### Before: Manual Investigation

```python
# Had to write custom hooks every time
l3_out = None
def hook(module, input, output):
    global l3_out
    l3_out = output.detach().cpu()

h = model.blocks[3].register_forward_hook(hook)
runner.run(bytecode)
h.remove()

# Then manually inspect
print(l3_out[0, -1, 272:288])  # Is AX_CARRY set?

# Repeat for every layer...
```

### After: Automated Tracing

```python
# Automatically tracks all layers
tracer = VMExecutionTracer(runner, track_dims=['AX_CARRY_LO'])
trace = tracer.trace_execution(bytecode)

# Shows full history
trace.show_dimension_history('AX_CARRY_LO')
```

## Common Debugging Workflows

### 1. Binary Operation Returns Wrong Value

```python
# Trace the operation
from neural_vm.embedding import Opcode

tracer = VMExecutionTracer(runner, track_dims=['AX_CARRY_LO', 'ALU_LO'])
trace = tracer.trace_execution(bytecode)

# Find where ADD executes
add_ops = trace.find_operation(Opcode.ADD)
step = add_ops[0]['step']

# Check both operands at Layer 8 (where ADD computes)
trace.show_dimension_history('ALU_LO', step=step)
trace.show_dimension_history('AX_CARRY_LO', step=step)

# If one is missing, trace backwards to find where it was lost
```

### 2. Contract Violation Warning

```
CONTRACT: READ-BEFORE-WRITE: L6_ffn_io reads 'AX_CARRY_LO' but no prior layer writes it
```

```python
# Validate configuration
violations = DimensionContract.validate_model(model)
DimensionContract.print_dimension_map(model)

# Output shows:
#   AX_CARRY_LO:
#     Expected writers: Layer3_Attn_Head1
#     Actual writers: (none)  ← Layer 3 not configured!
```

### 3. Value Lost Between Layers

```python
# Track dimension through all layers
trace.show_dimension_history('AX_CARRY_LO')

# Output:
#   Layer 3: max=1.000  ✓
#   Layer 4: max=1.000  ✓
#   Layer 5: max=0.000  ✗ Lost here!

# Check if Layer 5 FFN writes to AX_CARRY (it shouldn't)
violations = DimensionContract.validate_model(model)
# Shows: Layer5_FFN writes to reserved dimension AX_CARRY_LO
```

## Advanced Features

### Save and Load Traces

```python
# Save trace for later analysis
trace.save('traces/add_operation')

# Load trace
from neural_vm.debugger import ExecutionTrace
trace = ExecutionTrace.load('traces/add_operation')

# Analyze offline
trace.show_dimension_history('AX_CARRY_LO')
```

### Performance Profiling

```python
# Enable profiling (enabled by default)
tracer = VMExecutionTracer(runner, enable_profiling=True)
trace = tracer.trace_execution(bytecode)

# Show performance summary
trace.performance_summary()
# Output:
#   Total execution time: 2.345s
#   Forward passes: 142
#   Tokens generated: 142
#   Average time per token: 16.51ms
#   Time per layer:
#     Layer  0: 146.56ms (6.2%)
#     Layer  1: 146.56ms (6.2%)
#     ...
```

### Dataflow Visualization

```python
# Show how dimension flows between layers
trace.visualize_dataflow('AX_CARRY_LO', from_layer=3, to_layer=8, step=4)

# Output:
#   Dataflow: AX_CARRY_LO from Layer 3 to Layer 8
#   Step: 4
#
#   Tracking token 105 (step 4):
#
#     Layer  3: max=1.000, mean=0.125 ← SET
#     Layer  4: max=1.000, mean=0.125
#     Layer  5: max=1.000, mean=0.125
#     Layer  6: max=1.000, mean=0.125
#     Layer  7: max=1.000, mean=0.125
#     Layer  8: max=0.000, mean=0.000 ← LOST
```

### Validate Dataflow

```python
# Validate that dimension propagates correctly
success = trace.validate_flow('AX_CARRY_LO', from_layer=3, to_layer=8, min_value=0.5)

# Output:
#   ❌ AX_CARRY_LO value drops to 0.000 at Layer 8
#      → Lost at Layer 8
```

### Compare with Handler

```python
# Compare neural implementation with handler
tracer = VMExecutionTracer(runner)
result = tracer.compare_with_handler(bytecode, Opcode.ADD, max_steps=10)

# Output:
#   Comparing Neural vs Handler Execution
#
#   Running with handler...
#   Running without handler (neural only)...
#
#   Results:
#     Handler output: 42
#     Neural output:  10
#     Handler exit code: 0
#     Neural exit code:  0
#
#   ❌ Outputs differ! Neural implementation has issues.
#      Output mismatch: 42 vs 10

# Check result programmatically
if result['match']:
    print("Neural implementation works!")
else:
    print(f"Expected: {result['handler_output']}, Got: {result['neural_output']}")
```

## Step-Level Debugger

For interactive debugging:

```python
from neural_vm.step_debugger import StepDebugger

debugger = StepDebugger(runner)

# Set breakpoints
debugger.add_breakpoint(opcode=Opcode.ADD)
debugger.add_breakpoint(step=5)

# Run until breakpoint
debugger.run_until_break(bytecode, max_steps=10)

# Inspect at breakpoint
debugger.inspect(layer=3, dim='AX_CARRY_LO')
debugger.inspect(layer=8, dim='AX_CARRY_LO')

# Compare layers
debugger.compare_layers(3, 8, 'AX_CARRY_LO')

# Show dimension across all layers
debugger.show_dimension_across_layers('AX_CARRY_LO')
```

**Example Output:**
```
Breakpoint 0 added: Opcode ADD
Running until breakpoint (max_steps=10)...

🔴 Breakpoint hit: Opcode ADD

Interactive Debugger - Step 4, Token 105

Current state:
  Step: 4
  Token: 105
  Token ID: 152
  Is marker: False
  Opcode: 26

AX_CARRY_LO at Layer 3:
  Max: 1.000
  Mean: 0.125

AX_CARRY_LO at Layer 8:
  Max: 0.000
  Mean: 0.000

Comparing AX_CARRY_LO: Layer 3 vs Layer 8
  Layer 3: max=1.000, mean=0.125
  Layer 8: max=0.000, mean=0.000
  Max diff: 1.000
  Mean diff: 0.125
```

## Automated Tests

Run dimension dataflow tests:

```bash
pytest neural_vm/tests/test_dimension_dataflow.py -v
```

**Tests include:**
- AX_CARRY propagation during ADD
- ALU_LO set by Layer 7
- Dimension flow validation
- Contract violation detection
- Tracer functionality
- Save/load traces
- Performance tracking

## Completed Features

- [x] Interactive breakpoints
- [x] Automatic diff with handler execution
- [x] Performance profiling
- [x] Save/load traces
- [x] Regression testing integration
- [x] Dataflow validation
- [x] Step-level debugger

## Remaining Limitations

Full production version would add:

- [ ] Attention pattern visualization
- [ ] HTML/GUI output
- [ ] True interactive REPL (current debugger shows state but doesn't accept commands)
- [ ] Real-time breakpoints during execution (current version captures all, then checks)
- [ ] Conditional breakpoints with complex expressions

## Next Steps

See `docs/DEBUGGING_IMPROVEMENTS.md` for:
- Full feature roadmap
- Architectural documentation suggestions
- Testing infrastructure
- Long-term vision (Neural VM Studio GUI)

## Contributing

To add new dimension contracts:

```python
# In contracts.py, add to CONTRACTS dict:
CONTRACTS = {
    'MY_DIM': {
        'description': 'What this dimension does',
        'writers': ['Layer5_FFN'],  # Who should write
        'readers': ['Layer10_FFN'],  # Who reads
        'reserved': True,  # Enforce exclusivity
        'dim_start': BD.MY_DIM,
        'dim_count': 16,
    },
}
```

To track more dimensions in tracer:

```python
tracer = VMExecutionTracer(runner, track_dims=[
    'AX_CARRY_LO',
    'ALU_LO',
    'OUTPUT_LO',  # Add any dimension from _SetDim
])
```
