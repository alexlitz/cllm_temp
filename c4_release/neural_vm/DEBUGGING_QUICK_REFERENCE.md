# Neural VM Debugging - Quick Reference

**TL;DR**: Fast lookup for debugging commands and workflows.

## One-Liners

```python
# Quick debug any dimension
from neural_vm.debugger import quick_debug
trace = quick_debug(bytecode, opcode_to_find=Opcode.ADD, dim_to_track='AX_CARRY_LO')

# Validate contracts
from neural_vm.contracts import validate_and_print
validate_and_print(model)

# Compare neural vs handler
tracer = VMExecutionTracer(runner)
result = tracer.compare_with_handler(bytecode, Opcode.ADD)
```

## Common Tasks

### "Why is ADD returning wrong value?"

```python
# 1. Compare with handler
tracer = VMExecutionTracer(runner)
result = tracer.compare_with_handler(bytecode, Opcode.ADD)
# Shows: Neural: 10, Handler: 42

# 2. Trace AX_CARRY flow
tracer = VMExecutionTracer(runner, track_dims=['AX_CARRY_LO', 'ALU_LO'])
trace = tracer.trace_execution(bytecode)
trace.validate_flow('AX_CARRY_LO', from_layer=3, to_layer=8)
# Shows: ❌ AX_CARRY_LO value drops to 0.000 at Layer 8 → Lost at Layer 8

# 3. Visualize where it's lost
trace.visualize_dataflow('AX_CARRY_LO', from_layer=3, to_layer=8, step=4)
# Shows exact layer where value disappears
```

### "Which layer is breaking dimension X?"

```python
# Validate flow
trace.validate_flow('DIMENSION_NAME', from_layer=START, to_layer=END)

# Or visualize step-by-step
trace.visualize_dataflow('DIMENSION_NAME', from_layer=START, to_layer=END)
```

### "Is Layer X writing to dimension Y?"

```python
# Check contracts
from neural_vm.contracts import DimensionContract
violations = DimensionContract.validate_model(model)
DimensionContract.print_dimension_map(model)

# Or manually check
from neural_vm.vm_step import _SetDim
BD = _SetDim
layer_x = model.blocks[X]
w_o = layer_x.attn.W_o[BD.DIMENSION_Y:BD.DIMENSION_Y+16, :]
has_writes = (w_o.abs() > 1e-6).any().item()
```

### "What happens during opcode X?"

```python
# Set breakpoint
debugger = StepDebugger(runner)
debugger.add_breakpoint(opcode=Opcode.X)
debugger.run_until_break(bytecode)

# Inspect at breakpoint
debugger.inspect(layer=8, dim='OUTPUT_LO')
debugger.show_dimension_across_layers('AX_CARRY_LO')
```

### "How fast is execution?"

```python
# Profile
tracer = VMExecutionTracer(runner, enable_profiling=True)
trace = tracer.trace_execution(bytecode)
trace.performance_summary()
```

### "Save this trace for later"

```python
# Save
trace.save('investigation/issue_name')

# Load later
from neural_vm.debugger import ExecutionTrace
trace = ExecutionTrace.load('investigation/issue_name')
```

## Typical Workflows

### Workflow 1: Debugging New Opcode

```python
# 1. Compare with handler
result = tracer.compare_with_handler(bytecode, Opcode.NEW_OP)

if not result['match']:
    # 2. Trace relevant dimensions
    trace = tracer.trace_execution(bytecode)

    # 3. Find where opcode executes
    ops = trace.find_operation(Opcode.NEW_OP)
    step = ops[0]['step']

    # 4. Check inputs/outputs at that step
    trace.show_dimension_history('INPUT_DIM', step=step)
    trace.show_dimension_history('OUTPUT_DIM', step=step)

    # 5. Validate dataflow
    trace.validate_flow('INPUT_DIM', from_layer=X, to_layer=Y, step=step)
```

### Workflow 2: Contract Violation Investigation

```python
# 1. Check violations
violations = DimensionContract.validate_model(model)

if violations:
    # 2. See detailed map
    DimensionContract.print_dimension_map(model)

    # 3. Trace to verify runtime behavior
    trace = tracer.trace_execution(bytecode)
    trace.visualize_dataflow('VIOLATED_DIM', from_layer=0, to_layer=15)
```

### Workflow 3: Performance Debugging

```python
# 1. Profile execution
tracer = VMExecutionTracer(runner, enable_profiling=True)
trace = tracer.trace_execution(bytecode, max_steps=100)

# 2. Show summary
trace.performance_summary()

# 3. Identify bottlenecks
# (Look for layers with high time percentage)
```

## API Reference

### VMExecutionTracer

```python
tracer = VMExecutionTracer(
    runner,                              # AutoregressiveVMRunner
    track_dims=['AX_CARRY_LO'],         # Dimensions to track
    enable_profiling=True                # Track performance
)

trace = tracer.trace_execution(bytecode, max_steps=10)
result = tracer.compare_with_handler(bytecode, opcode, max_steps=10)
```

### ExecutionTrace

```python
# Get values
val = trace.get_dimension_value(layer_idx, token_idx, dim_name)

# Display
trace.show_dimension_history(dim_name, step=None)
trace.visualize_dataflow(dim_name, from_layer, to_layer, step=None)
trace.performance_summary()

# Validate
success = trace.validate_flow(dim_name, from_layer, to_layer, min_value=0.5, step=None)

# Find
ops = trace.find_operation(opcode)

# Persist
trace.save(path)
trace = ExecutionTrace.load(path)
```

### StepDebugger

```python
debugger = StepDebugger(runner, track_dims=['AX_CARRY_LO'])

# Breakpoints
bp_id = debugger.add_breakpoint(step=5)
bp_id = debugger.add_breakpoint(opcode=Opcode.ADD)
bp_id = debugger.add_breakpoint(condition=lambda t,s,i: custom_check())
debugger.remove_breakpoint(bp_id)
debugger.list_breakpoints()

# Run
debugger.run_until_break(bytecode, max_steps=10, interactive=False)

# Inspect
debugger.inspect(layer, dim)
debugger.compare_layers(layer1, layer2, dim_name)
debugger.show_dimension_across_layers(dim_name)
```

### DimensionContract

```python
# Validate
violations = DimensionContract.validate_model(model)

# Display
DimensionContract.print_violations_verbose(violations)
DimensionContract.print_dimension_map(model)

# Convenience
from neural_vm.contracts import validate_and_print
violations = validate_and_print(model)
```

## Dimension Names

Common dimensions to track:

- `AX_CARRY_LO` / `AX_CARRY_HI` - Previous AX for binary ops
- `ALU_LO` / `ALU_HI` - First operand for ALU
- `OUTPUT_LO` / `OUTPUT_HI` - Operation result
- `CLEAN_EMBED_LO` / `CLEAN_EMBED_HI` - Clean byte values
- `PC_BYTE0` through `PC_BYTE3` - PC bytes
- `AX_BYTE0` through `AX_BYTE3` - AX bytes
- `SP_BYTE0` through `SP_BYTE3` - SP bytes

(See `neural_vm/vm_step.py` `_SetDim` class for full list)

## Layer Responsibilities

Quick reference (see docs for full details):

- **L0-L1**: Positional encoding, step boundaries
- **L3**: Register carry-forward (AX_CARRY!)
- **L6-L7**: Operand gathering (ALU setup)
- **L8**: ALU operations (ADD, SUB, etc.)
- **L9**: Secondary ALU (MUL, DIV, etc.)
- **L10**: Memory operations
- **L15**: Output routing

## Tips

1. **Start with compare_with_handler**: Quickly shows if neural implementation works
2. **Use validate_flow for dataflow issues**: Pinpoints exact layer where values are lost
3. **Save traces**: Don't re-run slow executions, save and analyze offline
4. **Check contracts first**: Many issues are configuration problems, not runtime bugs
5. **Use step debugger for complex bugs**: Set breakpoint at problematic opcode, inspect state

## Examples

See `demo_debugging_tools.py` for comprehensive examples of all features.

Run demos:
```bash
# All demos
python demo_debugging_tools.py

# Specific demo
python demo_debugging_tools.py 1  # Execution tracer
python demo_debugging_tools.py 2  # Load trace
python demo_debugging_tools.py 3  # Contract validation
python demo_debugging_tools.py 4  # Handler comparison
python demo_debugging_tools.py 5  # Step debugger
python demo_debugging_tools.py 6  # Find operations
```

## Testing

Run automated tests:
```bash
pytest neural_vm/tests/test_dimension_dataflow.py -v -s
```

## Help

Full documentation:
- `neural_vm/DEBUG_TOOLS_README.md` - Complete usage guide
- `docs/DEBUGGING_IMPROVEMENTS.md` - Full roadmap and design
- `docs/DEBUGGING_IMPLEMENTATION_SUMMARY.md` - Implementation details
