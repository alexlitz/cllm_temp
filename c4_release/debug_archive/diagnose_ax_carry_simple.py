"""
Simple focused diagnostic for AX_CARRY issue.
Uses new debugging tools but only loads model once.
"""

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.debugger import VMExecutionTracer
from neural_vm.vm_step import Opcode

print("Simple AX_CARRY Diagnostic")
print("="*80 + "\n")

code = "int main() { return 10 + 32; }"
bytecode, data = compile_c(code)

print(f"Test: {code}")
print(f"Expected: 42\n")

# Single run with tracing
runner = AutoregressiveVMRunner()

# Remove ADD handler
if Opcode.ADD in runner._func_call_handlers:
    del runner._func_call_handlers[Opcode.ADD]
    print("Removed ADD handler - using neural implementation\n")

print("Tracing execution...")
tracer = VMExecutionTracer(runner, track_dims=['AX_CARRY_LO', 'ALU_LO'], enable_profiling=False)
trace = tracer.trace_execution(bytecode, max_steps=10)

print("\nValidating AX_CARRY flow from Layer 3 to Layer 8...")
success = trace.validate_flow('AX_CARRY_LO', from_layer=3, to_layer=8, min_value=0.5, step=4)

print("\nVisualizing dataflow:")
trace.visualize_dataflow('AX_CARRY_LO', from_layer=3, to_layer=8, step=4)

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if success:
    print("\n✓ AX_CARRY propagates correctly")
    print("  Issue must be elsewhere (Layer 8 FFN gates or ALU_LO)")
else:
    print("\n❌ AX_CARRY is lost between Layer 3 and Layer 8")
    print("  This is why ADD returns wrong value")
    print("  Next: Check Layer 7 attention configuration")

import torch
torch.cuda.empty_cache()
