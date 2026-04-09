"""
Use new debugging tools to precisely identify AX_CARRY issue.

This script uses the newly implemented debugging infrastructure to:
1. Validate AX_CARRY flow from Layer 3 to Layer 8
2. Visualize exactly where the value is lost
3. Compare neural vs handler execution
4. Inspect layer configurations
"""

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.debugger import VMExecutionTracer
from neural_vm.contracts import validate_and_print
from neural_vm.step_debugger import StepDebugger
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Opcode, _SetDim

BD = _SetDim

print("="*80)
print("INVESTIGATING AX_CARRY PROPAGATION ISSUE")
print("="*80 + "\n")

# Test program
code = "int main() { return 10 + 32; }"
bytecode, data = compile_c(code)

print(f"Test program: {code}\n")
print(f"Expected result: 42")
print(f"Bytecode operations: IMM 10 → PUSH → IMM 32 → ADD\n")

# Step 1: Compare neural vs handler
print("="*80)
print("STEP 1: COMPARE NEURAL VS HANDLER")
print("="*80 + "\n")

runner = AutoregressiveVMRunner()
tracer = VMExecutionTracer(runner, enable_profiling=False)

result = tracer.compare_with_handler(bytecode, Opcode.ADD, max_steps=10)

print(f"\nComparison result: {'PASS' if result['match'] else 'FAIL'}")

if not result['match']:
    print(f"\n⚠️  Neural implementation differs from handler")
    print(f"   Handler output: {result['handler_output']}")
    print(f"   Neural output: {result['neural_output']}")
    print("\nProceeding to detailed investigation...\n")

# Step 2: Trace AX_CARRY flow
print("="*80)
print("STEP 2: TRACE AX_CARRY DATAFLOW")
print("="*80 + "\n")

runner2 = AutoregressiveVMRunner()
if Opcode.ADD in runner2._func_call_handlers:
    del runner2._func_call_handlers[Opcode.ADD]

tracer2 = VMExecutionTracer(runner2, track_dims=['AX_CARRY_LO', 'AX_CARRY_HI', 'ALU_LO', 'ALU_HI'])
trace = tracer2.trace_execution(bytecode, max_steps=10)

# Save trace for later analysis
trace.save('investigation/ax_carry_issue')
print("\nTrace saved to investigation/ax_carry_issue\n")

# Find ADD operation
add_ops = trace.find_operation(Opcode.ADD)
if add_ops:
    add_step = add_ops[0]['step']
    print(f"ADD operation found at step {add_step}\n")
else:
    print("⚠️  ADD operation not found in trace\n")
    add_step = 4  # Assume step 4

# Step 3: Validate AX_CARRY flow
print("="*80)
print("STEP 3: VALIDATE AX_CARRY FLOW")
print("="*80 + "\n")

print("Checking if AX_CARRY propagates from Layer 3 to Layer 8...\n")
success = trace.validate_flow('AX_CARRY_LO', from_layer=3, to_layer=8, min_value=0.5, step=add_step)

# Step 4: Visualize dataflow
print("\n" + "="*80)
print("STEP 4: VISUALIZE DATAFLOW")
print("="*80 + "\n")

print("AX_CARRY_LO dataflow visualization:")
trace.visualize_dataflow('AX_CARRY_LO', from_layer=0, to_layer=15, step=add_step)

print("\nAX_CARRY_HI dataflow visualization:")
trace.visualize_dataflow('AX_CARRY_HI', from_layer=0, to_layer=15, step=add_step)

print("\nALU_LO dataflow visualization:")
trace.visualize_dataflow('ALU_LO', from_layer=0, to_layer=15, step=add_step)

# Step 5: Check dimension history at specific positions
print("\n" + "="*80)
print("STEP 5: DETAILED DIMENSION HISTORY")
print("="*80 + "\n")

print("AX_CARRY_LO across all layers during ADD step:")
trace.show_dimension_history('AX_CARRY_LO', step=add_step)

print("\nALU_LO across all layers during ADD step:")
trace.show_dimension_history('ALU_LO', step=add_step)

# Step 6: Use step debugger for detailed inspection
print("\n" + "="*80)
print("STEP 6: STEP DEBUGGER INSPECTION")
print("="*80 + "\n")

runner3 = AutoregressiveVMRunner()
if Opcode.ADD in runner3._func_call_handlers:
    del runner3._func_call_handlers[Opcode.ADD]

debugger = StepDebugger(runner3, track_dims=['AX_CARRY_LO', 'ALU_LO'])
debugger.add_breakpoint(opcode=Opcode.ADD, description="Break at ADD")
debugger.run_until_break(bytecode, max_steps=10, interactive=False)

print("\nInspecting at breakpoint:")
print("\n--- Layer 3 (should SET AX_CARRY) ---")
debugger.inspect(layer=3, dim='AX_CARRY_LO')

print("\n--- Layer 6 (intermediate) ---")
debugger.inspect(layer=6, dim='AX_CARRY_LO')

print("\n--- Layer 7 (should preserve AX_CARRY) ---")
debugger.inspect(layer=7, dim='AX_CARRY_LO')

print("\n--- Layer 8 (needs AX_CARRY for ADD) ---")
debugger.inspect(layer=8, dim='AX_CARRY_LO')

print("\n--- ALU_LO at Layer 8 ---")
debugger.inspect(layer=8, dim='ALU_LO')

# Step 7: Check contracts
print("\n" + "="*80)
print("STEP 7: CONTRACT VALIDATION")
print("="*80 + "\n")

model = AutoregressiveVM()
set_vm_weights(model)

violations = validate_and_print(model)

# Step 8: Summary and diagnosis
print("\n" + "="*80)
print("DIAGNOSTIC SUMMARY")
print("="*80 + "\n")

print("Findings:")
print("-" * 80)

# Analyze trace to determine issue
if add_ops:
    token = [t for t in trace.token_metadata if t['step'] == add_step][0] if [t for t in trace.token_metadata if t['step'] == add_step] else None

    if token:
        token_idx = token['token_idx']

        # Check each layer
        l3_carry = trace.get_dimension_value(3, token_idx, 'AX_CARRY_LO')
        l7_carry = trace.get_dimension_value(7, token_idx, 'AX_CARRY_LO')
        l8_carry = trace.get_dimension_value(8, token_idx, 'AX_CARRY_LO')
        l8_alu = trace.get_dimension_value(8, token_idx, 'ALU_LO')

        if l3_carry is not None:
            l3_max = float(l3_carry.max()) if hasattr(l3_carry, 'max') else l3_carry
            print(f"1. Layer 3 AX_CARRY_LO: max={l3_max:.3f}")
            if l3_max > 0.5:
                print(f"   ✓ Layer 3 IS setting AX_CARRY")
            else:
                print(f"   ❌ Layer 3 NOT setting AX_CARRY")

        if l7_carry is not None:
            l7_max = float(l7_carry.max()) if hasattr(l7_carry, 'max') else l7_carry
            print(f"2. Layer 7 AX_CARRY_LO: max={l7_max:.3f}")
            if l7_max > 0.5:
                print(f"   ✓ AX_CARRY survives to Layer 7")
            else:
                print(f"   ❌ AX_CARRY lost before Layer 7")

        if l8_carry is not None:
            l8_max = float(l8_carry.max()) if hasattr(l8_carry, 'max') else l8_carry
            print(f"3. Layer 8 AX_CARRY_LO: max={l8_max:.3f}")
            if l8_max > 0.5:
                print(f"   ✓ AX_CARRY reaches Layer 8")
            else:
                print(f"   ❌ AX_CARRY LOST at Layer 8 (THIS IS THE PROBLEM)")

        if l8_alu is not None:
            l8_alu_max = float(l8_alu.max()) if hasattr(l8_alu, 'max') else l8_alu
            print(f"4. Layer 8 ALU_LO: max={l8_alu_max:.3f}")
            if l8_alu_max > 0.5:
                print(f"   ✓ ALU_LO is set")
            else:
                print(f"   ❌ ALU_LO not set")

print("\nContract violations:", len(violations))
for v in violations[:3]:
    print(f"  - {v['message']}")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80 + "\n")

print("Based on this investigation:")
print()
print("1. If AX_CARRY is set by Layer 3 but lost by Layer 8:")
print("   → Check if Layer 7 overwrites AX_CARRY dimensions")
print("   → Check if Layer 8 FFN gates require AX_CARRY at specific position")
print("   → Verify attention patterns in Layer 6-7 preserve AX_CARRY")
print()
print("2. If AX_CARRY is not set by Layer 3:")
print("   → Check Layer 3 Head 1 configuration")
print("   → Verify previous step has AX value to copy")
print()
print("3. If contract violations exist:")
print("   → Fix unauthorized writes to AX_CARRY_LO/HI")
print("   → Ensure Layer 3 Head 1 is configured correctly")
print()
print("Next: Run check_layer7_weights.py and analyze attention patterns")
print()

print("="*80)
print("Investigation complete. Trace saved to investigation/ax_carry_issue")
print("="*80)
