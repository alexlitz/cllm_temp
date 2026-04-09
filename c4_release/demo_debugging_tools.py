"""
Comprehensive demonstration of Neural VM debugging tools.

This script showcases:
1. Execution Tracer with advanced features
2. Dimension Contract Validator
3. Step-Level Debugger
4. Performance profiling
5. Handler comparison
6. Dataflow visualization
"""

import sys
from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.debugger import VMExecutionTracer, ExecutionTrace
from neural_vm.contracts import validate_and_print
from neural_vm.step_debugger import StepDebugger
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Opcode


def demo_execution_tracer():
    """Demonstrate execution tracer with advanced features."""
    print("\n" + "="*80)
    print("DEMO 1: EXECUTION TRACER")
    print("="*80 + "\n")

    code = "int main() { return 10 + 32; }"
    bytecode, data = compile_c(code)

    print(f"Tracing execution of: {code}\n")

    runner = AutoregressiveVMRunner()

    # Remove ADD handler to trace neural execution
    if Opcode.ADD in runner._func_call_handlers:
        del runner._func_call_handlers[Opcode.ADD]

    tracer = VMExecutionTracer(runner, track_dims=['AX_CARRY_LO', 'ALU_LO'], enable_profiling=True)
    trace = tracer.trace_execution(bytecode, max_steps=10)

    print("\n--- Dimension History: AX_CARRY_LO ---")
    trace.show_dimension_history('AX_CARRY_LO', step=4)

    print("\n--- Dataflow Visualization ---")
    trace.visualize_dataflow('AX_CARRY_LO', from_layer=3, to_layer=8, step=4)

    print("\n--- Validate Flow ---")
    success = trace.validate_flow('AX_CARRY_LO', from_layer=3, to_layer=8, min_value=0.5, step=4)

    print("\n--- Performance Summary ---")
    trace.performance_summary()

    # Save trace for later
    print("\n--- Saving trace ---")
    trace.save('traces/demo_add_operation')

    return trace


def demo_load_trace():
    """Demonstrate loading a saved trace."""
    print("\n" + "="*80)
    print("DEMO 2: LOAD SAVED TRACE")
    print("="*80 + "\n")

    try:
        trace = ExecutionTrace.load('traces/demo_add_operation')
        print("Trace loaded successfully!")
        print(f"  Tokens: {len(trace.token_metadata)}")
        print(f"  Exit code: {trace.exit_code}")
        print(f"  Total time: {trace.performance.total_time:.3f}s")
    except FileNotFoundError:
        print("No saved trace found. Run demo_execution_tracer() first.")


def demo_contract_validation():
    """Demonstrate dimension contract validation."""
    print("\n" + "="*80)
    print("DEMO 3: DIMENSION CONTRACT VALIDATION")
    print("="*80 + "\n")

    model = AutoregressiveVM()
    set_vm_weights(model)

    print("Validating dimension contracts...\n")
    violations = validate_and_print(model)

    if violations:
        print(f"\n⚠️  Found {len(violations)} contract violations")
    else:
        print("\n✓ No contract violations found")

    return violations


def demo_handler_comparison():
    """Demonstrate comparing neural vs handler execution."""
    print("\n" + "="*80)
    print("DEMO 4: HANDLER COMPARISON")
    print("="*80 + "\n")

    code = "int main() { return 10 + 32; }"
    bytecode, data = compile_c(code)

    print(f"Comparing neural vs handler for: {code}\n")

    runner = AutoregressiveVMRunner()
    tracer = VMExecutionTracer(runner)

    result = tracer.compare_with_handler(bytecode, Opcode.ADD, max_steps=10)

    if result['match']:
        print("\n✓ SUCCESS: Neural implementation matches handler!")
    else:
        print(f"\n❌ FAILURE: Neural implementation differs from handler")
        print(f"   Expected: {result['handler_output']}")
        print(f"   Got: {result['neural_output']}")

    return result


def demo_step_debugger():
    """Demonstrate step-level debugger."""
    print("\n" + "="*80)
    print("DEMO 5: STEP-LEVEL DEBUGGER")
    print("="*80 + "\n")

    code = "int main() { return 10 + 32; }"
    bytecode, data = compile_c(code)

    print(f"Debugging: {code}\n")

    runner = AutoregressiveVMRunner()

    # Remove ADD handler
    if Opcode.ADD in runner._func_call_handlers:
        del runner._func_call_handlers[Opcode.ADD]

    debugger = StepDebugger(runner, track_dims=['AX_CARRY_LO', 'ALU_LO'])

    # Set breakpoint at ADD
    debugger.add_breakpoint(opcode=Opcode.ADD, description="Break at ADD operation")

    print("\nRunning until breakpoint...\n")
    debugger.run_until_break(bytecode, max_steps=10, interactive=False)

    print("\n--- Inspecting AX_CARRY_LO at Layer 3 ---")
    debugger.inspect(layer=3, dim='AX_CARRY_LO')

    print("\n--- Inspecting AX_CARRY_LO at Layer 8 ---")
    debugger.inspect(layer=8, dim='AX_CARRY_LO')

    print("\n--- Comparing Layer 3 vs Layer 8 ---")
    debugger.compare_layers(3, 8, 'AX_CARRY_LO')

    print("\n--- AX_CARRY_LO across all layers ---")
    debugger.show_dimension_across_layers('AX_CARRY_LO')

    return debugger


def demo_find_operations():
    """Demonstrate finding specific operations in trace."""
    print("\n" + "="*80)
    print("DEMO 6: FINDING OPERATIONS")
    print("="*80 + "\n")

    code = "int main() { int a = 10; int b = 32; return a + b; }"
    bytecode, data = compile_c(code)

    print(f"Finding operations in: {code}\n")

    runner = AutoregressiveVMRunner()
    tracer = VMExecutionTracer(runner)
    trace = tracer.trace_execution(bytecode, max_steps=20)

    # Find all ADD operations
    add_ops = trace.find_operation(Opcode.ADD)
    print(f"Found {len(add_ops)} ADD operations:")
    for i, op in enumerate(add_ops):
        print(f"  {i+1}. Token {op['token_idx']}, Step {op['step']}")

    # Find IMM operations
    imm_ops = trace.find_operation(Opcode.IMM)
    print(f"\nFound {len(imm_ops)} IMM operations:")
    for i, op in enumerate(imm_ops[:5]):  # Show first 5
        print(f"  {i+1}. Token {op['token_idx']}, Step {op['step']}")

    return trace


def demo_all():
    """Run all demos in sequence."""
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + "  NEURAL VM DEBUGGING TOOLS - COMPREHENSIVE DEMONSTRATION".center(78) + "#")
    print("#" + " "*78 + "#")
    print("#"*80)

    try:
        # Demo 1: Execution Tracer
        trace = demo_execution_tracer()

        # Demo 2: Load Trace
        demo_load_trace()

        # Demo 3: Contract Validation
        violations = demo_contract_validation()

        # Demo 4: Handler Comparison
        result = demo_handler_comparison()

        # Demo 5: Step Debugger
        debugger = demo_step_debugger()

        # Demo 6: Find Operations
        trace2 = demo_find_operations()

        print("\n" + "#"*80)
        print("#" + " "*78 + "#")
        print("#" + "  DEMONSTRATION COMPLETE".center(78) + "#")
        print("#" + " "*78 + "#")
        print("#"*80 + "\n")

        print("Summary:")
        print(f"  ✓ Execution tracer: {len(trace.token_metadata)} tokens traced")
        print(f"  ✓ Contract validation: {len(violations)} violations found")
        print(f"  ✓ Handler comparison: {'PASS' if result['match'] else 'FAIL'}")
        print(f"  ✓ Step debugger: Breakpoint system working")
        print(f"  ✓ Operation finder: {len(trace2.token_metadata)} tokens analyzed")

        print("\nAll debugging tools demonstrated successfully!")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import torch

    # Suppress some warnings
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)

    # Check if specific demo requested
    if len(sys.argv) > 1:
        demo_name = sys.argv[1]
        demos = {
            '1': demo_execution_tracer,
            '2': demo_load_trace,
            '3': demo_contract_validation,
            '4': demo_handler_comparison,
            '5': demo_step_debugger,
            '6': demo_find_operations,
            'all': demo_all,
        }

        if demo_name in demos:
            demos[demo_name]()
        else:
            print(f"Unknown demo: {demo_name}")
            print("Available demos: 1, 2, 3, 4, 5, 6, all")
    else:
        # Run all demos
        demo_all()

    # Cleanup
    torch.cuda.empty_cache()
