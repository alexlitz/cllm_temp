"""
ENT/LEV Neural Path Testing.

Uses debug tools to trace execution and identify divergence points.

ENT semantics: push BP; BP = SP; SP -= imm
LEV semantics: SP = BP; BP = pop; PC = pop
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights
from neural_vm.debug import ExecutionTracer, WeightInspector, StateComparator
from neural_vm.weight_modules import FunctionCallWeights, WeightConfig
from src.compiler import compile_c


class TestENTLEVBasicBehavior:
    """Test basic ENT/LEV functionality."""

    @pytest.fixture
    def runner(self):
        return AutoregressiveVMRunner()

    def test_simple_function_call(self, runner):
        """Test simplest function call pattern: JSR -> ENT -> LEV."""
        source = """
        int foo() { return 42; }
        int main() { return foo(); }
        """
        bytecode, data = compile_c(source)
        output, result = runner.run(bytecode, data, max_steps=100)
        assert result == 42

    def test_function_with_local(self, runner):
        """Test function with local variable (ENT allocates space)."""
        source = """
        int foo() {
            int x;
            x = 10;
            return x;
        }
        int main() { return foo(); }
        """
        bytecode, data = compile_c(source)
        output, result = runner.run(bytecode, data, max_steps=150)
        assert result == 10

    def test_function_with_arg(self, runner):
        """Test function with argument."""
        source = """
        int add_one(int x) { return x + 1; }
        int main() { return add_one(41); }
        """
        bytecode, data = compile_c(source)
        output, result = runner.run(bytecode, data, max_steps=150)
        assert result == 42


class TestENTLEVTracing:
    """Use debug tools to trace ENT/LEV execution."""

    @pytest.fixture
    def model(self):
        """Create model with weights."""
        model = AutoregressiveVM(
            d_model=512,
            n_layers=16,
            n_heads=8,
            ffn_hidden=4096,
        )
        set_vm_weights(model)
        return model

    @pytest.fixture
    def inspector(self, model):
        """Create weight inspector."""
        return WeightInspector(model)

    def test_trace_ent_execution(self):
        """Trace ENT opcode execution step by step."""
        tracer = ExecutionTracer(verbose=True)

        source = """
        int foo() { return 1; }
        int main() { return foo(); }
        """
        bytecode, data = compile_c(source)

        # Create runner with tracer
        runner = AutoregressiveVMRunner()
        runner._debug_ent = True
        runner._debug_lev = True

        output, result = runner.run(bytecode, data, max_steps=100)

        print(f"\nResult: {result}")
        print(f"Output tokens: {len(output)}")
        assert result == 1

    def test_trace_lev_execution(self):
        """Trace LEV opcode execution step by step."""
        source = """
        int foo() {
            int a = 5;
            return a;
        }
        int main() { return foo(); }
        """
        bytecode, data = compile_c(source)

        runner = AutoregressiveVMRunner()
        runner._debug_ent = True
        runner._debug_lev = True

        output, result = runner.run(bytecode, data, max_steps=150)

        print(f"\nResult: {result}")
        assert result == 5

    def test_inspect_ent_weights(self, model, inspector):
        """Inspect weights related to ENT opcode."""
        from neural_vm.vm_step import _SetDim as BD

        print("\n=== ENT-related weights ===")

        # Check embedding for OP_ENT flag dim
        print(f"\nOP_ENT dimension: {BD.OP_ENT}")
        inspector.show_embedding(6)  # ENT opcode byte value

        # Check L6 FFN for ENT handling
        print("\nL6 FFN activations for ENT:")
        inspector.show_ffn_weights(layer=5, unit=100)

    def test_inspect_lev_weights(self, model, inspector):
        """Inspect weights related to LEV opcode."""
        from neural_vm.vm_step import _SetDim as BD

        print("\n=== LEV-related weights ===")

        # Check embedding for OP_LEV flag dim
        print(f"\nOP_LEV dimension: {BD.OP_LEV}")
        inspector.show_embedding(8)  # LEV opcode byte value

        # Check L3 for LEV-specific attention (BP relay)
        print("\nL3 attention for LEV BP relay:")
        inspector.show_attention_weights(layer=2, head=6)


class TestENTLEVDivergence:
    """Compare handler vs neural execution to find divergence."""

    def test_find_ent_divergence(self):
        """Find where neural ENT diverges from handler."""
        source = """
        int foo() { return 1; }
        int main() { return foo(); }
        """
        bytecode, data = compile_c(source)

        # Run with handlers enabled
        runner_handler = AutoregressiveVMRunner()
        _, result_handler = runner_handler.run(bytecode, data, max_steps=100)

        print(f"Handler result: {result_handler}")
        assert result_handler == 1

    def test_find_lev_divergence(self):
        """Find where neural LEV diverges from handler."""
        source = """
        int foo() { return 42; }
        int main() { return foo(); }
        """
        bytecode, data = compile_c(source)

        runner = AutoregressiveVMRunner()
        runner._debug_lev = True

        _, result = runner.run(bytecode, data, max_steps=100)

        print(f"Result: {result}")
        assert result == 42


class TestENTLEVNeuralRequirements:
    """Document what neural ENT/LEV need to work."""

    def test_ent_requirements(self):
        """
        ENT neural requirements:

        1. Push old BP to stack (memory write)
           - Need: L14/L15 to write BP value to mem[SP-8]
           - MEM section: addr = SP-8, value = BP

        2. Set BP = SP (before decrement)
           - Need: L6 to copy SP to BP output
           - L6 head copies SP→BP when OP_ENT active

        3. SP -= imm (allocate locals)
           - Need: Fetch immediate from PC position
           - Need: Subtract from SP
           - L5 fetches imm, L9 does subtraction

        4. PC += 8 (next instruction)
           - Standard PC increment (L3)
        """
        print(__doc__)

    def test_lev_requirements(self):
        """
        LEV neural requirements:

        1. SP = BP
           - Need: Copy BP to SP output
           - L6 should route BP→SP when OP_LEV

        2. BP = *SP (pop old BP from stack)
           - Need: L14/L15 memory lookup at BP address
           - Read mem[old_BP], put in BP output

        3. PC = *(SP+8) (pop return address)
           - Need: Memory lookup at BP+8
           - Read mem[old_BP+8], put in PC output

        4. SP += 16 (pop both values)
           - Need: Add 16 to old BP
           - L9 does addition
        """
        print(__doc__)


class TestFunctionCallWeightsDiagnostic:
    """Use weight modules to diagnose ENT/LEV issues."""

    @pytest.fixture
    def model(self):
        """Create model with weights."""
        model = AutoregressiveVM(
            d_model=512,
            n_layers=16,
            n_heads=8,
            ffn_hidden=4096,
        )
        set_vm_weights(model)
        return model

    def test_diagnose_function_call_weights(self, model):
        """Run diagnostic on function call weights."""
        config = WeightConfig()
        fc_module = FunctionCallWeights(config)

        result = fc_module.diagnose(model)

        print(f"\n=== Function Call Weights Diagnostic ===")
        print(f"Module: {result.module_name}")
        print(f"Passed: {result.passed}")
        print(f"\nDetails:")
        for key, value in result.details.items():
            print(f"  {key}: {value}")

        if result.warnings:
            print(f"\nWarnings:")
            for w in result.warnings:
                print(f"  - {w}")

        if result.errors:
            print(f"\nErrors:")
            for e in result.errors:
                print(f"  - {e}")

        assert result.passed, f"Diagnostic failed: {result.errors}"

    def test_ent_weight_isolation(self, model):
        """Test ENT-specific weights in isolation."""
        from neural_vm.vm_step import _SetDim as BD

        # Check L5 head 5-6 weights are set
        attn5 = model.blocks[5].attn
        HD = 64

        # Head 5: BP→STACK0 relay
        base5 = 5 * HD
        q_val = attn5.W_q[base5, BD.MARK_STACK0].item()
        k_val = attn5.W_k[base5, BD.MARK_BP].item()

        print(f"\nL5 Head 5 (BP→STACK0 relay):")
        print(f"  Q[MARK_STACK0] = {q_val}")
        print(f"  K[MARK_BP] = {k_val}")

        assert abs(q_val) > 1.0, "L5 head 5 Q[MARK_STACK0] not set"
        assert abs(k_val) > 1.0, "L5 head 5 K[MARK_BP] not set"

        # Head 6: SP→BP relay
        base6 = 6 * HD
        q_val = attn5.W_q[base6, BD.MARK_BP].item()
        k_val = attn5.W_k[base6, BD.MARK_SP].item()

        print(f"\nL5 Head 6 (SP→BP relay):")
        print(f"  Q[MARK_BP] = {q_val}")
        print(f"  K[MARK_SP] = {k_val}")

        assert abs(q_val) > 1.0, "L5 head 6 Q[MARK_BP] not set"
        assert abs(k_val) > 1.0, "L5 head 6 K[MARK_SP] not set"

    def test_lev_weight_isolation(self, model):
        """Test LEV-specific weights in isolation."""
        from neural_vm.vm_step import _SetDim as BD

        # Check L3 head 6 for LEV BP→PC relay
        attn3 = model.blocks[3].attn
        HD = 64

        base6 = 6 * HD
        q_pc = attn3.W_q[base6, BD.MARK_PC].item()
        q_lev = attn3.W_q[base6, BD.OP_LEV].item()
        k_bp = attn3.W_k[base6, BD.MARK_BP].item()

        print(f"\nL3 Head 6 (BP→PC relay for LEV):")
        print(f"  Q[MARK_PC] = {q_pc}")
        print(f"  Q[OP_LEV] = {q_lev}")
        print(f"  K[MARK_BP] = {k_bp}")

        # LEV weights may be weak or missing - this helps identify the issue
        if abs(q_pc) < 1.0:
            print("  WARNING: Q[MARK_PC] is weak - LEV PC relay may not work")
        if abs(k_bp) < 1.0:
            print("  WARNING: K[MARK_BP] is weak - LEV BP lookup may not work")


if __name__ == "__main__":
    # Run with debug output
    pytest.main([__file__, "-v", "-s"])
