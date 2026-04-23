"""
Weight Setter Tests.

Tests for unified weight setting interface that supports both:
- Hand-set weights (production, set_vm_weights)
- Compiled weights (development, weight compiler)

Both approaches must:
1. Pass the same functional tests
2. Pass purity tests (no Python arithmetic in forward pass)
3. Produce equivalent outputs

These tests verify the infrastructure and will automatically test
both approaches once the compiler is complete.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from neural_vm.weight_setter import (
    WeightMode, set_weights, get_default_mode, set_default_mode,
    with_weight_mode, parametrize_weight_modes, verify_forward_purity,
)
from neural_vm.vm_step import AutoregressiveVM


class TestWeightSetterInterface:
    """Test the weight setter interface."""

    def test_weight_mode_enum(self):
        """WeightMode enum has expected values."""
        assert WeightMode.HAND_SET.value == "hand_set"
        assert WeightMode.COMPILED.value == "compiled"

    def test_default_mode_is_hand_set(self):
        """Default mode is HAND_SET."""
        # Reset to default
        set_default_mode(WeightMode.HAND_SET)
        assert get_default_mode() == WeightMode.HAND_SET

    def test_set_default_mode(self):
        """Can change default mode."""
        original = get_default_mode()
        try:
            set_default_mode(WeightMode.COMPILED)
            assert get_default_mode() == WeightMode.COMPILED
        finally:
            set_default_mode(original)

    def test_with_weight_mode_context(self):
        """Context manager temporarily changes mode."""
        original = get_default_mode()
        set_default_mode(WeightMode.HAND_SET)

        with with_weight_mode(WeightMode.COMPILED):
            assert get_default_mode() == WeightMode.COMPILED

        assert get_default_mode() == WeightMode.HAND_SET
        set_default_mode(original)

    def test_parametrize_weight_modes(self):
        """parametrize_weight_modes returns available modes."""
        modes = parametrize_weight_modes()
        assert WeightMode.HAND_SET in modes
        # COMPILED will be added once implemented


class TestHandSetWeights:
    """Test hand-set weight setting."""

    @pytest.fixture
    def model(self):
        """Create a model for testing."""
        return AutoregressiveVM(
            d_model=512,
            n_layers=17,
            n_heads=8,
            ffn_hidden=4096,
        )

    def test_set_weights_hand_set(self, model):
        """Can set weights using HAND_SET mode."""
        set_weights(model, mode=WeightMode.HAND_SET, verify_purity=False)
        # Model should have non-zero weights
        assert model.embed.embed.weight.abs().sum() > 0

    def test_set_weights_default_mode(self, model):
        """set_weights uses default mode when not specified."""
        set_default_mode(WeightMode.HAND_SET)
        set_weights(model, verify_purity=False)
        assert model.embed.embed.weight.abs().sum() > 0

    def test_purity_verification_passes(self, model):
        """Purity verification passes for hand-set weights."""
        set_weights(model, mode=WeightMode.HAND_SET, verify_purity=True)
        # If we get here without exception, purity check passed

    def test_verify_forward_purity_directly(self, model):
        """verify_forward_purity can be called directly."""
        set_weights(model, mode=WeightMode.HAND_SET, verify_purity=False)
        assert verify_forward_purity(model) == True


class TestCompiledWeights:
    """Test compiled weight setting."""

    @pytest.fixture
    def model(self):
        """Create a model for testing."""
        return AutoregressiveVM(
            d_model=512,
            n_layers=17,
            n_heads=8,
            ffn_hidden=4096,
        )

    def test_compiled_mode_works(self, model):
        """COMPILED mode successfully sets weights."""
        set_weights(model, mode=WeightMode.COMPILED)
        # Verify weights are non-zero
        assert model.blocks[0].ffn.W_up.abs().max().item() > 0

    def test_compiled_weights_purity(self, model):
        """Compiled weights pass purity verification."""
        set_weights(model, mode=WeightMode.COMPILED, verify_purity=True)


class TestPurityEnforcement:
    """Test purity enforcement for both weight modes."""

    @pytest.fixture
    def model(self):
        """Create a model for testing."""
        return AutoregressiveVM(
            d_model=512,
            n_layers=17,
            n_heads=8,
            ffn_hidden=4096,
        )

    def test_purity_check_structure(self, model):
        """Purity check examines forward method."""
        from neural_vm.purity_guard import verify_forward_purity

        set_weights(model, mode=WeightMode.HAND_SET, verify_purity=False)
        result = verify_forward_purity(model)
        assert result == True

    def test_purity_detects_forbidden_patterns(self):
        """Purity check would detect forbidden modifications."""
        from neural_vm.purity_guard import PurityViolationError

        # This is a structural test - the purity guard checks for
        # forbidden patterns like tensor indexing assignments
        # We don't actually modify the forward method here, just
        # verify the check infrastructure exists
        assert PurityViolationError is not None


class TestWeightEquivalence:
    """Tests for verifying weight equivalence between modes.

    These tests will be enabled once compiled weights are implemented.
    They ensure both approaches produce equivalent outputs.
    """

    @pytest.fixture
    def model_hand(self):
        """Create model with hand-set weights."""
        model = AutoregressiveVM(
            d_model=512,
            n_layers=17,
            n_heads=8,
            ffn_hidden=4096,
        )
        set_weights(model, mode=WeightMode.HAND_SET, verify_purity=False)
        return model

    def test_weight_tensors_match(self, model_hand):
        """Hand-set and compiled weights are numerically identical."""
        model_compiled = AutoregressiveVM(
            d_model=512,
            n_layers=17,
            n_heads=8,
            ffn_hidden=4096,
        )
        set_weights(model_compiled, mode=WeightMode.COMPILED, verify_purity=False)

        for i in range(17):
            bh = model_hand.blocks[i]
            bc = model_compiled.blocks[i]
            for name in ['attn.W_q','attn.W_k','attn.W_v','attn.W_o',
                         'ffn.W_up','ffn.b_up','ffn.W_gate','ffn.b_gate','ffn.W_down']:
                parts = name.split('.')
                t_h = getattr(getattr(bh, parts[0]), parts[1])
                t_c = getattr(getattr(bc, parts[0]), parts[1])
                assert torch.allclose(t_h, t_c, atol=1e-3), f"L{i} {name} mismatch"

    def test_embedding_weights_match(self, model_hand):
        """Embedding weights match between modes."""
        model_compiled = AutoregressiveVM(
            d_model=512,
            n_layers=17,
            n_heads=8,
            ffn_hidden=4096,
        )
        set_weights(model_compiled, mode=WeightMode.COMPILED, verify_purity=False)

        hand_embed = model_hand.embed.embed.weight
        compiled_embed = model_compiled.embed.embed.weight

        assert torch.allclose(hand_embed, compiled_embed, rtol=1e-4, atol=1e-6)


class TestFunctionalWithWeightModes:
    """Functional tests that should pass with both weight modes.

    Currently only runs with HAND_SET. Will automatically include
    COMPILED once implemented.
    """

    @pytest.fixture
    def compile_program(self):
        """Fixture to compile C programs."""
        from src.compiler import compile_c
        return compile_c

    @pytest.fixture(params=parametrize_weight_modes())
    def weight_mode(self, request):
        """Parametrized fixture for weight modes."""
        return request.param

    def test_return_42(self, compile_program, weight_mode):
        """Basic program returns correct value with both weight modes."""
        from neural_vm.batch_runner import BatchedSpeculativeRunner

        source = "int main() { return 42; }"
        bytecode, data = compile_program(source)

        # Use the specified weight mode
        with with_weight_mode(weight_mode):
            runner = BatchedSpeculativeRunner(batch_size=1)
            results = runner.run_batch([bytecode], [data], max_steps=500)

        assert results[0][1] == 42

    def test_arithmetic(self, compile_program, weight_mode):
        """Arithmetic works with both weight modes."""
        from neural_vm.batch_runner import BatchedSpeculativeRunner

        test_cases = [
            ("int main() { return 5 + 3; }", 8),
            ("int main() { return 10 - 4; }", 6),
            ("int main() { return 3 * 4; }", 12),
        ]

        with with_weight_mode(weight_mode):
            runner = BatchedSpeculativeRunner(batch_size=1)

            for source, expected in test_cases:
                bytecode, data = compile_program(source)
                results = runner.run_batch([bytecode], [data], max_steps=500)
                assert results[0][1] == expected, f"Failed: {source} with {weight_mode}"


class TestWeightCompilerInfrastructure:
    """Tests for weight compiler infrastructure (partial implementation)."""

    def test_weight_compiler_exists(self):
        """Weight compiler module exists."""
        from neural_vm.weight_compiler import WeightCompiler, ComputeMethod
        assert WeightCompiler is not None
        assert ComputeMethod is not None

    def test_graph_weight_compiler_exists(self):
        """Graph weight compiler module exists."""
        from neural_vm.graph_weight_compiler import OpType
        assert OpType is not None

    def test_cancel_pair_compiler(self):
        """CancelPairCompiler produces valid weights."""
        from neural_vm.weight_compiler import (
            WeightCompiler, ComputeMethod, CancelPairCompiler, ComputeNode
        )

        compiler = WeightCompiler(dim=512, hidden_dim=4096, scale=5.0)
        compiler.registers.add("A", 0, width=1)
        compiler.registers.add("B", 1, width=1)
        compiler.registers.add("OUT", 10, width=1)
        compiler.registers.add("GATE", 100, width=1)

        node = ComputeNode(
            layer=1,
            output="OUT",
            method=ComputeMethod.CANCEL_PAIR,
            inputs=["A", "B"],
            gate="GATE"
        )

        method_compiler = CancelPairCompiler()
        weights = method_compiler.compile(
            node, 0, compiler.registers, 5.0, 512, 4096
        )

        # Verify structure
        assert weights.W_up[0, 0] == 5.0  # Unit 0 reads A with scale S
        assert weights.W_up[0, 1] == 5.0  # Unit 0 reads B with scale S
        assert weights.W_up[1, 0] == -5.0  # Unit 1 reads A with -S
        assert weights.W_up[1, 1] == -5.0  # Unit 1 reads B with -S


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
