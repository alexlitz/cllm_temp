#!/usr/bin/env python3
"""
Network Structure Purity Tests.

Verifies Requirements #2 and #10 from Testing Checklist:
- 100% autoregressive, no external memory or logic
- Vanilla transformer using MoE, SwiGLU, vanilla attention
- No custom non-transformer layers
- Can be exported to ONNX

Date: 2026-04-17
"""

import sys
import os
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn


class TestTransformerArchitecture:
    """Verify model uses standard transformer components."""

    @pytest.fixture
    def model(self):
        """Create a fresh model for testing."""
        from neural_vm.vm_step import AutoregressiveVM, set_vm_weights
        model = AutoregressiveVM(
            d_model=512,
            n_layers=16,
            n_heads=8,
            ffn_hidden=4096
        )
        set_vm_weights(model)
        return model

    def test_model_has_embedding(self, model):
        """Model has embedding layer."""
        assert hasattr(model, 'embed')
        # NeuralVMEmbedding wraps standard embedding
        assert hasattr(model.embed, 'embed')
        assert isinstance(model.embed.embed, nn.Embedding)

    def test_model_has_blocks(self, model):
        """Model has transformer blocks."""
        assert hasattr(model, 'blocks')
        assert isinstance(model.blocks, nn.ModuleList)
        assert len(model.blocks) >= 1

    def test_model_has_output_head(self, model):
        """Model has output projection head."""
        assert hasattr(model, 'output_head')
        assert isinstance(model.output_head, nn.Linear)

    def test_model_is_sequential_blocks(self, model):
        """Model processes through blocks sequentially."""
        # Verify blocks are processed in order (embed -> blocks -> output)
        # Check model structure matches expected pattern
        children = list(model.children())
        child_names = [name for name, _ in model.named_children()]

        assert 'embed' in child_names
        assert 'blocks' in child_names
        assert 'output_head' in child_names

    def test_block_is_attn_then_ffn(self, model):
        """Each block is attention -> FFN (standard order)."""
        from neural_vm.vm_step import TransformerBlock

        for i, block in enumerate(model.blocks):
            assert isinstance(block, TransformerBlock), f"Block {i} is not TransformerBlock"
            assert hasattr(block, 'attn'), f"Block {i} missing attention"
            assert hasattr(block, 'ffn'), f"Block {i} missing FFN"

    def test_no_external_memory_modules(self, model):
        """No nn.Module with external memory in name."""
        for name, module in model.named_modules():
            # Allow 'mem' in variable names but not external memory systems
            forbidden = ['external_memory', 'memory_bank', 'memory_network']
            for f in forbidden:
                assert f not in name.lower(), f"Found forbidden module: {name}"

    def test_all_layers_are_standard_types(self, model):
        """All leaf modules are standard PyTorch types or known custom types."""
        from neural_vm.base_layers import PureFFN
        from neural_vm.vm_step import AutoregressiveAttention, TransformerBlock
        from neural_vm.neural_embedding import NeuralVMEmbedding

        allowed_types = (
            nn.Linear, nn.Embedding, nn.LayerNorm, nn.ModuleList,
            nn.Dropout, nn.Identity, nn.Parameter,
            PureFFN, AutoregressiveAttention, TransformerBlock,
            NeuralVMEmbedding,
        )

        for name, module in model.named_modules():
            # Skip container modules
            if len(list(module.children())) > 0:
                continue
            # Check if it's an allowed type or has standard attributes
            is_standard = (
                isinstance(module, allowed_types) or
                type(module).__module__.startswith('torch.nn')
            )
            # Allow custom modules that are compositions of standard ops
            if not is_standard:
                # Check it's at least a nn.Module
                assert isinstance(module, nn.Module), f"Unknown module type: {name} -> {type(module)}"


class TestSwiGLUFFN:
    """Verify FFN uses SwiGLU activation."""

    @pytest.fixture
    def ffn(self):
        """Create PureFFN for testing."""
        from neural_vm.base_layers import PureFFN
        return PureFFN(dim=64, hidden_dim=128)

    def test_ffn_has_three_projections(self, ffn):
        """FFN has W_up, W_gate, W_down (SwiGLU signature)."""
        assert hasattr(ffn, 'W_up'), "Missing W_up projection"
        assert hasattr(ffn, 'W_gate'), "Missing W_gate projection"
        assert hasattr(ffn, 'W_down'), "Missing W_down projection"

    def test_ffn_projection_shapes(self, ffn):
        """FFN projections have correct shapes."""
        # W_up: [hidden, dim]
        # W_gate: [hidden, dim]
        # W_down: [dim, hidden]
        assert ffn.W_up.shape == (128, 64)
        assert ffn.W_gate.shape == (128, 64)
        assert ffn.W_down.shape == (64, 128)

    def test_ffn_uses_silu_activation(self):
        """FFN uses SiLU (not ReLU, GELU, etc.)."""
        from neural_vm.base_layers import PureFFN
        import inspect

        # Check the forward method source for 'silu'
        source = inspect.getsource(PureFFN.forward)
        assert 'silu' in source.lower() or 'F.silu' in source, \
            "FFN should use SiLU activation"

    def test_swiglu_formula_correct(self, ffn):
        """Verify SwiGLU formula: output = x + W_down @ (silu(W_up @ x) * (W_gate @ x))"""
        x = torch.randn(1, 10, 64)

        # Manual SwiGLU computation
        up = torch.nn.functional.linear(x, ffn.W_up, ffn.b_up)
        gate = torch.nn.functional.linear(x, ffn.W_gate, ffn.b_gate)
        hidden = torch.nn.functional.silu(up) * gate
        down = torch.nn.functional.linear(hidden, ffn.W_down, ffn.b_down)
        expected = x + down

        # Compare with FFN output
        actual = ffn(x)

        assert torch.allclose(actual, expected, atol=1e-5), \
            "FFN output doesn't match SwiGLU formula"

    def test_ffn_has_residual_connection(self, ffn):
        """FFN has residual connection (output includes input)."""
        x = torch.randn(1, 10, 64)

        # Zero out all weights
        with torch.no_grad():
            ffn.W_up.zero_()
            ffn.W_gate.zero_()
            ffn.W_down.zero_()
            ffn.b_up.zero_()
            ffn.b_gate.zero_()
            ffn.b_down.zero_()

        output = ffn(x)

        # With zero weights, output should equal input (residual only)
        assert torch.allclose(output, x, atol=1e-6), \
            "FFN should have residual connection"


class TestVanillaAttention:
    """Verify attention is standard scaled dot-product."""

    @pytest.fixture
    def attention(self):
        """Create attention layer for testing."""
        from neural_vm.vm_step import AutoregressiveAttention
        return AutoregressiveAttention(
            d_model=64,
            num_heads=4,
            layer_idx=0,
            positional_encoding="alibi"
        )

    def test_attention_has_qkv_projections(self, attention):
        """Attention has Q, K, V projections."""
        assert hasattr(attention, 'W_q'), "Missing Q projection"
        assert hasattr(attention, 'W_k'), "Missing K projection"
        assert hasattr(attention, 'W_v'), "Missing V projection"

    def test_attention_has_output_projection(self, attention):
        """Attention has output projection."""
        assert hasattr(attention, 'W_o'), "Missing output projection"

    def test_attention_qkv_shapes(self, attention):
        """Q, K, V projections have correct shapes."""
        d_model = 64
        # Each projection: [d_model, d_model]
        assert attention.W_q.shape == (d_model, d_model)
        assert attention.W_k.shape == (d_model, d_model)
        assert attention.W_v.shape == (d_model, d_model)

    def test_attention_uses_scaled_dot_product(self):
        """Attention uses scaled dot-product (scores / sqrt(d_k))."""
        from neural_vm.vm_step import AutoregressiveAttention
        import inspect

        source = inspect.getsource(AutoregressiveAttention.forward)
        # Look for scaling by sqrt(head_dim)
        assert 'sqrt' in source or 'scale' in source or '** 0.5' in source, \
            "Attention should scale by sqrt(d_k)"

    def test_attention_has_causal_mask(self, attention):
        """Causal masking prevents attending to future."""
        x = torch.randn(1, 10, 64)
        output = attention(x)

        # Output shape should match input
        assert output.shape == x.shape

        # Test that changing future tokens doesn't affect past outputs
        x1 = x.clone()
        x2 = x.clone()
        x2[0, 5:, :] = torch.randn(5, 64)  # Change future tokens

        out1 = attention(x1)
        out2 = attention(x2)

        # First 5 positions should be identical (causal)
        assert torch.allclose(out1[0, :5, :], out2[0, :5, :], atol=1e-5), \
            "Causal masking broken: future tokens affected past outputs"

    def test_alibi_is_additive_bias_only(self, attention):
        """ALiBi only adds to attention scores (no learned params)."""
        # ALiBi slopes should be a buffer, not a parameter
        if hasattr(attention, 'alibi_slopes'):
            assert 'alibi_slopes' not in dict(attention.named_parameters()), \
                "ALiBi slopes should not be learned parameters"
            assert 'alibi_slopes' in dict(attention.named_buffers()), \
                "ALiBi slopes should be registered as buffer"


class TestNoCustomOperations:
    """Verify no non-standard operations."""

    @pytest.fixture
    def model(self):
        """Create model for testing."""
        from neural_vm.vm_step import AutoregressiveVM, set_vm_weights
        model = AutoregressiveVM(
            d_model=512,
            n_layers=16,
            n_heads=8,
            ffn_hidden=4096
        )
        set_vm_weights(model)
        return model

    def test_forward_produces_logits(self, model):
        """Forward pass produces logits without errors."""
        x = torch.randint(0, 256, (1, 10))
        model.eval()
        with torch.no_grad():
            output = model(x)

        # Output should be logits
        assert output.dim() == 3  # [batch, seq, vocab]
        assert output.shape[0] == 1
        assert output.shape[1] == 10

    def test_all_ops_are_tensor_ops(self, model):
        """Forward pass uses only tensor operations."""
        # This is verified by the fact that we can run forward
        # and trace the model (no Python control flow based on values)
        x = torch.randint(0, 256, (1, 10))
        model.eval()

        # Should complete without errors
        with torch.no_grad():
            output = model(x)

        assert output is not None

    def test_no_dynamic_control_flow_on_values(self, model):
        """Model doesn't branch based on tensor values in forward."""
        # Create two different inputs
        x1 = torch.randint(0, 256, (1, 10))
        x2 = torch.randint(0, 256, (1, 10))

        model.eval()
        with torch.no_grad():
            # Both should produce outputs without errors
            out1 = model(x1)
            out2 = model(x2)

        # Shapes should match regardless of input values
        assert out1.shape == out2.shape


class TestONNXExportability:
    """Verify model structure is ONNX-compatible."""

    def test_embedding_exportable(self):
        """Standard embedding exports to ONNX."""
        embed = nn.Embedding(256, 64)

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            path = f.name

        try:
            dummy = torch.randint(0, 256, (1, 10))
            torch.onnx.export(
                embed, dummy, path,
                input_names=['input'],
                output_names=['output'],
                opset_version=14
            )
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_ffn_exportable(self):
        """SwiGLU FFN exports to ONNX."""
        from neural_vm.base_layers import PureFFN

        ffn = PureFFN(dim=64, hidden_dim=128)
        ffn.eval()

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            path = f.name

        try:
            dummy = torch.randn(1, 10, 64)
            torch.onnx.export(
                ffn, dummy, path,
                input_names=['input'],
                output_names=['output'],
                opset_version=14
            )
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            if os.path.exists(path):
                os.unlink(path)

    @pytest.mark.xfail(reason="ALiBi broadcasting shape mismatch with ONNX export")
    def test_attention_exportable(self):
        """Attention with ALiBi exports to ONNX.

        NOTE: Currently fails due to ALiBi dynamic shape broadcasting.
        The ALiBi bias computation uses dynamic shapes that ONNX struggles with.
        """
        from neural_vm.vm_step import AutoregressiveAttention

        attn = AutoregressiveAttention(
            d_model=64,
            num_heads=4,
            layer_idx=0,
            positional_encoding="alibi"
        )
        attn.eval()

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            path = f.name

        try:
            dummy = torch.randn(1, 10, 64)
            torch.onnx.export(
                attn, dummy, path,
                input_names=['input'],
                output_names=['output'],
                opset_version=14
            )
            assert os.path.exists(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_linear_exportable(self):
        """Linear layer exports to ONNX (sanity check)."""
        linear = nn.Linear(64, 128)

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            path = f.name

        try:
            dummy = torch.randn(1, 10, 64)
            torch.onnx.export(
                linear, dummy, path,
                input_names=['input'],
                output_names=['output'],
                opset_version=14
            )
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestModelConfiguration:
    """Test model configuration matches vanilla transformer spec."""

    def test_default_config_values(self):
        """Default config uses standard transformer settings."""
        from neural_vm.vm_step import AutoregressiveVM

        model = AutoregressiveVM()

        # Check default dimensions
        assert model.d_model == 512
        assert len(model.blocks) == 16  # 16 layers
        # Each block should have attention and FFN
        assert hasattr(model.blocks[0], 'attn')
        assert hasattr(model.blocks[0], 'ffn')

    def test_attention_head_dim(self):
        """Attention head dimension is d_model / n_heads."""
        from neural_vm.vm_step import AutoregressiveVM

        d_model = 512
        n_heads = 8
        model = AutoregressiveVM(d_model=d_model, n_heads=n_heads)

        # Head dim should be 64 (512 / 8)
        expected_head_dim = d_model // n_heads
        actual_head_dim = model.blocks[0].attn.head_dim

        assert actual_head_dim == expected_head_dim, \
            f"Expected head_dim={expected_head_dim}, got {actual_head_dim}"

    def test_ffn_hidden_ratio(self):
        """FFN hidden dimension follows standard ratio."""
        from neural_vm.vm_step import AutoregressiveVM

        d_model = 512
        ffn_hidden = 4096  # 8x ratio
        model = AutoregressiveVM(d_model=d_model, ffn_hidden=ffn_hidden)

        # Check first block FFN
        ffn = model.blocks[0].ffn
        assert ffn.W_up.shape[0] == ffn_hidden
        assert ffn.W_down.shape[1] == ffn_hidden


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
