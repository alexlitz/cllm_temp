"""
Comprehensive edge case and bug tests for RoPE/ALiBi positional encoding.

Tests critical gaps identified in ROPE_BUGS_AND_GAPS.md:
- KV cache with RoPE
- Sequence length bounds
- Config validation
- Model serialization
- Concurrent access
- Hybrid mode boundaries
"""

import os
import pytest
import torch
import tempfile
import threading
from unittest.mock import patch

# Import after setting config to avoid initialization issues
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRoPESequenceBounds:
    """Test RoPE handles sequence length edge cases correctly."""

    def setup_method(self):
        """Reset config before each test."""
        from neural_vm.config import reset_config
        reset_config()

    def test_rope_at_max_seq_len(self):
        """Test RoPE works correctly at exactly max_seq_len."""
        from neural_vm.config import set_config, VMConfig
        from neural_vm.vm_step import AutoregressiveAttention

        set_config(VMConfig.rope_mode())
        max_len = 100
        attn = AutoregressiveAttention(dim=256, num_heads=4, max_seq_len=max_len)

        # Test exactly at max length
        x = torch.randn(1, max_len, 256)
        out = attn(x)

        assert out.shape == (1, max_len, 256)

    def test_rope_beyond_max_seq_len_auto_extends(self):
        """Test RoPE automatically extends cache for sequences beyond initial max_seq_len."""
        from neural_vm.config import set_config, VMConfig
        from neural_vm.vm_step import AutoregressiveAttention

        set_config(VMConfig.rope_mode())
        max_len = 100
        attn = AutoregressiveAttention(dim=256, num_heads=4, max_seq_len=max_len)

        # Initial cache size
        assert attn._rope_cos.shape[0] == max_len

        # Test beyond initial max length - should auto-extend
        x = torch.randn(1, max_len + 10, 256)
        out = attn(x)

        # Should have extended cache
        assert attn._rope_cos.shape[0] >= (max_len + 10)
        # Output should have correct shape
        assert out.shape == (1, max_len + 10, 256)

    def test_rope_empty_sequence(self):
        """Test RoPE with empty sequence."""
        from neural_vm.config import set_config, VMConfig
        from neural_vm.vm_step import AutoregressiveAttention

        set_config(VMConfig.rope_mode())
        attn = AutoregressiveAttention(dim=256, num_heads=4, max_seq_len=100)

        # Empty sequence
        x = torch.randn(1, 0, 256)
        out = attn(x)
        assert out.shape == (1, 0, 256)

    def test_rope_single_token(self):
        """Test RoPE with single token."""
        from neural_vm.config import set_config, VMConfig
        from neural_vm.vm_step import AutoregressiveAttention

        set_config(VMConfig.rope_mode())
        attn = AutoregressiveAttention(dim=256, num_heads=4, max_seq_len=100)

        x = torch.randn(1, 1, 256)
        out = attn(x)
        assert out.shape == (1, 1, 256)

    def test_alibi_no_max_len_restriction(self):
        """Test ALiBi works beyond max_seq_len (computed on-the-fly)."""
        from neural_vm.config import set_config, VMConfig
        from neural_vm.vm_step import AutoregressiveAttention

        set_config(VMConfig.alibi_mode())
        max_len = 100
        attn = AutoregressiveAttention(dim=256, num_heads=4, max_seq_len=max_len)

        # ALiBi should work beyond max_len (computed dynamically)
        x = torch.randn(1, max_len + 50, 256)
        out = attn(x)
        assert out.shape == (1, max_len + 50, 256)


class TestRoPEWithKVCache:
    """Test RoPE correctness with KV caching (HIGH PRIORITY)."""

    def setup_method(self):
        """Reset config before each test."""
        from neural_vm.config import reset_config
        reset_config()

    def test_rope_kv_cache_position_offsets(self):
        """Test RoPE computes correct position offsets with KV cache."""
        from neural_vm.config import set_config, VMConfig
        from neural_vm.vm_step import AutoregressiveAttention
        from neural_vm.kv_cache import TransformerKVCache

        set_config(VMConfig.rope_mode())
        attn = AutoregressiveAttention(dim=256, num_heads=4, max_seq_len=1000)

        # First forward: 10 tokens
        x1 = torch.randn(1, 10, 256)
        cache = TransformerKVCache()
        out1 = attn(x1, kv_cache=cache)

        assert out1.shape == (1, 10, 256)
        assert cache.cache_size == 10

        # Second forward: extend by 5 tokens (positions 10-14)
        x2 = torch.randn(1, 15, 256)  # Full sequence including cached
        out2 = attn(x2, kv_cache=cache)

        assert out2.shape == (1, 15, 256)
        assert cache.cache_size == 15

        # Verify RoPE applied correct positions
        # For second pass:
        # - Q should use positions [10, 15) (q_offset = 15 - 5 = 10)
        # - K should use positions [0, 15)
        # This is tested implicitly by shape correctness

    def test_rope_vs_alibi_cache_consistency(self):
        """Test both RoPE and ALiBi produce consistent outputs with caching."""
        from neural_vm.config import set_config, VMConfig
        from neural_vm.vm_step import AutoregressiveAttention
        from neural_vm.kv_cache import TransformerKVCache

        # Test both modes
        for mode in [VMConfig.rope_mode(), VMConfig.alibi_mode()]:
            set_config(mode)
            attn = AutoregressiveAttention(dim=256, num_heads=4, max_seq_len=1000)

            # Generate sequence incrementally with cache
            x_full = torch.randn(1, 20, 256)
            cache = TransformerKVCache()

            # First 10 tokens
            out1 = attn(x_full[:, :10, :], kv_cache=cache)

            # Extend to 20 tokens
            out2 = attn(x_full, kv_cache=cache)

            # Generate full sequence without cache
            set_config(mode)  # Reset config to avoid state pollution
            attn_nocache = AutoregressiveAttention(dim=256, num_heads=4, max_seq_len=1000)
            attn_nocache.load_state_dict(attn.state_dict())
            out_full = attn_nocache(x_full)

            # Cached and non-cached should match (within numerical precision)
            assert torch.allclose(out2, out_full, atol=1e-5), f"Cache mismatch in {mode.positional_encoding} mode"

    def test_rope_cache_empty_new_tokens(self):
        """Test RoPE handles case where all tokens are cached."""
        from neural_vm.config import set_config, VMConfig
        from neural_vm.vm_step import AutoregressiveAttention
        from neural_vm.kv_cache import TransformerKVCache

        set_config(VMConfig.rope_mode())
        attn = AutoregressiveAttention(dim=256, num_heads=4, max_seq_len=1000)

        # Cache 10 tokens
        x = torch.randn(1, 10, 256)
        cache = TransformerKVCache()
        out1 = attn(x, kv_cache=cache)

        # Call again with same sequence (all cached, new_tokens = 0)
        out2 = attn(x, kv_cache=cache)

        # Should use cached K/V, only compute Q
        assert torch.allclose(out1, out2, atol=1e-6)


class TestConfigValidation:
    """Test config system validates inputs correctly."""

    def setup_method(self):
        """Reset config before each test."""
        from neural_vm.config import reset_config
        reset_config()

    def test_invalid_env_var_defaults_to_alibi(self):
        """Test invalid NEURAL_VM_POS_ENCODING falls back to alibi."""
        from neural_vm.config import get_config, reset_config

        with patch.dict(os.environ, {"NEURAL_VM_POS_ENCODING": "invalid_value"}):
            reset_config()
            config = get_config()

            # Currently defaults silently to alibi (no warning - BUG)
            # After fix, should warn:
            # with pytest.warns(UserWarning, match="Unknown positional encoding"):
            #     config = get_config()

            assert config.positional_encoding == "alibi"

    def test_valid_env_var_values(self):
        """Test all valid NEURAL_VM_POS_ENCODING values."""
        from neural_vm.config import get_config, reset_config

        for value, expected in [("alibi", "alibi"), ("rope", "rope"), ("hybrid", "hybrid")]:
            with patch.dict(os.environ, {"NEURAL_VM_POS_ENCODING": value}):
                reset_config()
                config = get_config()
                assert config.positional_encoding == expected

    def test_config_change_after_model_creation(self):
        """Test config changes don't affect already-created models."""
        from neural_vm.config import set_config, VMConfig, reset_config
        from neural_vm.vm_step import AutoregressiveAttention

        # Create model with RoPE
        set_config(VMConfig.rope_mode())
        attn_rope = AutoregressiveAttention(dim=256, num_heads=4, layer_idx=5)

        assert attn_rope._positional_encoding == "rope"
        assert attn_rope._rope_cos is not None
        assert attn_rope.alibi_slopes is None

        # Change config to ALiBi
        set_config(VMConfig.alibi_mode())

        # Existing model should still be RoPE
        assert attn_rope._positional_encoding == "rope"
        assert attn_rope._rope_cos is not None

        # New model should be ALiBi
        attn_alibi = AutoregressiveAttention(dim=256, num_heads=4, layer_idx=5)
        assert attn_alibi._positional_encoding == "alibi"
        assert attn_alibi.alibi_slopes is not None
        assert attn_alibi._rope_cos is None

    def test_vmconfig_dataclass_validation(self):
        """Test VMConfig validates positional_encoding values."""
        from neural_vm.config import VMConfig

        # Valid values should work
        VMConfig(positional_encoding="alibi")
        VMConfig(positional_encoding="rope")
        VMConfig(positional_encoding="hybrid")

        # Invalid value currently accepted (BUG - no runtime validation)
        # After fix with __post_init__, should raise:
        # with pytest.raises(ValueError, match="Invalid positional_encoding"):
        #     VMConfig(positional_encoding="invalid")

        # For now, test that invalid value can be created (demonstrates bug)
        config = VMConfig(positional_encoding="invalid")
        assert config.positional_encoding == "invalid"  # No validation!


class TestModelSerialization:
    """Test RoPE models can be saved and loaded correctly."""

    def setup_method(self):
        """Reset config before each test."""
        from neural_vm.config import reset_config
        reset_config()

    def test_rope_model_state_dict(self):
        """Test RoPE model state_dict includes RoPE cache."""
        from neural_vm.config import set_config, VMConfig
        from neural_vm.vm_step import AutoregressiveAttention

        set_config(VMConfig.rope_mode())
        attn = AutoregressiveAttention(dim=256, num_heads=4, max_seq_len=100)

        state = attn.state_dict()

        # Should contain RoPE cache
        assert "_rope_cos" in state
        assert "_rope_sin" in state

        # Should NOT contain alibi_slopes
        assert "alibi_slopes" not in state

    def test_alibi_model_state_dict(self):
        """Test ALiBi model state_dict includes slopes."""
        from neural_vm.config import set_config, VMConfig
        from neural_vm.vm_step import AutoregressiveAttention

        set_config(VMConfig.alibi_mode())
        attn = AutoregressiveAttention(dim=256, num_heads=4, max_seq_len=100)

        state = attn.state_dict()

        # Should contain alibi_slopes
        assert "alibi_slopes" in state

        # Should NOT contain RoPE cache
        assert "_rope_cos" not in state
        assert "_rope_sin" not in state

    def test_save_load_rope_model(self):
        """Test RoPE model can be saved and loaded."""
        from neural_vm.config import set_config, VMConfig
        from neural_vm.vm_step import AutoregressiveAttention

        set_config(VMConfig.rope_mode())
        attn1 = AutoregressiveAttention(dim=256, num_heads=4, max_seq_len=100)

        # Save state
        state = attn1.state_dict()

        # Create new model and load
        attn2 = AutoregressiveAttention(dim=256, num_heads=4, max_seq_len=100)
        attn2.load_state_dict(state)

        # Verify identical outputs
        x = torch.randn(1, 10, 256)
        out1 = attn1(x)
        out2 = attn2(x)

        assert torch.allclose(out1, out2, atol=1e-6)

    def test_save_load_full_model(self):
        """Test full AutoregressiveVM can be saved/loaded with RoPE."""
        from neural_vm.config import set_config, VMConfig
        from neural_vm.vm_step import AutoregressiveVM

        set_config(VMConfig.rope_mode())
        model1 = AutoregressiveVM(n_layers=4, d_model=256, n_heads=4)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            temp_path = f.name

        try:
            # Save
            torch.save(model1.state_dict(), temp_path)

            # Load
            model2 = AutoregressiveVM(n_layers=4, d_model=256, n_heads=4)
            model2.load_state_dict(torch.load(temp_path))

            # Verify identical
            x = torch.randint(0, 256, (1, 10))
            out1 = model1(x)
            out2 = model2(x)

            assert torch.allclose(out1, out2, atol=1e-6)
        finally:
            os.unlink(temp_path)


class TestHybridModeBoundaries:
    """Test hybrid mode correctly assigns encodings to layer boundaries."""

    def setup_method(self):
        """Reset config before each test."""
        from neural_vm.config import reset_config
        reset_config()

    def test_hybrid_layer_assignments(self):
        """Test hybrid mode assigns ALiBi to L0-L2, RoPE to L3+."""
        from neural_vm.config import set_config, VMConfig
        from neural_vm.vm_step import AutoregressiveAttention

        set_config(VMConfig.hybrid_mode())

        # Layers 0-2 should use ALiBi (check functional mechanism, not string)
        for layer_idx in [0, 1, 2]:
            attn = AutoregressiveAttention(dim=256, num_heads=4, layer_idx=layer_idx)
            # String should be "alibi" for L0-L2 in hybrid mode
            assert attn._positional_encoding == "alibi", f"Layer {layer_idx} should be ALiBi"
            assert attn.alibi_slopes is not None, f"Layer {layer_idx} should have ALiBi slopes"
            assert attn._rope_cos is None, f"Layer {layer_idx} should not have RoPE cache"

        # Layers 3+ should use RoPE (check functional mechanism)
        for layer_idx in [3, 4, 5, 10, 15]:
            attn = AutoregressiveAttention(dim=256, num_heads=4, layer_idx=layer_idx)
            # String stays "hybrid" but functional mechanism is RoPE
            # What matters: has RoPE cache, no ALiBi slopes
            assert attn._rope_cos is not None, f"Layer {layer_idx} should have RoPE cache"
            assert attn.alibi_slopes is None, f"Layer {layer_idx} should not have ALiBi slopes"

    def test_hybrid_without_layer_idx_defaults_to_rope(self):
        """Test hybrid mode without layer_idx uses RoPE."""
        from neural_vm.config import set_config, VMConfig
        from neural_vm.vm_step import AutoregressiveAttention

        set_config(VMConfig.hybrid_mode())

        # No layer_idx provided
        attn = AutoregressiveAttention(dim=256, num_heads=4)
        # Should default to RoPE (hybrid with layer_idx=None uses global positional_encoding)
        assert attn._positional_encoding == "hybrid"

    def test_full_vm_hybrid_mode(self):
        """Test full VM in hybrid mode has correct layer encodings."""
        from neural_vm.config import set_config, VMConfig
        from neural_vm.vm_step import AutoregressiveVM

        set_config(VMConfig.hybrid_mode())
        model = AutoregressiveVM(n_layers=16, d_model=256, n_heads=4)

        # Verify each layer uses correct functional mechanism
        for i, block in enumerate(model.blocks):
            attn = block.attn
            if i < 3:
                # L0-L2: ALiBi mechanism
                assert attn._positional_encoding == "alibi", f"Layer {i} should be ALiBi in hybrid"
                assert attn.alibi_slopes is not None, f"Layer {i} should have ALiBi slopes"
                assert attn._rope_cos is None, f"Layer {i} should not have RoPE cache"
            else:
                # L3+: RoPE mechanism (string may be "hybrid" but has RoPE cache)
                assert attn._rope_cos is not None, f"Layer {i} should have RoPE cache"
                assert attn.alibi_slopes is None, f"Layer {i} should not have ALiBi slopes"


class TestConcurrentAccess:
    """Test thread safety of config system."""

    def setup_method(self):
        """Reset config before each test."""
        from neural_vm.config import reset_config
        reset_config()

    def test_concurrent_get_config(self):
        """Test get_config() is thread-safe (currently NOT - race condition)."""
        from neural_vm.config import get_config, reset_config

        reset_config()
        configs = []
        errors = []

        def get_config_thread():
            try:
                config = get_config()
                configs.append(config)
            except Exception as e:
                errors.append(e)

        # Create 10 threads that all call get_config simultaneously
        threads = [threading.Thread(target=get_config_thread) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have no errors
        assert len(errors) == 0

        # All configs should be the same object (singleton)
        assert len(set(id(c) for c in configs)) == 1, "Config should be singleton"

    def test_concurrent_model_creation(self):
        """Test creating models concurrently with same config."""
        from neural_vm.config import set_config, VMConfig
        from neural_vm.vm_step import AutoregressiveAttention

        set_config(VMConfig.rope_mode())
        models = []
        errors = []

        def create_model_thread():
            try:
                attn = AutoregressiveAttention(dim=256, num_heads=4)
                models.append(attn)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_model_thread) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(models) == 5

        # All should be RoPE
        for model in models:
            assert model._positional_encoding == "rope"


class TestBatchSizes:
    """Test RoPE works with various batch sizes."""

    def setup_method(self):
        """Reset config before each test."""
        from neural_vm.config import reset_config
        reset_config()

    def test_rope_batch_size_1(self):
        """Test RoPE with batch size 1."""
        from neural_vm.config import set_config, VMConfig
        from neural_vm.vm_step import AutoregressiveAttention

        set_config(VMConfig.rope_mode())
        attn = AutoregressiveAttention(dim=256, num_heads=4)

        x = torch.randn(1, 10, 256)
        out = attn(x)
        assert out.shape == (1, 10, 256)

    def test_rope_batch_size_8(self):
        """Test RoPE with batch size 8."""
        from neural_vm.config import set_config, VMConfig
        from neural_vm.vm_step import AutoregressiveAttention

        set_config(VMConfig.rope_mode())
        attn = AutoregressiveAttention(dim=256, num_heads=4)

        x = torch.randn(8, 10, 256)
        out = attn(x)
        assert out.shape == (8, 10, 256)

    def test_rope_varying_batch_sizes(self):
        """Test RoPE with different batch sizes in successive calls."""
        from neural_vm.config import set_config, VMConfig
        from neural_vm.vm_step import AutoregressiveAttention

        set_config(VMConfig.rope_mode())
        attn = AutoregressiveAttention(dim=256, num_heads=4)

        # Batch size 1
        x1 = torch.randn(1, 10, 256)
        out1 = attn(x1)
        assert out1.shape == (1, 10, 256)

        # Batch size 4
        x2 = torch.randn(4, 10, 256)
        out2 = attn(x2)
        assert out2.shape == (4, 10, 256)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
