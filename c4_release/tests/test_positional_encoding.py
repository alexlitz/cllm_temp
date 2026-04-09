"""
Tests for RoPE/ALiBi positional encoding toggle functionality.

Tests both PureAttention and AutoregressiveAttention classes with:
- ALiBi mode (default)
- RoPE mode
- Hybrid mode (ALiBi for L0-L2, RoPE for rest)
"""

import torch
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.config import VMConfig, set_config, reset_config, get_config
from neural_vm.base_layers import PureAttention, rotate_half, apply_rotary_emb, precompute_rope_cache
from neural_vm.vm_step import AutoregressiveAttention


class TestConfigSystem:
    """Test VMConfig dataclass and global config management."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_default_config(self):
        """Test default config is ALiBi."""
        config = get_config()
        assert config.positional_encoding == "alibi"
        assert config.rope_base == 10000.0

    def test_alibi_mode_factory(self):
        """Test ALiBi mode factory method."""
        config = VMConfig.alibi_mode()
        assert config.positional_encoding == "alibi"

    def test_rope_mode_factory(self):
        """Test RoPE mode factory method."""
        config = VMConfig.rope_mode()
        assert config.positional_encoding == "rope"

    def test_hybrid_mode_factory(self):
        """Test hybrid mode factory method."""
        config = VMConfig.hybrid_mode()
        assert config.positional_encoding == "hybrid"

    def test_set_config(self):
        """Test setting global config."""
        config = VMConfig.rope_mode()
        set_config(config)
        assert get_config().positional_encoding == "rope"

    def test_env_var_alibi(self):
        """Test environment variable for ALiBi mode."""
        os.environ["NEURAL_VM_POS_ENCODING"] = "alibi"
        reset_config()
        config = get_config()
        assert config.positional_encoding == "alibi"
        del os.environ["NEURAL_VM_POS_ENCODING"]

    def test_env_var_rope(self):
        """Test environment variable for RoPE mode."""
        os.environ["NEURAL_VM_POS_ENCODING"] = "rope"
        reset_config()
        config = get_config()
        assert config.positional_encoding == "rope"
        del os.environ["NEURAL_VM_POS_ENCODING"]

    def test_env_var_hybrid(self):
        """Test environment variable for hybrid mode."""
        os.environ["NEURAL_VM_POS_ENCODING"] = "hybrid"
        reset_config()
        config = get_config()
        assert config.positional_encoding == "hybrid"
        del os.environ["NEURAL_VM_POS_ENCODING"]


class TestRoPEHelpers:
    """Test RoPE helper functions."""

    def test_rotate_half(self):
        """Test rotate_half function."""
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        rotated = rotate_half(x)
        expected = torch.tensor([[-2.0, 1.0, -4.0, 3.0]])
        assert torch.allclose(rotated, expected)

    def test_precompute_rope_cache(self):
        """Test RoPE cache precomputation."""
        head_dim = 64
        max_seq_len = 100
        cos, sin = precompute_rope_cache(head_dim, max_seq_len)

        assert cos.shape == (max_seq_len, head_dim)
        assert sin.shape == (max_seq_len, head_dim)

        # First position should be all ones for cos, all zeros for sin
        assert torch.allclose(cos[0], torch.ones(head_dim), atol=1e-6)
        assert torch.allclose(sin[0], torch.zeros(head_dim), atol=1e-6)

    def test_apply_rotary_emb(self):
        """Test apply_rotary_emb function."""
        B, H, S, HD = 1, 4, 10, 64
        q = torch.randn(B, H, S, HD)
        k = torch.randn(B, H, S, HD)

        cos, sin = precompute_rope_cache(HD, S)
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, S, HD]
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, S, HD]

        q_rope, k_rope = apply_rotary_emb(q, k, cos, sin)

        assert q_rope.shape == q.shape
        assert k_rope.shape == k.shape
        assert not torch.allclose(q_rope, q)  # Should be different after rotation
        assert not torch.allclose(k_rope, k)


# PureAttention tests removed - VM uses AutoregressiveAttention


class TestAutoregressiveAttention:
    """Test AutoregressiveAttention with different positional encodings."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_alibi_mode_creates_slopes(self):
        """Test ALiBi mode creates alibi_slopes buffer."""
        set_config(VMConfig.alibi_mode())
        attn = AutoregressiveAttention(dim=256, num_heads=4)

        assert hasattr(attn, 'alibi_slopes')
        assert attn.alibi_slopes is not None
        assert attn.alibi_slopes.shape == (4,)
        assert attn._positional_encoding == "alibi"
        assert attn._rope_cos is None
        assert attn._rope_sin is None

    def test_rope_mode_creates_cache(self):
        """Test RoPE mode creates RoPE cache."""
        set_config(VMConfig.rope_mode())
        attn = AutoregressiveAttention(dim=256, num_heads=4, max_seq_len=100)

        assert hasattr(attn, '_rope_cos')
        assert hasattr(attn, '_rope_sin')
        assert attn._rope_cos is not None
        assert attn._rope_sin is not None
        assert attn._rope_cos.shape == (100, 64)  # max_seq_len, head_dim
        assert attn._rope_sin.shape == (100, 64)
        assert attn._positional_encoding == "rope"
        assert attn.alibi_slopes is None

    def test_hybrid_mode_layer0(self):
        """Test hybrid mode uses ALiBi for layer 0."""
        set_config(VMConfig.hybrid_mode())
        attn = AutoregressiveAttention(dim=256, num_heads=4, layer_idx=0)

        assert attn._positional_encoding == "alibi"
        assert attn.alibi_slopes is not None
        assert attn._rope_cos is None

    def test_hybrid_mode_layer3(self):
        """Test hybrid mode uses RoPE for layer 3."""
        set_config(VMConfig.hybrid_mode())
        attn = AutoregressiveAttention(dim=256, num_heads=4, layer_idx=3, max_seq_len=100)

        # Hybrid mode keeps "hybrid" as the encoding type
        assert attn._positional_encoding == "hybrid"
        # But it should have RoPE cache (since layer > 2)
        assert attn._rope_cos is not None
        assert attn._rope_sin is not None
        # And no ALiBi slopes (since layer > 2)
        assert attn.alibi_slopes is None

    def test_forward_alibi(self):
        """Test forward pass works with ALiBi."""
        set_config(VMConfig.alibi_mode())
        attn = AutoregressiveAttention(dim=256, num_heads=4)

        x = torch.randn(2, 10, 256)
        out = attn(x)

        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_forward_rope(self):
        """Test forward pass works with RoPE."""
        set_config(VMConfig.rope_mode())
        attn = AutoregressiveAttention(dim=256, num_heads=4, max_seq_len=100)

        x = torch.randn(2, 10, 256)
        out = attn(x)

        assert out.shape == x.shape
        assert not torch.isnan(out).any()



class TestVMIntegration:
    """Test full VM with different positional encodings."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_vm_creation_alibi(self):
        """Test VM creation with ALiBi mode."""
        set_config(VMConfig.alibi_mode())
        from neural_vm.vm_step import AutoregressiveVM

        model = AutoregressiveVM(n_layers=4, d_model=256, n_heads=4)

        # Check all attention layers have ALiBi
        for i, block in enumerate(model.blocks):
            assert block.attn._positional_encoding == "alibi"
            assert block.attn.alibi_slopes is not None
            assert block.attn._rope_cos is None

    def test_vm_creation_rope(self):
        """Test VM creation with RoPE mode."""
        set_config(VMConfig.rope_mode())
        from neural_vm.vm_step import AutoregressiveVM

        model = AutoregressiveVM(n_layers=4, d_model=256, n_heads=4)

        # Check all attention layers have RoPE
        for i, block in enumerate(model.blocks):
            assert block.attn._positional_encoding == "rope"
            assert block.attn._rope_cos is not None
            assert block.attn._rope_sin is not None
            assert block.attn.alibi_slopes is None

    def test_vm_creation_hybrid(self):
        """Test VM creation with hybrid mode."""
        set_config(VMConfig.hybrid_mode())
        from neural_vm.vm_step import AutoregressiveVM

        model = AutoregressiveVM(n_layers=6, d_model=256, n_heads=4)

        # Check L0-L2 have ALiBi
        for i in range(3):
            assert model.blocks[i].attn._positional_encoding == "alibi"
            assert model.blocks[i].attn.alibi_slopes is not None
            assert model.blocks[i].attn._rope_cos is None

        # Check L3+ have RoPE (hybrid mode)
        for i in range(3, 6):
            assert model.blocks[i].attn._positional_encoding == "hybrid"
            assert model.blocks[i].attn._rope_cos is not None
            assert model.blocks[i].attn._rope_sin is not None
            assert model.blocks[i].attn.alibi_slopes is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
