"""Test softmax1 and positional encoding toggles."""

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights
from neural_vm.embedding import Opcode

def test_toggle_configurations():
    """Test all combinations of softmax and positional encoding toggles."""

    print("Testing toggle configurations...")

    # Test configurations
    configs = [
        {"use_softmax1": True, "pos_encoding": "alibi", "name": "softmax1 + ALiBi (default)"},
        {"use_softmax1": False, "pos_encoding": "alibi", "name": "F.softmax + ALiBi"},
        {"use_softmax1": True, "pos_encoding": "rope", "name": "softmax1 + RoPE"},
        {"use_softmax1": False, "pos_encoding": "rope", "name": "F.softmax + RoPE"},
        {"use_softmax1": True, "pos_encoding": "none", "name": "softmax1 + no pos encoding"},
        {"use_softmax1": False, "pos_encoding": "none", "name": "F.softmax + no pos encoding"},
    ]

    # Simple test program: IMM 42; EXIT
    bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]

    for config in configs:
        print(f"\nTesting: {config['name']}")

        # Create model with this configuration
        model = AutoregressiveVM(
            use_softmax1=config['use_softmax1'],
            pos_encoding=config['pos_encoding']
        )

        # Set weights only for default configuration (ALiBi + softmax1)
        # Other configurations would need to be trained from scratch
        if config['use_softmax1'] and config['pos_encoding'] == 'alibi':
            set_vm_weights(model)
        else:
            # For non-default configs, just initialize with random weights
            # (normally these would be trained from scratch)
            pass

        # Verify toggle settings were applied
        for i, block in enumerate(model.blocks):
            attn = block.attn
            assert attn.use_softmax1 == config['use_softmax1'], \
                f"Layer {i}: use_softmax1 mismatch"
            assert attn.pos_encoding == config['pos_encoding'], \
                f"Layer {i}: pos_encoding mismatch"

            # Verify buffers are correctly set
            if config['pos_encoding'] == 'alibi':
                assert hasattr(attn, 'alibi_slopes'), \
                    f"Layer {i}: ALiBi slopes not found"
                assert not hasattr(attn, 'rope_freqs'), \
                    f"Layer {i}: RoPE freqs should not exist"
            elif config['pos_encoding'] == 'rope':
                assert hasattr(attn, 'rope_freqs'), \
                    f"Layer {i}: RoPE freqs not found"
                assert not hasattr(attn, 'alibi_slopes'), \
                    f"Layer {i}: ALiBi slopes should not exist"
            else:  # 'none'
                assert not hasattr(attn, 'alibi_slopes'), \
                    f"Layer {i}: ALiBi slopes should not exist"
                assert not hasattr(attn, 'rope_freqs'), \
                    f"Layer {i}: RoPE freqs should not exist"

        # Test forward pass with a simple random input
        # Use a small sequence to test attention mechanisms
        # Use CPU to avoid device transfer issues
        device = 'cpu'
        model = model.to(device)

        # Create a simple input with 10 random tokens
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=device)

        try:
            with torch.no_grad():
                logits = model(input_ids)
            print(f"  ✓ Forward pass successful, output shape: {logits.shape}")
        except Exception as e:
            print(f"  ✗ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    print("\n" + "="*60)
    print("All toggle configurations tested successfully!")
    print("="*60)

if __name__ == "__main__":
    test_toggle_configurations()
