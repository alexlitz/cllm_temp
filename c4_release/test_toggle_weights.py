"""Test which toggle configurations work with set_vm_weights()."""

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights

def test_config(use_softmax1, pos_encoding):
    """Test if configuration works with hand-crafted weights."""
    print(f"\nTesting: use_softmax1={use_softmax1}, pos_encoding='{pos_encoding}'")

    model = AutoregressiveVM(
        use_softmax1=use_softmax1,
        pos_encoding=pos_encoding
    )

    try:
        set_vm_weights(model)
        print(f"  ✓ set_vm_weights() succeeded - can use hand-crafted weights")
        return True
    except ValueError as e:
        print(f"  ✗ set_vm_weights() rejected this configuration")
        print(f"     Reason: {str(e)[:100]}...")
        return False

if __name__ == "__main__":
    print("="*70)
    print("Testing Which Configurations Work With Hand-Crafted Weights")
    print("="*70)

    configs = [
        (True, 'alibi'),   # Default
        (False, 'alibi'),  # F.softmax + ALiBi
        (True, 'rope'),    # softmax1 + RoPE
        (False, 'rope'),   # F.softmax + RoPE
        (True, 'none'),    # softmax1 + no pos encoding
        (False, 'none'),   # F.softmax + no pos encoding
    ]

    results = []
    for use_softmax1, pos_encoding in configs:
        result = test_config(use_softmax1, pos_encoding)
        results.append((use_softmax1, pos_encoding, result))

    print("\n" + "="*70)
    print("Summary:")
    print("="*70)
    working = []
    not_working = []
    for use_softmax1, pos_encoding, result in results:
        config_name = f"softmax1={use_softmax1}, pos_encoding='{pos_encoding}'"
        if result:
            working.append(config_name)
        else:
            not_working.append(config_name)

    print("\n✓ WORKS WITH HAND-CRAFTED WEIGHTS:")
    for config in working:
        print(f"  - {config}")

    print("\n✗ REQUIRES TRAINING FROM SCRATCH:")
    for config in not_working:
        print(f"  - {config}")

    print("\n" + "="*70)
    print("IMPORTANT: Only the default configuration (softmax1=True, pos_encoding='alibi')")
    print("can use the hand-crafted weights. Other configurations would need to be")
    print("trained from scratch to work correctly for VM execution.")
    print("="*70)
