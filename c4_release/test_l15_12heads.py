#!/usr/bin/env python3
"""
Test that L15 now has 12 heads after Phase 2 extension.
"""

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights
from neural_vm.dim_registry import DimRegistry

def test_l15_has_12_heads():
    """Verify L15 has 12 heads (not 4)."""
    print("Testing L15 head count...")

    # Create VM and set weights (matrix resizing happens in set_vm_weights)
    vm = AutoregressiveVM()
    set_vm_weights(vm)

    # Get L15 attention layer
    l15_attn = vm.blocks[15].attn

    # Check Q matrix shape
    # With 12 heads and HD=64, Q should be (12 * 64, D) = (768, 512)
    HD = 64  # head_dim for 8-head model
    BD = DimRegistry()
    D = BD.d_model

    print(f"L15 W_q shape: {l15_attn.W_q.shape}")

    expected_q_rows = 12 * HD  # 12 heads × 64 dims/head = 768
    actual_q_rows = l15_attn.W_q.shape[0]

    print(f"Expected Q rows for 12 heads: {expected_q_rows}")
    print(f"Actual Q rows: {actual_q_rows}")

    # Check if heads 4-11 have non-zero weights
    for h in [4, 5, 6, 7, 8, 9, 10, 11]:
        base = h * HD
        # Check first row of each head
        head_weights = l15_attn.W_q[base, :].abs().sum().item()
        print(f"Head {h} (row {base}): {head_weights:.2f} total weight magnitude")

    # Count non-zero elements in different head ranges
    heads_0_3_nonzero = (l15_attn.W_q[0:4*HD, :] != 0).sum().item()
    heads_4_7_nonzero = (l15_attn.W_q[4*HD:8*HD, :] != 0).sum().item()
    heads_8_11_nonzero = (l15_attn.W_q[8*HD:12*HD, :] != 0).sum().item()
    print(f"\nNon-zero elements:")
    print(f"  Heads 0-3: {heads_0_3_nonzero}")
    print(f"  Heads 4-7: {heads_4_7_nonzero}")
    print(f"  Heads 8-11: {heads_8_11_nonzero}")

    if actual_q_rows == expected_q_rows:
        print(f"✓ L15 has 12 heads (Q rows = {actual_q_rows})")
        return True
    else:
        actual_heads = actual_q_rows // HD
        print(f"✗ L15 has {actual_heads} heads (expected 12)")
        return False

if __name__ == "__main__":
    success = test_l15_has_12_heads()
    exit(0 if success else 1)
