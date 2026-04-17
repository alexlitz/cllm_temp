#!/usr/bin/env python3
"""
Test if model creation succeeds or fails with the 12-head L15 code.
"""

import sys

try:
    print("Creating AutoregressiveVM...")
    from neural_vm.vm_step import AutoregressiveVM, set_vm_weights

    vm = AutoregressiveVM()
    print("✓ Model created successfully")

    print("\nSetting VM weights...")
    set_vm_weights(vm)
    print("✓ Weights set successfully")

    # Check L15
    l15_attn = vm.blocks[15].attn
    print(f"\nL15 num_heads: {l15_attn.num_heads}")
    print(f"L15 head_dim: {l15_attn.head_dim}")
    print(f"L15 W_q shape: {l15_attn.W_q.shape}")

    # Check if any non-zero weights were set
    import torch
    num_nonzero = (l15_attn.W_q != 0).sum().item()
    print(f"L15 W_q non-zero elements: {num_nonzero}")

except Exception as e:
    print(f"✗ Error during model creation:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
