#!/usr/bin/env python3
"""Test layer allocation with and without sharing."""

from neural_vm.layer_allocator import LayerAllocator

print("WITHOUT SHARING (Sequential Allocation):")
print()
allocator_no_share = LayerAllocator(use_sharing=False)
allocator_no_share.print_allocation()

print("\n\n")

print("WITH SHARING (Graph Coloring):")
print()
allocator_share = LayerAllocator(use_sharing=True)
allocator_share.print_allocation()

print("\n\n")
print(f"Savings: {allocator_no_share.total_layers} layers → {allocator_share.total_layers} layers")
print(f"Reduction: {allocator_no_share.total_layers - allocator_share.total_layers} layers ({100 * (1 - allocator_share.total_layers / allocator_no_share.total_layers):.1f}% reduction)")
