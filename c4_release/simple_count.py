#!/usr/bin/env python3
"""Simple parameter count for Neural VM."""

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights

print('Building Neural VM and counting parameters...')
print('=' * 70)

# Create model
model = AutoregressiveVM(
    d_model=512,
    n_layers=16,
    n_heads=8,
    ffn_hidden=4096,
    max_seq_len=512
)

# Set weights (uses default configuration)
set_vm_weights(model)

print('\nCounting parameters...')

# Count total and non-zero parameters
total_params = 0
nonzero_params = 0

for name, param in model.named_parameters():
    total_params += param.numel()
    nonzero_params += (param != 0).sum().item()

print()
print('=' * 70)
print('NEURAL VM PARAMETER COUNT')
print('=' * 70)
print()
print(f'Total capacity:     {total_params:,} parameters')
print(f'Non-zero params:    {nonzero_params:,} parameters')
print(f'Sparsity:           {(1 - nonzero_params / total_params) * 100:.2f}%')
print()
print('=' * 70)
