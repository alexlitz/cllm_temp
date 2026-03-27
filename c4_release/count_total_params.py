#!/usr/bin/env python3
"""Count total non-zero parameters in Neural VM with efficient vs lookup ALU."""

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights
from neural_vm.batch_runner_v2 import ALUConfig

print('Building models and counting parameters...')
print('=' * 70)

# Efficient model
print('\nBuilding model with EFFICIENT ALU...')
model_eff = AutoregressiveVM(d_model=512, n_layers=16, n_heads=8, ffn_hidden=4096, max_seq_len=512)
set_vm_weights(model_eff, alu_config=ALUConfig.efficient_nibble())

# Lookup model
print('Building model with LOOKUP tables...')
model_lookup = AutoregressiveVM(d_model=512, n_layers=16, n_heads=8, ffn_hidden=4096, max_seq_len=512)
set_vm_weights(model_lookup, alu_config=ALUConfig.all_nibble())

# Count total parameters
print('\nCounting parameters...')
total_params = sum(p.numel() for p in model_eff.parameters())
nonzero_eff = sum((p != 0).sum().item() for p in model_eff.parameters())
nonzero_lookup = sum((p != 0).sum().item() for p in model_lookup.parameters())

print()
print('=' * 70)
print('RESULTS')
print('=' * 70)
print()
print('TOTAL MODEL CAPACITY:', f'{total_params:,}', 'parameters')
print()
print('NON-ZERO PARAMETERS:')
print(f'  Efficient ALU:  {nonzero_eff:,}')
print(f'  Lookup tables:  {nonzero_lookup:,}')
reduction = nonzero_lookup - nonzero_eff
pct = (reduction / nonzero_lookup) * 100
print(f'  Reduction:      {reduction:,} ({pct:.2f}%)')
print()
print('SPARSITY:')
eff_sparse = (1 - nonzero_eff / total_params) * 100
lookup_sparse = (1 - nonzero_lookup / total_params) * 100
print(f'  Efficient:  {eff_sparse:.2f}% sparse')
print(f'  Lookup:     {lookup_sparse:.2f}% sparse')
print()
print('=' * 70)
