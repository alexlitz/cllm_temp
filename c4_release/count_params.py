#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights
from neural_vm.alu_weights import ALUConfig

print('Counting total non-zero parameters')
print('=' * 70)

# Model with efficient ALU
print('\nBuilding model with EFFICIENT ALU...')
model_eff = AutoregressiveVM(
    d_model=512, n_layers=16, n_heads=8,
    ffn_hidden=4096, max_seq_len=512
)
set_vm_weights(model_eff, alu_config=ALUConfig.efficient_nibble())

# Model with lookup tables
print('Building model with LOOKUP tables...')
model_lookup = AutoregressiveVM(
    d_model=512, n_layers=16, n_heads=8,
    ffn_hidden=4096, max_seq_len=512
)
set_vm_weights(model_lookup, alu_config=ALUConfig.all_nibble())

print()
print('=' * 70)

# Count total parameters
def count_params(model):
    total = 0
    nonzero = 0
    for p in model.parameters():
        total += p.numel()
        nonzero += (p != 0).sum().item()
    return total, nonzero

total_eff, nonzero_eff = count_params(model_eff)
total_lookup, nonzero_lookup = count_params(model_lookup)

print('TOTAL PARAMETERS:')
print(f'  Model capacity: {total_eff:,} parameters')
print()

print('NON-ZERO PARAMETERS:')
print(f'  Efficient ALU:  {nonzero_eff:,}')
print(f'  Lookup tables:  {nonzero_lookup:,}')
reduction = nonzero_lookup - nonzero_eff
pct = (reduction / nonzero_lookup) * 100
print(f'  Reduction:      {reduction:,} ({pct:.2f}%)')
print()

print('SPARSITY:')
eff_sparse = (1 - nonzero_eff / total_eff) * 100
lookup_sparse = (1 - nonzero_lookup / total_lookup) * 100
print(f'  Efficient:  {eff_sparse:.2f}% sparse')
print(f'  Lookup:     {lookup_sparse:.2f}% sparse')

print('=' * 70)
