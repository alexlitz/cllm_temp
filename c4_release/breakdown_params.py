#!/usr/bin/env python3
"""Detailed parameter breakdown for Neural VM."""

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights

print('Building Neural VM and analyzing parameter distribution...')
print('=' * 70)

# Create model
model = AutoregressiveVM(
    d_model=512,
    n_layers=16,
    n_heads=8,
    ffn_hidden=4096,
    max_seq_len=512
)

# Set weights
set_vm_weights(model)

print('\nAnalyzing parameter distribution...')
print()
print('=' * 70)
print('NEURAL VM PARAMETER BREAKDOWN')
print('=' * 70)

# Embedding
embed_params = (model.embed.weight != 0).sum().item()
print(f'\nEmbedding:          {embed_params:>10,} parameters')

# Head (output projection)
head_params = (model.head.weight != 0).sum().item()
if model.head.bias is not None:
    head_params += (model.head.bias != 0).sum().item()
print(f'Head (output):      {head_params:>10,} parameters')

# Per-layer breakdown
print('\nPer-Layer Breakdown:')
print('-' * 70)
print(f'{"Layer":<8} {"Attention":<15} {"FFN":<15} {"Total":<15}')
print('-' * 70)

layer_totals = []
for i in range(16):
    block = model.blocks[i]

    # Attention parameters
    attn_params = 0
    for name, param in block.attn.named_parameters():
        attn_params += (param != 0).sum().item()

    # FFN parameters
    ffn_params = 0
    for name, param in block.ffn.named_parameters():
        ffn_params += (param != 0).sum().item()

    layer_total = attn_params + ffn_params
    layer_totals.append((i, attn_params, ffn_params, layer_total))

    print(f'L{i:<7} {attn_params:<15,} {ffn_params:<15,} {layer_total:<15,}')

print('-' * 70)

# Summary by component
total_attn = sum(x[1] for x in layer_totals)
total_ffn = sum(x[2] for x in layer_totals)
total_layers = sum(x[3] for x in layer_totals)

print(f'{"TOTAL":<8} {total_attn:<15,} {total_ffn:<15,} {total_layers:<15,}')

# Overall summary
print()
print('=' * 70)
print('SUMMARY')
print('=' * 70)
print(f'Embedding:          {embed_params:>10,} ({embed_params/(embed_params+head_params+total_layers)*100:>5.2f}%)')
print(f'Attention (all):    {total_attn:>10,} ({total_attn/(embed_params+head_params+total_layers)*100:>5.2f}%)')
print(f'FFN (all):          {total_ffn:>10,} ({total_ffn/(embed_params+head_params+total_layers)*100:>5.2f}%)')
print(f'Head:               {head_params:>10,} ({head_params/(embed_params+head_params+total_layers)*100:>5.2f}%)')
print('-' * 70)
print(f'TOTAL:              {embed_params+head_params+total_layers:>10,}')

# Show top FFN layers
print()
print('Top 5 FFN layers by parameter count:')
print('-' * 70)
ffn_sorted = sorted(layer_totals, key=lambda x: x[2], reverse=True)[:5]
for i, attn, ffn, total in ffn_sorted:
    print(f'  L{i:<2}: {ffn:>10,} FFN parameters')

print()
print('=' * 70)
