#!/usr/bin/env python3
"""Debug weighted eviction."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from neural_vm.weighted_eviction import WeightedEviction, WeightConfig
from neural_vm.vm_step import Token

class MockModel:
    def __init__(self):
        self.blocks = [None] * 16

model = MockModel()
config = WeightConfig()
eviction = WeightedEviction(model, config)

# Test PC marker
print("Testing PC marker score computation")
print("=" * 60)

token_ids = torch.tensor([[Token.REG_PC]])
embeddings = torch.zeros(1, 1, 512)

print(f"\nToken: REG_PC ({Token.REG_PC})")

# Extract features
features = eviction.extract_features(token_ids, embeddings, 0)
print(f"\nFeatures extracted:")
for k, v in features.items():
    if v != 0.0:
        print(f"  {k}: {v}")

# Compute score for each layer
print(f"\nLayer scores:")
for layer_idx in range(16):
    score = eviction.compute_layer_score(layer_idx, features)
    if score != -float('inf'):
        print(f"  Layer {layer_idx}: {score}")

# Compute overall score
overall_score = eviction.compute_score(token_ids, embeddings, 0)
print(f"\nOverall score (max across layers): {overall_score}")
print(f"Expected: 50.0")

print("\n" + "=" * 60)
print("Layer 3 weights:")
for k, v in config.l3_weights.items():
    print(f"  {k}: {v}")
