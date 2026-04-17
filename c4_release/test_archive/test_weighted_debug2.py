#!/usr/bin/env python3
"""Debug overwritten MEM score."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from neural_vm.weighted_eviction import WeightedEviction, WeightConfig
from neural_vm.vm_step import Token, _SetDim as BD

class MockModel:
    def __init__(self):
        self.blocks = [None] * 16

model = MockModel()
config = WeightConfig()
eviction = WeightedEviction(model, config)

# Test overwritten MEM
print("Testing overwritten MEM score computation")
print("=" * 60)

token_ids = torch.tensor([[Token.MEM]])
embeddings = torch.zeros(1, 1, 512)
embeddings[0, 0, BD.MEM_STORE] = 0.0  # Overwritten

print(f"\nToken: MEM ({Token.MEM})")
print(f"MEM_STORE: {embeddings[0, 0, BD.MEM_STORE].item()}")

# Extract features
features = eviction.extract_features(token_ids, embeddings, 0)
print(f"\nFeatures extracted:")
for k, v in sorted(features.items()):
    print(f"  {k}: {v}")

# Compute score for layer 15
print(f"\nLayer 15 weights:")
for k, v in config.l15_weights.items():
    print(f"  {k}: {v}")

layer_15_score = eviction.compute_layer_score(15, features)
print(f"\nLayer 15 score: {layer_15_score}")

# Manual calculation
print(f"\nManual calculation:")
calc = 0
for feature_name, weight in config.l15_weights.items():
    feature_value = features.get(feature_name, 0.0)
    contribution = weight * feature_value
    print(f"  {feature_name}: {weight} * {feature_value} = {contribution}")
    if feature_value != 0.0:
        calc += contribution

print(f"\nTotal: {calc}")
print(f"Expected: -612.5")
