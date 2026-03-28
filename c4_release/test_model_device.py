#!/usr/bin/env python3
"""Check model device and weight loading."""
import sys
sys.path.insert(0, '.')
import torch

from neural_vm.run_vm import AutoregressiveVMRunner

print("Creating runner...")
runner = AutoregressiveVMRunner()

print(f"Model device: {next(runner.model.parameters()).device}")
print(f"Model dtype: {next(runner.model.parameters()).dtype}")

# Check if weights are all zeros (not loaded)
weight_sum = 0
param_count = 0
for name, param in runner.model.named_parameters():
    weight_sum += param.abs().sum().item()
    param_count += 1
    
print(f"Total params: {param_count}")
print(f"Weight sum (should be non-zero if loaded): {weight_sum:.2f}")

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
