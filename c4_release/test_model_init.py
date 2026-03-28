#!/usr/bin/env python3
"""Test if model initialization hangs."""
import sys
sys.path.insert(0, '.')
import time

print("1. Importing AutoregressiveVM...")
start = time.time()
from neural_vm.vm_step import AutoregressiveVM
print(f"   Imported in {time.time() - start:.2f}s")

print("2. Creating model...")
start = time.time()
model = AutoregressiveVM(
    d_model=512,
    n_layers=16,
    n_heads=8,
    ffn_hidden=4096,
    max_seq_len=4096,
)
print(f"   Created in {time.time() - start:.2f}s")

print("3. Setting eval mode...")
start = time.time()
model.eval()
print(f"   Set eval in {time.time() - start:.2f}s")

print("✅ Model initialization complete")
