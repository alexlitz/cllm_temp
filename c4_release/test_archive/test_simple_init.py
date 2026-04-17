#!/usr/bin/env python3
"""Simple test - just initialize model and print success."""

import sys
print("Starting test...", file=sys.stderr)

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights

print("Imports successful", file=sys.stderr)

model = AutoregressiveVM(n_layers=17)
print("Model created", file=sys.stderr)

set_vm_weights(model)
print("Weights configured", file=sys.stderr)

print("✅ SUCCESS - Model initialization works")
