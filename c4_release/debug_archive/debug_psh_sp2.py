#!/usr/bin/env python3
"""Debug PSH SP using the actual batch runner."""
import os
import sys
sys.path.insert(0, os.getcwd())

import torch
from neural_vm.batch_runner_v2 import UltraBatchRunner
from neural_vm.embedding import Opcode

# Test program: IMM 0, PSH, IMM 0, MUL, EXIT
bytecode = [Opcode.IMM | (0 << 8), Opcode.PSH, Opcode.IMM | (0 << 8), Opcode.MUL, Opcode.EXIT]

# Run with strict mode
runner = UltraBatchRunner(batch_size=1, device='cpu', strict=True)
try:
    result = runner.run_batch([bytecode])
    print(f"Success! Exit code: {result[0]}")
except AssertionError as e:
    print(f"Failed: {e}")
