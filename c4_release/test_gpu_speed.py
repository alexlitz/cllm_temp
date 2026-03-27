#!/usr/bin/env python3
"""Test neural VM on GPU vs CPU."""

import torch
import time
from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import set_vm_weights

print('GPU Speed Test')
print('=' * 60)
print()

# Check GPU
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print()

# Test program
code = 'int main() { return 0; }'
bytecode_obj, data = compile_c(code)
bytecode = [instr for instr in bytecode_obj]

# Test on CPU
print('Testing on CPU...')
cpu_runner = AutoregressiveVMRunner(
    d_model=512, n_layers=16, n_heads=8, ffn_hidden=4096, max_seq_len=4096
)
set_vm_weights(cpu_runner.model)
cpu_runner.model.compact(block_size=32)
cpu_runner.model.compact_moe()

cpu_start = time.time()
cpu_output, cpu_exit = cpu_runner.run(bytecode, data, max_steps=3)
cpu_time = time.time() - cpu_start

print(f'  Time: {cpu_time:.1f}s')
print(f'  Result: {cpu_exit}')
print()

# Test on GPU
print('Testing on GPU...')
gpu_runner = AutoregressiveVMRunner(
    d_model=512, n_layers=16, n_heads=8, ffn_hidden=4096, max_seq_len=4096
)
set_vm_weights(gpu_runner.model)

# Move to GPU
print('  Moving model to GPU...')
gpu_runner.model = gpu_runner.model.cuda()
gpu_runner.model.compact(block_size=32)
gpu_runner.model.compact_moe()

gpu_start = time.time()
gpu_output, gpu_exit = gpu_runner.run(bytecode, data, max_steps=3)
gpu_time = time.time() - gpu_start

print(f'  Time: {gpu_time:.1f}s')
print(f'  Result: {gpu_exit}')
print()

# Compare
print('=' * 60)
print('COMPARISON')
print('=' * 60)
print(f'CPU time:  {cpu_time:.1f}s')
print(f'GPU time:  {gpu_time:.1f}s')
print(f'Speedup:   {cpu_time/gpu_time:.1f}x faster on GPU')
print()

# Extrapolate
print('Extrapolation to full test suite:')
print(f'  CPU:  {cpu_time * 1096 / 3600:.1f} hours')
print(f'  GPU:  {gpu_time * 1096 / 3600:.1f} hours')
print(f'  Time saved: {(cpu_time - gpu_time) * 1096 / 3600:.1f} hours')
