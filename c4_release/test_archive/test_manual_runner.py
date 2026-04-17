#!/usr/bin/env python3
"""Test runner with debug output."""

import sys
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import set_vm_weights, Token
from src.compiler import compile_c

print("Creating runner...")
runner = AutoregressiveVMRunner()

print("Setting weights...")
set_vm_weights(runner.model)

print("Compiling program...")
source = "int main() { return 0; }"
bytecode, data = compile_c(source)
print(f"Bytecode: {bytecode}")

print("\nManually running the logic from runner.run()...")

# Initialize state (from run() method)
runner._bytecode = bytecode
runner._stdin_buffer = []
runner._stdin_pos = 0
runner._tool_handler = None
runner._tool_call_id = 0
runner._last_pc = None
runner._last_bp = 0
runner._last_sp = 0
runner._mem_history = {}
runner.model._mem_history_end = 0

# Load data section
data = data or b""
if isinstance(data, (bytes, bytearray)):
    for i, b in enumerate(data):
        runner._memory[0x10000 + i] = b

data_end = 0x10000 + len(data)
runner._heap_base = (data_end + 7) & ~7
runner._heap_ptr = runner._heap_base

print("Building context...")
sys.stdout.flush()
context = runner._build_context(bytecode, data, [], "")
print(f"Context built: {len(context)} tokens")
sys.stdout.flush()

prefix_len = len(context)
output = []

print("Setting initial opcode for MoE...")
sys.stdout.flush()
init_exec = runner._exec_pc() // 5
if 0 <= init_exec < len(bytecode):
    runner.model.set_active_opcode(bytecode[init_exec] & 0xFF)
print(f"Initial exec PC: {runner._exec_pc()}, opcode index: {init_exec}")
sys.stdout.flush()

print("\nGenerating first 5 tokens...")
sys.stdout.flush()
max_steps = 2
for i in range(5):
    print(f"  Token {i}...")
    sys.stdout.flush()
    next_token = runner.model.generate_next(context)
    context.append(next_token)
    print(f"    Generated: {next_token}")
    sys.stdout.flush()

print(f"\nSuccess! Generated {len(context) - prefix_len} tokens")
