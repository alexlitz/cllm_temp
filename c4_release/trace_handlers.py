#!/usr/bin/env python3
"""Trace handler calls."""
import sys
sys.path.insert(0, '.')
import torch

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token
from neural_vm.constants import INSTR_WIDTH

code = 'int main() { return 42; }'
bytecode, _ = compile_c(code)

runner = AutoregressiveVMRunner()
if torch.cuda.is_available():
    runner.model.cuda()

# Wrap handlers to see if they're called
original_jsr = runner._handler_jsr
original_ent = runner._handler_ent
calls = []

def debug_jsr(context, output):
    pc_before = runner._extract_register(context, Token.REG_PC)
    calls.append(f"JSR handler: PC_before={pc_before}")
    original_jsr(context, output)
    pc_after = runner._extract_register(context, Token.REG_PC)
    calls.append(f"  → PC_after={pc_after}")

def debug_ent(context, output):
    pc_before = runner._extract_register(context, Token.REG_PC)
    calls.append(f"ENT handler: PC_before={pc_before}")
    original_ent(context, output)
    pc_after = runner._extract_register(context, Token.REG_PC)
    calls.append(f"  → PC_after={pc_after}")

runner._handler_jsr = debug_jsr
runner._handler_ent = debug_ent

print("Running program...")
result = runner.run(bytecode, max_steps=10)
print(f"Result: {result}")

print(f"\nHandler calls: {len([c for c in calls if 'handler' in c])}")
for call in calls[:10]:  # First 10 entries
    print(f"  {call}")
