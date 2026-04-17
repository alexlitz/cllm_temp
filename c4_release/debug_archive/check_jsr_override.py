#!/usr/bin/env python3
"""Check if JSR override is applied."""
import sys
sys.path.insert(0, '.')
import torch

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token

code = 'int main() { return 42; }'
bytecode, _ = compile_c(code)

runner = AutoregressiveVMRunner()
if torch.cuda.is_available():
    runner.model.cuda()

runner._bytecode = bytecode
context = runner._build_context(bytecode, b'', [])
output = []

print("Step 0: Generate without handler")
for i in range(40):
    next_token = runner.model.generate_next(context)
    context.append(next_token)
    if next_token == Token.STEP_END:
        break

pc_before = runner._extract_register(context, Token.REG_PC)
sp_before = runner._extract_register(context, Token.REG_SP)
print(f"Before JSR handler: PC={pc_before}, SP={sp_before}")

# Now call the JSR handler
runner._last_pc = None  # First step
runner._handler_jsr(context, output)

pc_after = runner._extract_register(context, Token.REG_PC)
sp_after = runner._extract_register(context, Token.REG_SP)
print(f"After JSR handler:  PC={pc_after}, SP={sp_after}")

print(f"\nJSR handler changed PC: {pc_before} → {pc_after}")
print(f"JSR handler changed SP: {sp_before} → {sp_after}")
