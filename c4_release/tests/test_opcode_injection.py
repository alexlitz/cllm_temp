"""Test if ACTIVE_OPCODE_PRTF is injected correctly."""

import sys
import os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token, _SetDim as BD

runner = AutoregressiveVMRunner(conversational_io=True)

# Test context
context = torch.tensor([[
    Token.REG_PC,
    10,  # byte
    Token.REG_AX,
]], dtype=torch.long)

print("Test 1: Without active_opcode")
with torch.no_grad():
    x = runner.model.embed(context, active_opcode=None)
    val = x[0, 0, BD.ACTIVE_OPCODE_PRTF].item()
    print(f"  ACTIVE_OPCODE_PRTF: {val:.2f} (expected 0.00)")

print("\nTest 2: With active_opcode=33 (PRTF)")
with torch.no_grad():
    x = runner.model.embed(context, active_opcode=33)
    val = x[0, 0, BD.ACTIVE_OPCODE_PRTF].item()
    print(f"  ACTIVE_OPCODE_PRTF: {val:.2f} (expected 1.00)")

print("\nTest 3: With active_opcode=31 (READ)")
with torch.no_grad():
    x = runner.model.embed(context, active_opcode=31)
    prtf_val = x[0, 0, BD.ACTIVE_OPCODE_PRTF].item()
    read_val = x[0, 0, BD.ACTIVE_OPCODE_READ].item()
    print(f"  ACTIVE_OPCODE_PRTF: {prtf_val:.2f} (expected 0.00)")
    print(f"  ACTIVE_OPCODE_READ: {read_val:.2f} (expected 1.00)")

# Summary
test1_pass = abs(val) < 0.1
test2_pass = abs(val - 1.0) < 0.1
test3_pass = abs(prtf_val) < 0.1 and abs(read_val - 1.0) < 0.1

if test2_pass:
    print("\n✅ ACTIVE_OPCODE injection is working!")
else:
    print("\n❌ ACTIVE_OPCODE injection is broken")
