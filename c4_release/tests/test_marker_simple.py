"""Test if THINKING markers are injected correctly."""

import sys
import os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token, _SetDim as BD

print("Creating runner...")
runner = AutoregressiveVMRunner(conversational_io=True)

# Test tokens: REG_PC, byte, THINKING_START, THINKING_END
context = torch.tensor([[
    Token.REG_PC,
    10,  # byte
    Token.THINKING_START,
    Token.THINKING_END,
]], dtype=torch.long)

print(f"Context tokens: {context[0].tolist()}")
print(f"  Token 0: REG_PC ({Token.REG_PC})")
print(f"  Token 1: byte 10")
print(f"  Token 2: THINKING_START ({Token.THINKING_START})")
print(f"  Token 3: THINKING_END ({Token.THINKING_END})")

print("\nRunning embedding...")
with torch.no_grad():
    x = runner.model.embed(context, active_opcode=None)

print("\nChecking markers:")
for i in range(4):
    ts_val = x[0, i, BD.MARK_THINKING_START].item()
    te_val = x[0, i, BD.MARK_THINKING_END].item()
    print(f"  Position {i}: TS={ts_val:.2f}, TE={te_val:.2f}")

ts_correct = x[0, 2, BD.MARK_THINKING_START].item() > 0.9
te_correct = x[0, 3, BD.MARK_THINKING_END].item() > 0.9
others_zero = (x[0, 0, BD.MARK_THINKING_START].item() < 0.1 and
               x[0, 1, BD.MARK_THINKING_START].item() < 0.1)

if ts_correct and te_correct and others_zero:
    print("\n✅ Markers are set correctly!")
else:
    print(f"\n❌ Markers are wrong!")
    print(f"   TS at pos 2: {x[0, 2, BD.MARK_THINKING_START].item():.2f} (expected 1.0)")
    print(f"   TE at pos 3: {x[0, 3, BD.MARK_THINKING_END].item():.2f} (expected 1.0)")
