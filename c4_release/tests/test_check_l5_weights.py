"""Check if L5 FFN conversational I/O weights are set."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import _SetDim as BD

print("Creating runner with conversational_io=True...")
runner = AutoregressiveVMRunner(conversational_io=True)

print("\nChecking L5 FFN weights...")
l5_ffn = runner.model.blocks[5].ffn

# Check unit 410 (PRTF detection)
unit = 410
w_up = l5_ffn.W_up[unit, BD.ACTIVE_OPCODE_PRTF].item()
b_up = l5_ffn.b_up[unit].item()
b_gate = l5_ffn.b_gate[unit].item()
w_down = l5_ffn.W_down[BD.IO_IS_PRTF, unit].item()

print(f"Unit 410 (PRTF detection):")
print(f"  W_up[{unit}, ACTIVE_OPCODE_PRTF={BD.ACTIVE_OPCODE_PRTF}] = {w_up:.2f}")
print(f"  b_up[{unit}] = {b_up:.2f}")
print(f"  b_gate[{unit}] = {b_gate:.2f}")
print(f"  W_down[IO_IS_PRTF={BD.IO_IS_PRTF}, {unit}] = {w_down:.4f}")

if abs(w_up - 100.0) < 0.1:
    print("\n✅ L5 FFN weights are set correctly!")
else:
    print(f"\n❌ L5 FFN weights are NOT set (expected W_up=100.0, got {w_up:.2f})")
