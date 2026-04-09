"""Check if L5 FFN conversational_io weights are set."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import _SetDim as BD

runner = AutoregressiveVMRunner(conversational_io=True)

ffn5 = runner.model.blocks[5].ffn

# Check unit 410 (PRTF detection)
unit = 410

print("L5 FFN Unit 410 (PRTF detection):")
print(f"  W_up[{unit}, ACTIVE_OPCODE_PRTF={BD.ACTIVE_OPCODE_PRTF}]: {ffn5.W_up[unit, BD.ACTIVE_OPCODE_PRTF].item():.2f}")
print(f"  b_up[{unit}]: {ffn5.b_up[unit].item():.2f}")
print(f"  W_gate[{unit}, :] non-zero: {(ffn5.W_gate[unit] != 0).sum().item()} dims")
print(f"  b_gate[{unit}]: {ffn5.b_gate[unit].item():.2f}")
print(f"  W_down[IO_IS_PRTF={BD.IO_IS_PRTF}, {unit}]: {ffn5.W_down[BD.IO_IS_PRTF, unit].item():.2f}")

if ffn5.W_up[unit, BD.ACTIVE_OPCODE_PRTF].item() == 0:
    print("\n❌ ERROR: W_up is not set! Conversational I/O weights may not be applied.")
else:
    print("\n✓ Weights are set correctly.")
