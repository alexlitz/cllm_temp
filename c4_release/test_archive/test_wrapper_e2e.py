"""End-to-end test of SHIFT wrapper with BD format."""

import torch
import sys
sys.path.insert(0, '.')

from neural_vm.vm_step import _SetDim
from neural_vm.efficient_wrappers import EfficientShiftFFN

# Create wrapper
S = 100.0
BD = _SetDim()
shift_ffn = EfficientShiftFFN(S, BD)

# Create BD format input for: SHL 0x7A by 2 = 0xE8
x_bd = torch.zeros(1, 1, 512)  # [batch=1, seq_len=1, 512]

# Set MARK_AX
x_bd[0, 0, BD.MARK_AX] = 1.0

# Set operand A = 0x7A (lo=10, hi=7) as one-hot
x_bd[0, 0, BD.ALU_LO + 10] = 1.0  # 0xA
x_bd[0, 0, BD.ALU_HI + 7] = 1.0   # 0x7

# Set shift amount = 2 as one-hot
x_bd[0, 0, BD.AX_CARRY_LO + 2] = 1.0

# Set SHL opcode
x_bd[0, 0, BD.OP_SHL] = 1.0  # Active

print("Test: SHL 0x7A by 2")
print(f"Input A: hi=7, lo=10")
print(f"Shift amount: 2")
print(f"Expected: 0xE8 (hi=14, lo=8)")

# Run forward pass
with torch.no_grad():
    x_out = shift_ffn(x_bd)

# Extract result from OUTPUT_LO/HI
result_lo = x_out[0, 0, BD.OUTPUT_LO:BD.OUTPUT_LO+16].argmax().item()
result_hi = x_out[0, 0, BD.OUTPUT_HI:BD.OUTPUT_HI+16].argmax().item()

result = (result_hi << 4) | result_lo if result_lo >= 0 and result_hi >= 0 else -1

print(f"\nResult: hi={result_hi}, lo={result_lo}")
print(f"Combined: 0x{result:02X}" if result >= 0 else "No result")
print(f"{'✓ PASS' if result == 0xE8 else '✗ FAIL'}")
