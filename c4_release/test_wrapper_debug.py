"""Debug the wrapper conversion."""

import torch
from neural_vm.vm_step import _SetDim
from neural_vm.efficient_wrappers import EfficientShiftFFN
from neural_vm.alu.chunk_config import NIBBLE
from neural_vm.alu.ops.common import GenericE

BD = _SetDim
S = 100.0
ge = GenericE(NIBBLE)

# Create wrapper
wrapper = EfficientShiftFFN(S, BD)

# Test input: SHL 0x7A by 2
a = 0x7A
shift_amount = 2
expected = (a << shift_amount) & 0xFF

print(f"Test: SHL {a:#04x} by {shift_amount}")
print(f"Expected: {expected:#04x}")

# Create BD format input for single position
x_bd_single = torch.zeros(512)

# Set ALU_LO (low nibble = 0xA = 10)
x_bd_single[BD.ALU_LO + 10] = 1.0

# Set ALU_HI (high nibble = 0x7 = 7)
x_bd_single[BD.ALU_HI + 7] = 1.0

# Set shift amount in AX_CARRY_LO
x_bd_single[BD.AX_CARRY_LO + 2] = 1.0

# Set opcode SHL (opcode 23)
x_bd_single[BD.OPCODE_BASE + 23] = 1.0

# Set MARK_AX
x_bd_single[BD.MARK_AX] = 1.0

print(f"\nBD input summary:")
print(f"  ALU_LO active: {x_bd_single[BD.ALU_LO:BD.ALU_LO+16].argmax().item()}")
print(f"  ALU_HI active: {x_bd_single[BD.ALU_HI:BD.ALU_HI+16].argmax().item()}")
print(f"  AX_CARRY_LO active: {x_bd_single[BD.AX_CARRY_LO:BD.AX_CARRY_LO+16].argmax().item()}")
print(f"  OP_SHL: {x_bd_single[BD.OP_SHL].item()}")

# Convert to GenericE
x_ge = wrapper.bd_to_ge(x_bd_single)

print(f"\nGenericE after conversion:")
for pos in range(8):
    nib_a = x_ge[pos, ge.NIB_A].item()
    nib_b = x_ge[pos, ge.NIB_B].item()
    op_shl = x_ge[pos, ge.OP_START + 23].item()
    print(f"  pos {pos}: NIB_A={nib_a:.0f}, NIB_B={nib_b:.0f}, OP_SHL={op_shl:.1f}")

# Run efficient layers
x_ge_batch = x_ge.unsqueeze(0)  # [1, 8, 160]
print(f"\nRunning SHL layers...")
with torch.no_grad():
    for i, layer in enumerate(wrapper.shl_layers):
        x_ge_batch = layer(x_ge_batch)
        print(f"  After layer {i}: result[0]={x_ge_batch[0, 0, ge.RESULT].item():.2f}, result[1]={x_ge_batch[0, 1, ge.RESULT].item():.2f}")

x_ge_result = x_ge_batch.squeeze(0)

print(f"\nGenericE result:")
result_chunks = [x_ge_result[pos, ge.RESULT].item() for pos in range(8)]
print(f"  RESULT chunks: {result_chunks}")

# Convert back
x_bd_out = x_bd_single.clone()
wrapper.ge_to_bd(x_ge_result, x_bd_out)

print(f"\nBD output:")
output_lo = x_bd_out[BD.OUTPUT_LO:BD.OUTPUT_LO+16].argmax().item()
output_hi = x_bd_out[BD.OUTPUT_HI:BD.OUTPUT_HI+16].argmax().item()
result = (output_hi << 4) | output_lo

print(f"  OUTPUT_LO: {output_lo}")
print(f"  OUTPUT_HI: {output_hi}")
print(f"  Combined: {result:#04x}")
print(f"\n{'✓ PASS' if result == expected else '✗ FAIL'}")
