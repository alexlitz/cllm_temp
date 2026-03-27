"""Debug bitwise operations."""

import torch
from neural_vm.vm_step import AutoregressiveVM, _SetDim, set_vm_weights
from neural_vm.efficient_alu_integrated import EfficientALU_L10

print("="*70)
print("Debugging Bitwise Operations")
print("="*70)

BD = _SetDim
S = 100.0

# Create the bitwise module
bitwise = EfficientALU_L10(S, BD)

# Create a simple test: 0x0F AND 0x33 = 0x03
print("\nTest: 0x0F AND 0x33 = 0x03")
x = torch.zeros(1, 1, 512)
x[0, 0, BD.MARK_AX] = 1.0
x[0, 0, BD.ALU_LO + 0xF] = 1.0  # 0x0F low nibble
x[0, 0, BD.ALU_HI + 0x0] = 1.0  # 0x0F high nibble
x[0, 0, BD.AX_CARRY_LO + 0x3] = 1.0  # 0x33 low nibble
x[0, 0, BD.AX_CARRY_HI + 0x3] = 1.0  # 0x33 high nibble
x[0, 0, BD.OPCODE_BASE + 30] = 1.0  # AND opcode

print(f"\nInput tensor:")
print(f"  MARK_AX: {x[0, 0, BD.MARK_AX].item()}")
print(f"  ALU_LO: argmax = {x[0, 0, BD.ALU_LO:BD.ALU_LO+16].argmax().item()}")
print(f"  ALU_HI: argmax = {x[0, 0, BD.ALU_HI:BD.ALU_HI+16].argmax().item()}")
print(f"  AX_CARRY_LO: argmax = {x[0, 0, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16].argmax().item()}")
print(f"  AX_CARRY_HI: argmax = {x[0, 0, BD.AX_CARRY_HI:BD.AX_CARRY_HI+16].argmax().item()}")
print(f"  OP_AND: {x[0, 0, BD.OP_AND].item()}")

# Test the bd_to_ge conversion
x_ge = bitwise.bd_to_ge(x[0, 0])
print(f"\nAfter bd_to_ge:")
print(f"  x_ge[0, NIB_A] = {x_ge[0, bitwise.ge.NIB_A].item()} (should be 15)")
print(f"  x_ge[1, NIB_A] = {x_ge[1, bitwise.ge.NIB_A].item()} (should be 0)")
print(f"  x_ge[0, NIB_B] = {x_ge[0, bitwise.ge.NIB_B].item()} (should be 3)")
print(f"  x_ge[1, NIB_B] = {x_ge[1, bitwise.ge.NIB_B].item()} (should be 3)")
print(f"  x_ge[0, OP_START+30] = {x_ge[0, bitwise.ge.OP_START + 30].item()} (should be 1.0)")

# Run through AND layers
print(f"\nRunning AND layers...")
x_ge_batch = x_ge.unsqueeze(0)
print(f"  Input shape: {x_ge_batch.shape}")

with torch.no_grad():
    for i, layer in enumerate(bitwise.and_layers):
        x_ge_batch = layer(x_ge_batch)
        print(f"  After layer {i}: shape={x_ge_batch.shape}")
        print(f"    x_ge[0, 0, RESULT] = {x_ge_batch[0, 0, bitwise.ge.RESULT].item()}")
        print(f"    x_ge[0, 1, RESULT] = {x_ge_batch[0, 1, bitwise.ge.RESULT].item()}")

x_ge_result = x_ge_batch.squeeze(0)

# Convert back
print(f"\nBefore ge_to_bd:")
print(f"  x_ge_result[0, RESULT] = {x_ge_result[0, bitwise.ge.RESULT].item()}")
print(f"  x_ge_result[1, RESULT] = {x_ge_result[1, bitwise.ge.RESULT].item()}")

x_out = x.clone()
bitwise.ge_to_bd(x_ge_result, x_out[0, 0])

print(f"\nAfter ge_to_bd:")
result_lo = x_out[0, 0, BD.OUTPUT_LO:BD.OUTPUT_LO+16].argmax().item()
result_hi = x_out[0, 0, BD.OUTPUT_HI:BD.OUTPUT_HI+16].argmax().item()
result_val = result_lo + (result_hi << 4)
print(f"  OUTPUT_LO: argmax = {result_lo}")
print(f"  OUTPUT_HI: argmax = {result_hi}")
print(f"  Result: {result_val:#04x} (expected 0x03)")

# Now test the full forward pass
print(f"\n" + "="*70)
print("Testing full forward pass")
print("="*70)

result = bitwise(x)
result_lo = result[0, 0, BD.OUTPUT_LO:BD.OUTPUT_LO+16].argmax().item()
result_hi = result[0, 0, BD.OUTPUT_HI:BD.OUTPUT_HI+16].argmax().item()
result_val = result_lo + (result_hi << 4)
print(f"Result: {result_val:#04x} (expected 0x03)")

if result_val == 0x03:
    print("✓ Test passed!")
else:
    print(f"✗ Test failed: got {result_val:#04x}, expected 0x03")
