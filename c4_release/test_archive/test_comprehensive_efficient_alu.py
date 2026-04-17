"""Comprehensive test of all efficient ALU operations integration."""

import torch
from neural_vm.vm_step import AutoregressiveVM, _SetDim, set_vm_weights
from neural_vm.efficient_alu_integrated import integrate_efficient_alu

print("="*70)
print("Comprehensive Efficient ALU Integration Test")
print("="*70)

# Build model with standard weights
print("\nBuilding model...")
model = AutoregressiveVM()
set_vm_weights(model)

BD = _SetDim
S = 100.0

# Integrate all efficient ALU operations
print("\n" + "="*70)
stats = integrate_efficient_alu(model, S, BD)
print("="*70)

# Test forward pass
print("\nTesting forward pass...")
try:
    x = torch.randint(0, 256, (1, 10))
    with torch.no_grad():
        output = model(x)
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()

# Verify specific operations if forward pass succeeded
print("\n" + "="*70)
print("Verifying individual operations...")

# Test helper to create simple ALU operation tokens
def create_alu_test(bd_opcode_dim, a_val, b_val):
    """Create a token sequence for testing an ALU operation."""
    x = torch.zeros(1, 1, 512)

    # Set AX marker
    x[0, 0, BD.MARK_AX] = 1.0

    # Set operand A (ALU_LO, ALU_HI)
    x[0, 0, BD.ALU_LO + (a_val & 0xF)] = 1.0
    x[0, 0, BD.ALU_HI + ((a_val >> 4) & 0xF)] = 1.0

    # Set operand B (AX_CARRY_LO, AX_CARRY_HI)
    x[0, 0, BD.AX_CARRY_LO + (b_val & 0xF)] = 1.0
    x[0, 0, BD.AX_CARRY_HI + ((b_val >> 4) & 0xF)] = 1.0

    # Set opcode using BD dimension
    x[0, 0, bd_opcode_dim] = 1.0

    return x

# Test each operation using actual BD dimension indices
tests = [
    ("ADD", BD.OP_ADD, 5, 3, 5+3, 8),
    ("SUB", BD.OP_SUB, 10, 3, 10-3, 8),
    ("AND", BD.OP_AND, 0x0F, 0x33, 0x0F & 0x33, 10),
    ("OR", BD.OP_OR, 0x0F, 0x30, 0x0F | 0x30, 10),
    ("XOR", BD.OP_XOR, 0xFF, 0x55, 0xFF ^ 0x55, 10),
    ("MUL", BD.OP_MUL, 5, 3, (5*3) & 0xFF, 11),  # Lower 8 bits only
]

print("\nOperation tests:")
for name, bd_dim, a, b, expected, layer_num in tests:
    try:
        x = create_alu_test(bd_dim, a, b)
        with torch.no_grad():
            # Run through the specific layer
            result = model.blocks[layer_num].ffn(x)

        # Extract result
        result_lo = result[0, 0, BD.OUTPUT_LO:BD.OUTPUT_LO+16].argmax().item()
        result_hi = result[0, 0, BD.OUTPUT_HI:BD.OUTPUT_HI+16].argmax().item()
        result_val = result_lo + (result_hi << 4)

        # Compare
        if result_val == expected:
            print(f"  ✓ {name}: {a} op {b} = {result_val} (expected {expected})")
        else:
            print(f"  ✗ {name}: {a} op {b} = {result_val} (expected {expected})")
    except Exception as e:
        print(f"  ✗ {name}: Error - {e}")

# Test SHIFT operations
print("\nSHIFT tests:")
shift_tests = [
    ("SHL", BD.OP_SHL, 0x12, 4, (0x12 << 4) & 0xFF),
    ("SHR", BD.OP_SHR, 0x80, 4, 0x80 >> 4),
]

for name, bd_dim, val, shift, expected in shift_tests:
    try:
        x = torch.zeros(1, 1, 512)
        x[0, 0, BD.MARK_AX] = 1.0
        x[0, 0, BD.ALU_LO + (val & 0xF)] = 1.0
        x[0, 0, BD.ALU_HI + ((val >> 4) & 0xF)] = 1.0
        x[0, 0, BD.AX_CARRY_LO + shift] = 1.0  # Shift amount
        x[0, 0, bd_dim] = 1.0

        with torch.no_grad():
            result = model.blocks[13].ffn(x)

        result_lo = result[0, 0, BD.OUTPUT_LO:BD.OUTPUT_LO+16].argmax().item()
        result_hi = result[0, 0, BD.OUTPUT_HI:BD.OUTPUT_HI+16].argmax().item()
        result_val = result_lo + (result_hi << 4)

        if result_val == expected:
            print(f"  ✓ {name}: {val:#04x} shift {shift} = {result_val:#04x} (expected {expected:#04x})")
        else:
            print(f"  ✗ {name}: {val:#04x} shift {shift} = {result_val:#04x} (expected {expected:#04x})")
    except Exception as e:
        print(f"  ✗ {name}: Error - {e}")

print("\n" + "="*70)
print("Test Summary")
print("="*70)

# Calculate final stats
total_before = sum(stats['before'].values())
total_after = sum(stats['after'].values())
total_savings = total_before - total_after
total_pct = total_savings / total_before * 100 if total_before > 0 else 0

print(f"\nTotal parameter reduction: {total_savings:,} params ({total_pct:.1f}%)")
print(f"From {total_before:,} → {total_after:,} params")

# Count integrated layers
integrated = []
for i in [8, 10, 11, 13]:
    if stats['before'][f'L{i}'] > stats['after'][f'L{i}']:
        integrated.append(i)

print(f"\nIntegrated layers: L{', L'.join(map(str, integrated))}")
print(f"Operations covered: ADD, SUB, MUL, AND, OR, XOR, SHL, SHR")
print("\n" + "="*70)
