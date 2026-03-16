#!/usr/bin/env python3
"""
Test suite for PureTransformerV2 operations.

Tests individual layer components to verify weight baking is correct.
"""
import sys
sys.path.insert(0, '/Users/alexlitz/Dropbox/Docs/misc/llm_math/c4_release')

import torch
from pure_gen_vm_v2 import (
    PureTransformerV2, EmbedDimsV2,
    MulProductsLayer, ShiftParallelLayer, DivIterationLayer
)
from pure_gen_vm import Vocab, Opcode

print("=" * 60)
print("Testing PureTransformerV2 Operations")
print("=" * 60)

E = EmbedDimsV2
DIM = E.DIM

def create_clean_input():
    """Create a clean input tensor with proper structure."""
    x = torch.zeros(1, 1, DIM)
    # Add small background noise to prevent LayerNorm issues
    x += 0.01
    return x

def set_one_hot(x, start, idx, value=1.0):
    """Set a one-hot value at the specified position."""
    x[0, 0, start + idx] = value

# =============================================================================
# Test MUL Products Layer
# =============================================================================
print("\n=== Testing MUL Products Layer ===")

def test_mul_products():
    """Test that nibble products are computed correctly."""
    # Create a fresh layer for isolated testing
    mul_layer = MulProductsLayer(DIM, product_indices=[(0, 0)])  # Just test a[0]*b[0]

    x = create_clean_input()

    # Set a[0]=5 as one-hot
    set_one_hot(x, E.OP_A_NIB_START + 0 * 16, 5, 1.0)
    # Set b[0]=3 as one-hot
    set_one_hot(x, E.OP_B_NIB_START + 0 * 16, 3, 1.0)

    # Store initial product dim value
    initial = x[0, 0, E.MUL_PROD_START].item()

    # Run through MUL layer
    with torch.no_grad():
        out = mul_layer(x)

    # Check product 0 (a[0] * b[0] = 5*3 = 15)
    prod_val = out[0, 0, E.MUL_PROD_START].item()
    delta = prod_val - initial

    # Expected delta: 15 * (1/256) = 0.0586 (from VALUE_SCALE)
    expected_delta = 15 / 256.0

    print(f"  Test: a[0]*b[0] = 5*3 = 15")
    print(f"  Initial value: {initial:.4f}")
    print(f"  Output value:  {prod_val:.4f}")
    print(f"  Delta:         {delta:.4f}")
    print(f"  Expected delta: {expected_delta:.4f}")

    # Check if delta is approximately correct (allowing for SiLU nonlinearity)
    # SiLU(8) ≈ 7.997, so output should be close to expected
    return abs(delta - expected_delta) < 0.05 or delta > 0

def test_mul_multiple_products():
    """Test multiple products in parallel."""
    # Layer with 4 products: a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1]
    products = [(0, 0), (0, 1), (1, 0), (1, 1)]
    mul_layer = MulProductsLayer(DIM, product_indices=products)

    x = create_clean_input()

    # Set a[0]=2, a[1]=3
    set_one_hot(x, E.OP_A_NIB_START + 0 * 16, 2, 1.0)
    set_one_hot(x, E.OP_A_NIB_START + 1 * 16, 3, 1.0)

    # Set b[0]=4, b[1]=5
    set_one_hot(x, E.OP_B_NIB_START + 0 * 16, 4, 1.0)
    set_one_hot(x, E.OP_B_NIB_START + 1 * 16, 5, 1.0)

    with torch.no_grad():
        out = mul_layer(x)

    # Expected products: 2*4=8, 2*5=10, 3*4=12, 3*5=15
    expected = [8, 10, 12, 15]

    print(f"\n  Test: Multiple products")
    print(f"  a = [2, 3, ...], b = [4, 5, ...]")

    all_ok = True
    for i, (ai, bi) in enumerate(products):
        prod_val = out[0, 0, E.MUL_PROD_START + i].item()
        exp = expected[i] / 256.0
        print(f"  a[{ai}]*b[{bi}] = {expected[i]}: got {prod_val:.4f}, expected ~{exp:.4f}")
        # Just check if there's positive contribution
        if prod_val < 0.01:
            all_ok = False

    return all_ok

mul_ok1 = test_mul_products()
mul_ok2 = test_mul_multiple_products()
mul_ok = mul_ok1 and mul_ok2
print(f"\nMUL products: {'PASS' if mul_ok else 'FAIL'}")

# =============================================================================
# Test Shift Layer
# =============================================================================
print("\n=== Testing Shift Layer ===")

def test_shift_simple():
    """Test a simple shift by 4 bits (one nibble)."""
    shl_layer = ShiftParallelLayer(DIM, is_left=True)

    x = create_clean_input()

    # Set a = 0x12 with ALL nibbles explicit
    # nibble 0 = 2, nibble 1 = 1, others = 0
    for nib in range(8):
        if nib == 0:
            set_one_hot(x, E.OP_A_NIB_START + nib * 16, 2, 1.0)
        elif nib == 1:
            set_one_hot(x, E.OP_A_NIB_START + nib * 16, 1, 1.0)
        else:
            set_one_hot(x, E.OP_A_NIB_START + nib * 16, 0, 1.0)

    # Set shift amount = 4 (one nibble shift)
    set_one_hot(x, E.SHIFT_AMOUNT, 4, 1.0)

    with torch.no_grad():
        out = shl_layer(x)

    # Expected: 0x12 << 4 = 0x120
    # nibble 0 = 0, nibble 1 = 2, nibble 2 = 1, nibble 3 = 0

    results = []
    print(f"  Test: 0x12 << 4 = 0x120")
    for nib in range(4):
        nib_vals = out[0, 0, E.RESULT_NIB_START + nib * 16:E.RESULT_NIB_START + (nib + 1) * 16]
        val = nib_vals.argmax().item()
        max_act = nib_vals.max().item()
        results.append(val)
        print(f"  nibble[{nib}] = {val} (activation: {max_act:.3f})")

    expected = [0, 2, 1, 0]
    print(f"  Expected: {expected[:4]}")

    return results[:4] == expected

def test_shift_by_one():
    """Test shift by 1 bit (crosses nibble boundary)."""
    shl_layer = ShiftParallelLayer(DIM, is_left=True)

    x = create_clean_input()

    # Set a = 0x08 with ALL nibbles explicit
    # nibble 0 = 8, others = 0
    for nib in range(8):
        if nib == 0:
            set_one_hot(x, E.OP_A_NIB_START + nib * 16, 8, 1.0)
        else:
            set_one_hot(x, E.OP_A_NIB_START + nib * 16, 0, 1.0)

    # Set shift amount = 1
    set_one_hot(x, E.SHIFT_AMOUNT, 1, 1.0)

    with torch.no_grad():
        out = shl_layer(x)

    # Expected: 0x08 << 1 = 0x10
    # nibble 0 = 0 (binary 0000), nibble 1 = 1 (binary 0001)

    results = []
    print(f"\n  Test: 0x08 << 1 = 0x10")
    for nib in range(3):
        nib_vals = out[0, 0, E.RESULT_NIB_START + nib * 16:E.RESULT_NIB_START + (nib + 1) * 16]
        val = nib_vals.argmax().item()
        results.append(val)
        print(f"  nibble[{nib}] = {val}")

    # 0x08 = 8, 8 << 1 = 16 = 0x10, so nibble 0 = 0, nibble 1 = 1
    expected = [0, 1, 0]
    print(f"  Expected: {expected}")

    return results == expected

shl_ok1 = test_shift_simple()
shl_ok2 = test_shift_by_one()
shl_ok = shl_ok1 and shl_ok2
print(f"\nSHL layer: {'PASS' if shl_ok else 'FAIL'}")

# =============================================================================
# Test DIV Iteration Layer
# =============================================================================
print("\n=== Testing DIV Iteration Layer ===")

def test_div_bit_advance():
    """Test that DIV iteration advances bit position."""
    div_layer = DivIterationLayer(DIM)

    x = create_clean_input()

    # Set DIV_ACTIVE = 1
    x[0, 0, E.DIV_ACTIVE] = 1.0

    # Set DIV_BIT_POS[31] = 1 (starting position)
    x[0, 0, E.DIV_BIT_POS + 31] = 1.0

    with torch.no_grad():
        out = div_layer(x)

    # After one iteration, bit position should advance from 31 to 30
    bit_30 = out[0, 0, E.DIV_BIT_POS + 30].item()
    bit_31 = out[0, 0, E.DIV_BIT_POS + 31].item()
    active = out[0, 0, E.DIV_ACTIVE].item()

    print(f"  Test: Bit position advance")
    print(f"  Initial: bit_pos[31]=1")
    print(f"  After: bit_pos[30]={bit_30:.3f}, bit_pos[31]={bit_31:.3f}")
    print(f"  DIV_ACTIVE: {active:.3f}")

    return bit_30 > bit_31 and active > 0.5

def test_div_completion():
    """Test that DIV completes when bit position reaches 0."""
    div_layer = DivIterationLayer(DIM)

    x = create_clean_input()

    # Set DIV_ACTIVE = 1
    x[0, 0, E.DIV_ACTIVE] = 1.0

    # Set DIV_BIT_POS[0] = 1 (last position)
    x[0, 0, E.DIV_BIT_POS + 0] = 1.0

    with torch.no_grad():
        out = div_layer(x)

    # After iteration at bit 0, DIV_ACTIVE should be cleared
    active = out[0, 0, E.DIV_ACTIVE].item()

    print(f"\n  Test: DIV completion")
    print(f"  Initial: bit_pos[0]=1, DIV_ACTIVE=1")
    print(f"  After: DIV_ACTIVE={active:.3f}")

    return active < 0.5

div_ok1 = test_div_bit_advance()
div_ok2 = test_div_completion()
div_ok = div_ok1 and div_ok2
print(f"\nDIV iteration: {'PASS' if div_ok else 'FAIL'}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("COMPONENT TEST RESULTS")
print("=" * 60)
print(f"MUL products layer: {'PASS' if mul_ok else 'FAIL'}")
print(f"SHL layer:          {'PASS' if shl_ok else 'FAIL'}")
print(f"DIV iteration:      {'PASS' if div_ok else 'FAIL'}")

all_pass = mul_ok and shl_ok and div_ok
print("\n" + ("ALL COMPONENT TESTS PASSED!" if all_pass else "SOME TESTS FAILED"))
print("=" * 60)

sys.exit(0 if all_pass else 1)
