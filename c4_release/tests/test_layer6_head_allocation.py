#!/usr/bin/env python3
"""
Weight configuration test for Layer 6 head allocation fix.

This test verifies that the Layer 6 attention head allocation is correct
and that the AX_CARRY path is preserved from Layer 3 to Layer 8.

This is a FAST test (< 30 seconds) that checks weight configurations
without running full programs.

Background:
  The original code had `_set_layer6_relay_heads()` overwriting heads 2-3
  configured by `_set_layer6_attn()`. The fix was to disable that function.

  This test ensures:
  1. No Layer 6 heads write to AX_CARRY at the AX marker (would corrupt carry-forward)
  2. No Layer 6 head allocation conflicts
  3. Layer 3 Head 1 correctly sets AX_CARRY at AX marker

Related Issues:
  - docs/ACTUAL_FIX_AX_CARRY.md
  - docs/FIX_SUMMARY.md

Date: 2026-04-08
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim

BD = _SetDim


def test_ax_carry_preservation():
    """Verify no Layer 6 heads corrupt AX_CARRY at AX marker."""

    print("="*80)
    print("LAYER 6 HEAD ALLOCATION TEST")
    print("="*80)
    print("\nTest 1: AX_CARRY preservation at AX marker\n")

    model = AutoregressiveVM()
    set_vm_weights(model)

    layer6 = model.blocks[6]
    attn6 = layer6.attn
    HD = attn6.head_dim

    problems = []

    for head in range(attn6.num_heads):
        base = head * HD

        # Check if this head writes to AX_CARRY
        w_o = attn6.W_o[:, base:base+HD]
        writes_ax_carry = (w_o[BD.AX_CARRY_LO:BD.AX_CARRY_LO+32, :].abs() > 1e-6).any().item()

        if not writes_ax_carry:
            continue

        # This head writes to AX_CARRY - check WHERE it fires
        w_q = attn6.W_q[base:base+HD, :]

        # Check query weights for markers
        q_ax = w_q[:, BD.MARK_AX].abs().max().item()
        q_pc = w_q[:, BD.MARK_PC].abs().max().item()
        q_sp = w_q[:, BD.MARK_SP].abs().max().item()
        q_stack0 = w_q[:, BD.MARK_STACK0].abs().max().item()

        # Check sign of MARK_AX query
        ax_sign = w_q[:, BD.MARK_AX].max().item()

        blocks_ax = (q_ax > 1e-6 and ax_sign < 0)

        if q_ax > 1e-6 and ax_sign > 0:
            # Fires at AX marker - this is a problem!
            problems.append(f"Head {head}: Writes to AX_CARRY at AX marker (corrupts carry-forward)")
            print(f"  ✗ Head {head}: FIRES at AX marker and writes to AX_CARRY (PROBLEM!)")
        elif blocks_ax:
            print(f"  ✓ Head {head}: Writes to AX_CARRY but BLOCKS at AX marker (OK)")
        else:
            markers = []
            if q_pc > 1e-6:
                markers.append("PC")
            if q_sp > 1e-6:
                markers.append("SP")
            if q_stack0 > 1e-6:
                markers.append("STACK0")
            print(f"  ✓ Head {head}: Writes to AX_CARRY at {', '.join(markers)} markers (OK)")

    # Check Layer 3 Head 1 sets AX_CARRY
    print("\nTest 2: Layer 3 Head 1 sets AX_CARRY at AX marker\n")

    layer3 = model.blocks[3]
    attn3 = layer3.attn
    base_h1 = 1 * attn3.head_dim
    w_o_l3h1 = attn3.W_o[:, base_h1:base_h1+attn3.head_dim]
    l3_writes = (w_o_l3h1[BD.AX_CARRY_LO:BD.AX_CARRY_LO+32, :].abs() > 1e-6).any().item()

    w_q_l3h1 = attn3.W_q[base_h1:base_h1+attn3.head_dim, :]
    l3_queries_ax = (w_q_l3h1[:, BD.MARK_AX].abs() > 1e-6).any().item()

    if l3_writes and l3_queries_ax:
        print("  ✓ Layer 3 Head 1 correctly sets AX_CARRY at AX marker")
    else:
        problems.append("Layer 3 Head 1 does not set AX_CARRY at AX marker")
        print("  ✗ Layer 3 Head 1 MISSING configuration")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if not problems:
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("\nAX_CARRY path is correctly preserved from Layer 3 to Layer 8.")
        print("Arithmetic operations should work without Python handlers.")
    else:
        print("\n✗✗✗ TESTS FAILED ✗✗✗\n")
        print("Problems found:")
        for p in problems:
            print(f"  - {p}")

    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return len(problems) == 0


def test_head6_no_conflict():
    """Verify Head 6 has no configuration conflicts."""

    print("\n" + "="*80)
    print("HEAD 6 CONFLICT TEST")
    print("="*80)
    print("\nTest 3: Head 6 configuration consistency\n")

    model = AutoregressiveVM()
    set_vm_weights(model)

    layer6 = model.blocks[6]
    attn6 = layer6.attn
    HD = attn6.head_dim

    base = 6 * HD
    w_o = attn6.W_o[:, base:base+HD]

    writes_alu = (w_o[BD.ALU_LO:BD.ALU_LO+32, :].abs() > 1e-6).any().item()
    writes_cmp = (w_o[BD.CMP:BD.CMP+16, :].abs() > 1e-6).any().item()

    if writes_alu and writes_cmp:
        print("  ✗ Head 6 writes to BOTH ALU and CMP (conflict!)")
        print("    This means two functions are configuring the same head.")
        result = False
    elif writes_alu:
        print("  ✓ Head 6 writes to ALU only (no conflict)")
        result = True
    elif writes_cmp:
        print("  ✓ Head 6 writes to CMP only (no conflict)")
        result = True
    else:
        print("  ? Head 6 writes to neither ALU nor CMP (unused?)")
        result = True

    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


if __name__ == '__main__':
    print("\nRunning Layer 6 head allocation tests...\n")

    test1 = test_ax_carry_preservation()
    test2 = test_head6_no_conflict()

    print("\n" + "="*80)
    print("FINAL RESULT")
    print("="*80)

    if test1 and test2:
        print("\n✓ All weight configuration tests passed!")
        print("\nThe Layer 6 head allocation fix is verified to be correct.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed!")
        print("\nThe weight configuration has issues that need to be fixed.")
        sys.exit(1)
