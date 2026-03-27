#!/usr/bin/env python3
"""Test that our PC formula fixes work correctly."""
import sys
sys.path.insert(0, '.')

from neural_vm.speculative import DraftVM
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from neural_vm.embedding import Opcode
import torch

def test_draft_vm_pc():
    """Test DraftVM PC formula."""
    print("Testing DraftVM PC formula...")
    bytecode = [
        Opcode.IMM | (42 << 8),  # instruction 0: PC should be 0 initially, then 8 after execution
        Opcode.PSH,              # instruction 1: PC should be 8 initially, then 16 after execution
    ]

    draft = DraftVM(bytecode)
    assert draft.pc == 0, f"Initial PC should be 0, got {draft.pc}"

    # Step 0: IMM 42
    draft.step()
    assert draft.pc == 8, f"After step 0, PC should be 8, got {draft.pc}"
    assert draft.ax == 42, f"After IMM 42, AX should be 42, got {draft.ax}"

    # Step 1: PSH
    draft.step()
    assert draft.pc == 16, f"After step 1, PC should be 16, got {draft.pc}"
    assert draft.sp == 0xFFFFFFF8, f"After PSH, SP should be -8, got {draft.sp:#x}"

    print("  ✓ DraftVM PC formula correct (idx*8)")
    return True

def test_neural_vm_step0():
    """Test neural VM step 0 (IMM instruction)."""
    print("\nTesting Neural VM step 0 (IMM 42)...")

    bytecode = [Opcode.IMM | (42 << 8)]

    # Build context
    context = [Token.CODE_START]
    for instr in bytecode:
        opcode = instr & 0xFF
        imm = (instr >> 8) & 0xFFFFFFFF
        context.append(opcode)
        for i in range(4):
            context.append((imm >> (i * 8)) & 0xFF)
        context.extend([0, 0, 0])
    context.extend([Token.CODE_END, Token.DATA_START, Token.DATA_END])

    # Get draft tokens from DraftVM
    draft = DraftVM(bytecode)
    draft.step()
    draft_tokens = draft.draft_tokens()

    # Full context with draft tokens
    full_context = context + draft_tokens

    # Run neural VM
    model = AutoregressiveVM()
    set_vm_weights(model)
    model.eval()

    device = model.embed.weight.device
    x = torch.tensor([full_context], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(x)

    # Check predictions (with autoregressive offset)
    pc_marker_pos = len(context)
    ax_marker_pos = pc_marker_pos + 5

    pc_pred = torch.argmax(logits[0, pc_marker_pos]).item()
    ax_pred = torch.argmax(logits[0, ax_marker_pos]).item()

    # Draft values
    pc_draft = draft_tokens[1]  # PC byte 0
    ax_draft = draft_tokens[6]  # AX byte 0

    print(f"  PC: neural={pc_pred}, draft={pc_draft}, expected=8")
    print(f"  AX: neural={ax_pred}, draft={ax_draft}, expected=42")

    assert pc_pred == pc_draft == 8, f"PC mismatch: neural={pc_pred}, draft={pc_draft}"
    assert ax_pred == ax_draft == 42, f"AX mismatch: neural={ax_pred}, draft={ax_draft}"

    print("  ✓ Neural VM step 0 correct")
    return True

def main():
    print("=" * 60)
    print("Testing PC Formula Fixes")
    print("=" * 60)

    try:
        test_draft_vm_pc()
        test_neural_vm_step0()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
