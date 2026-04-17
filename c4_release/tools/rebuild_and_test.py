#!/usr/bin/env python3
"""Rebuild model cache from scratch and test."""

import torch
import os
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from neural_vm.embedding import Opcode

def build_and_save_model():
    """Build model with weights and save to cache."""
    print("=" * 80)
    print("Building fresh model from scratch...")
    print("=" * 80)

    model = AutoregressiveVM()
    print("\nSetting VM weights...")
    set_vm_weights(model)

    print("Compacting model...")
    model.compact(block_size=32)

    print("Compacting MoE...")
    model.compact_moe()

    model.eval()

    cache_path = 'neural_vm/tests/.compact_moe_model.pt'
    print(f"\nSaving to {cache_path}...")
    model.save_compact(cache_path)

    # Check file size
    size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    print(f"Cache file size: {size_mb:.1f} MB")

    return model


def test_model(model):
    """Test that model works for basic IMM+EXIT."""
    print("\n" + "=" * 80)
    print("Testing IMM+EXIT...")
    print("=" * 80)

    if torch.cuda.is_available():
        print("Moving to GPU...")
        model = model.cuda()

    value = 42
    bytecode = [
        Opcode.IMM | (value << 8),
        Opcode.EXIT,
    ]

    print(f"\nBytecode: IMM {value}; EXIT")

    from neural_vm.run_vm import AutoregressiveVMRunner
    runner = AutoregressiveVMRunner()
    runner.model = model
    context = runner._build_context(bytecode, b'', [])

    print(f"Context: {len(context)} tokens")

    # Generate
    generated = []
    for i in range(5 * Token.STEP_TOKENS + 10):
        tok = model.generate_next(context)
        context.append(tok)
        generated.append(tok)

        # Show first tokens
        if i < 50:
            token_name = None
            for name, val in vars(Token).items():
                if isinstance(val, int) and val == tok and name.isupper():
                    token_name = name
                    break
            if token_name:
                print(f"  [{i}] {token_name}")
            elif tok < 256:
                print(f"  [{i}] byte_{tok:02x}")

        if tok == Token.HALT:
            print(f"HALT at token {i}")
            break

    # Extract exit code
    exit_code = None
    for i in range(len(context) - 1, -1, -1):
        if context[i] == Token.REG_AX and i + 4 < len(context):
            exit_code = sum(context[i + 1 + j] << (j * 8) for j in range(4))
            break

    print(f"Exit code: {exit_code}")

    if exit_code == value:
        print(f"✓ PASS: Model works correctly!")
        return True
    else:
        print(f"✗ FAIL: Expected {value}, got {exit_code}")
        return False


if __name__ == "__main__":
    model = build_and_save_model()
    success = test_model(model)

    if success:
        print("\n" + "=" * 80)
        print("SUCCESS: Fresh model built and verified!")
        print("=" * 80)
        exit(0)
    else:
        print("\n" + "=" * 80)
        print("FAILURE: Fresh model does not work!")
        print("=" * 80)
        exit(1)
