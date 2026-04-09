#!/usr/bin/env python3
"""Direct test of SI/LI using loaded compact model."""

import torch
from neural_vm.vm_step import AutoregressiveVM, Token
from neural_vm.embedding import Opcode

def test_si_li():
    print("=" * 80)
    print("Testing SI/LI with compact model")
    print("=" * 80)

    # Load compact model
    import os
    cache_path = os.path.join('neural_vm/tests', '.compact_moe_model.pt')

    if not os.path.exists(cache_path):
        print(f"ERROR: Cache file not found: {cache_path}")
        return False

    print(f"\nLoading model from {cache_path}...")
    model = AutoregressiveVM.load_compact(cache_path)
    model.eval()

    if torch.cuda.is_available():
        print("Moving to GPU...")
        model = model.cuda()

    # Test: Store 42 at 0x100, load it back
    addr = 0x100
    value = 42

    bytecode = [
        Opcode.IMM | (addr << 8),   # AX = addr
        Opcode.PSH,                  # push addr
        Opcode.IMM | (value << 8),   # AX = value
        Opcode.SI,                   # memory[addr] = value
        Opcode.IMM | (addr << 8),    # AX = addr
        Opcode.LI,                   # AX = memory[addr]
        Opcode.EXIT,
    ]

    print(f"\nBytecode: SI/LI round-trip")
    print(f"  Store {value} at address 0x{addr:04x}")
    print(f"  Load from address 0x{addr:04x}")
    print(f"  Expected exit code: {value}")

    # Build context
    from neural_vm.run_vm import AutoregressiveVMRunner
    runner = AutoregressiveVMRunner()
    runner.model = model
    context = runner._build_context(bytecode, b'', [])

    print(f"\nContext length: {len(context)} tokens")

    # Generate tokens
    print("\nGenerating tokens...")
    max_tokens = 10 * Token.STEP_TOKENS + 50
    generated = []

    for i in range(max_tokens):
        tok = model.generate_next(context)
        context.append(tok)
        generated.append(tok)

        if tok == Token.HALT:
            print(f"  HALT at token {i+1}")
            break

        # Show first few tokens
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

    # Extract exit code
    print("\nExtracting exit code...")
    exit_code = None
    for i in range(len(context) - 1, -1, -1):
        if context[i] == Token.REG_AX and i + 4 < len(context):
            exit_code = sum(context[i + 1 + j] << (j * 8) for j in range(4))
            break

    print(f"Exit code: {exit_code}")

    # Check result
    print("\n" + "=" * 80)
    if exit_code == value:
        print(f"✓ PASS: SI/LI works! Exit code = {exit_code}")
        return True
    else:
        print(f"✗ FAIL: Expected {value}, got {exit_code} (0x{exit_code:08x})")
        if exit_code == 0x01010101:
            print("\n⚠️  ERROR: byte_01 loop detected")
        return False

def test_imm_exit():
    """Test basic IMM+EXIT to verify model works at all."""
    print("\n" + "=" * 80)
    print("Testing IMM+EXIT (baseline)")
    print("=" * 80)

    import os
    cache_path = os.path.join('neural_vm/tests', '.compact_moe_model.pt')
    model = AutoregressiveVM.load_compact(cache_path)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    value = 42
    bytecode = [
        Opcode.IMM | (value << 8),
        Opcode.EXIT,
    ]

    print(f"\nBytecode: IMM {value}; EXIT")
    print(f"  Expected exit code: {value}")

    from neural_vm.run_vm import AutoregressiveVMRunner
    runner = AutoregressiveVMRunner()
    runner.model = model
    context = runner._build_context(bytecode, b'', [])

    # Generate
    for i in range(5 * Token.STEP_TOKENS + 10):
        tok = model.generate_next(context)
        context.append(tok)
        if tok == Token.HALT:
            break

    # Extract exit code
    exit_code = None
    for i in range(len(context) - 1, -1, -1):
        if context[i] == Token.REG_AX and i + 4 < len(context):
            exit_code = sum(context[i + 1 + j] << (j * 8) for j in range(4))
            break

    print(f"Exit code: {exit_code}")

    if exit_code == value:
        print(f"✓ PASS: IMM+EXIT works")
        return True
    else:
        print(f"✗ FAIL: IMM+EXIT broken (expected {value}, got {exit_code})")
        return False


if __name__ == "__main__":
    # Test basic operation first
    baseline_ok = test_imm_exit()

    if not baseline_ok:
        print("\n" + "!" * 80)
        print("CRITICAL: Even basic IMM+EXIT fails!")
        print("The model itself is broken, not just memory operations.")
        print("!" * 80)
        exit(1)

    # Test memory operations
    success = test_si_li()
    exit(0 if success else 1)
