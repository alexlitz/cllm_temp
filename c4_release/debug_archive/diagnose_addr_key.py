#!/usr/bin/env python3
"""Diagnose ADDR_KEY setup in bytecode section."""

import sys
import torch
sys.path.insert(0, '.')

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.vm_step import _SetDim as BD, Token

def trace():
    runner = AutoregressiveVMRunner()
    model = runner.model

    bytecode = [
        Opcode.JMP | (16 << 8),   # addr 0
        Opcode.IMM | (99 << 8),   # addr 8
        Opcode.IMM | (42 << 8),   # addr 16 (0x10)
        Opcode.EXIT               # addr 24
    ]

    print("=== ADDR_KEY in Bytecode Section ===\n")

    # Build initial context
    context = runner._build_context(bytecode, b'', [])
    device = next(model.parameters()).device
    token_ids = torch.tensor([context], device=device)

    # Get embedding with augmentations
    x = model.embed(token_ids)

    # Find CODE_START and CODE_END markers
    code_start = None
    code_end = None
    for i, tok in enumerate(context):
        if tok == Token.CODE_START:
            code_start = i
        if tok == Token.CODE_END:
            code_end = i
            break

    print(f"CODE_START at position {code_start}")
    print(f"CODE_END at position {code_end}")
    print()

    # Print ADDR_KEY for each bytecode position
    if code_start is not None and code_end is not None:
        print("Position | Token | ADDR_KEY (lo, mid, hi) | Expected Addr")
        print("-" * 60)
        for pos in range(code_start + 1, code_end):
            tok = context[pos]
            addr_key_lo = x[0, pos, BD.ADDR_KEY:BD.ADDR_KEY+16].argmax().item()
            addr_key_mid = x[0, pos, BD.ADDR_KEY+16:BD.ADDR_KEY+32].argmax().item()
            addr_key_hi = x[0, pos, BD.ADDR_KEY+32:BD.ADDR_KEY+48].argmax().item()
            computed_addr = addr_key_hi * 256 + addr_key_mid * 16 + addr_key_lo

            # Expected address: (pos - code_start - 1) for sequential bytes
            expected_addr = pos - code_start - 1

            print(f"  {pos:3d}   | 0x{tok:02x}  | ({addr_key_lo:2d}, {addr_key_mid:2d}, {addr_key_hi:2d}) = {computed_addr:3d} | expected {expected_addr}")

    # Check specifically at address 0x10 (16)
    print("\n=== Address 0x10 (16) Check ===")
    target_addr = 16
    expected_pos = code_start + 1 + target_addr if code_start else None

    if expected_pos and expected_pos < len(context):
        tok = context[expected_pos]
        addr_key_lo = x[0, expected_pos, BD.ADDR_KEY:BD.ADDR_KEY+16].argmax().item()
        addr_key_mid = x[0, expected_pos, BD.ADDR_KEY+16:BD.ADDR_KEY+32].argmax().item()
        addr_key_hi = x[0, expected_pos, BD.ADDR_KEY+32:BD.ADDR_KEY+48].argmax().item()
        computed_addr = addr_key_hi * 256 + addr_key_mid * 16 + addr_key_lo

        print(f"Position {expected_pos}: token=0x{tok:02x}")
        print(f"ADDR_KEY = ({addr_key_lo}, {addr_key_mid}, {addr_key_hi}) = {computed_addr}")
        print(f"Expected addr = {target_addr}")

        if computed_addr == target_addr:
            print("ADDR_KEY is CORRECT for address 0x10!")
        else:
            print(f"ADDR_KEY MISMATCH! Got {computed_addr}, expected {target_addr}")

if __name__ == "__main__":
    trace()
