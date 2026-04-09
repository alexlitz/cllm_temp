#!/usr/bin/env python3
"""Debug PSH SP - trace all steps."""
import os
import sys
sys.path.insert(0, os.getcwd())

from neural_vm.vm_step import Token
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

# Test program: IMM 0, PSH, IMM 0, MUL, EXIT
bytecode = [Opcode.IMM | (0 << 8), Opcode.PSH, Opcode.IMM | (0 << 8), Opcode.MUL, Opcode.EXIT]
print(f"Bytecode: {bytecode}")
print(f"Instructions: IMM 0, PSH, IMM 0, MUL, EXIT")

vm = DraftVM(bytecode)

for step in range(5):
    if not vm.step():
        print(f"\nStep {step}: VM halted")
        break
    tokens = vm.draft_tokens()

    # Parse tokens
    pc_marker, pc = tokens[0], tokens[1:5]
    ax_marker, ax = tokens[5], tokens[6:10]
    sp_marker, sp = tokens[10], tokens[11:15]
    bp_marker, bp = tokens[15], tokens[16:20]
    s0_marker, s0 = tokens[20], tokens[21:25]

    # Convert byte arrays to values
    pc_val = sum(b << (i*8) for i, b in enumerate(pc))
    ax_val = sum(b << (i*8) for i, b in enumerate(ax))
    sp_val = sum(b << (i*8) for i, b in enumerate(sp))
    bp_val = sum(b << (i*8) for i, b in enumerate(bp))
    s0_val = sum(b << (i*8) for i, b in enumerate(s0))

    print(f"\nStep {step}: After executing instruction {step}")
    print(f"  PC = 0x{pc_val:05X} (bytes: {pc})")
    print(f"  AX = 0x{ax_val:08X} (bytes: {ax})")
    print(f"  SP = 0x{sp_val:05X} (bytes: {sp})")
    print(f"  BP = 0x{bp_val:05X} (bytes: {bp})")
    print(f"  STACK0 = 0x{s0_val:08X} (bytes: {s0})")

    # Highlight SP byte 1 for step 2
    if step == 2:
        print(f"\n  !!! Step 2, Token 12 = SP byte 1 = {sp[1]} (0x{sp[1]:02X})")
