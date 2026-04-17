#\!/usr/bin/env python3
"""Check opcode fetch at step 0."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

BD = _SetDim

print("Building model...")
model = AutoregressiveVM()
set_vm_weights(model)
model = model.cuda()
model.eval()

runner = AutoregressiveVMRunner()
runner.model = model
bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]

context = runner._build_context(bytecode, b'', [])
print(f"Program: IMM 42, EXIT\n")

print("Code section in context:")
code_start = context.index(264)  # CODE_START
code_end = context.index(265)  # CODE_END
print(f"  CODE_START at position {code_start}")
print(f"  CODE_END at position {code_end}")
print(f"  Bytes between:")
for pos in range(code_start + 1, code_end):
    byte_val = context[pos]
    print(f"    pos {pos - code_start - 1}: 0x{byte_val:02x}")

print(f"\nFirst instruction (should be IMM):")
print(f"  Opcode (pos 0): 0x{context[code_start + 1]:02x}")
print(f"  Imm byte 0 (pos 1): 0x{context[code_start + 2]:02x}")
print(f"  Imm byte 1 (pos 2): 0x{context[code_start + 3]:02x}")
print(f"  Imm byte 2 (pos 3): 0x{context[code_start + 4]:02x}")
print(f"  Imm byte 3 (pos 4): 0x{context[code_start + 5]:02x}")

print(f"\n  Expected IMM (0x01) at pos 0: {context[code_start + 1] == Opcode.IMM}")
print(f"  Expected imm=0x2a at pos 1: {context[code_start + 2] == 0x2a}")

print(f"\nPC should fetch from address 2 (PC_OFFSET=2)")
print(f"  That's code byte index 2 (pos {code_start + 1 + 2})")
print(f"  Byte at that position: 0x{context[code_start + 3]:02x}")
print(f"  Should be: imm byte 1 = 0x00")

print(f"\nFor first step (HAS_SE=0), PC defaults to PC_OFFSET=2")
print(f"  Opcode fetch should look at address 2")
print(f"  That's byte 2 in code section")
print(f"  Which is position {code_start + 1 + 2} in context")
print(f"  Context[{code_start + 3}] = 0x{context[code_start + 3]:02x}")
print(f"  This is imm byte 1, NOT the opcode\!")

print(f"\nThis explains the bug:")
print(f"  Layer 5 Head 1 fetches opcode from address PC=2")
print(f"  But the opcode is at address 0 (first byte after CODE_START)")
print(f"  PC should be 0 for first instruction fetch, not 2")
