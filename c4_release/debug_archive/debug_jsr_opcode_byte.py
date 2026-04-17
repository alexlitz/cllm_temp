#!/usr/bin/env python3
"""Check if JSR opcode byte (3) has OP_JSR in embedding."""

import torch
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import _SetDim as BD

runner = AutoregressiveVMRunner()
model = runner.model
device = next(model.parameters()).device

# Check byte 3 (JSR opcode) embedding
test_tokens = torch.tensor([[3]], device=device)  # JSR opcode
embed = model.embed(test_tokens)

print("JSR opcode byte 3 embedding:")
print(f"  OP_JSR: {embed[0, 0, BD.OP_JSR].item():.3f}")
print(f"  OP_IMM: {embed[0, 0, BD.OP_IMM].item():.3f}")
print(f"  OP_LEA: {embed[0, 0, BD.OP_LEA].item():.3f}")

# The CODE section should have OP_JSR propagating through
# But on first step, PC marker is at position 12, not at the code byte
# L6 head 5 should relay from PC marker to AX marker

# The question is: where does OP_JSR come from at PC marker?
# Answer: It doesn't! The PC marker (token 257) doesn't have OP_JSR
# The OP_JSR needs to be relayed from somewhere else

print("\n" + "=" * 80)
print("INSIGHT:")
print("The PC MARKER token (257) doesn't have OP_JSR in embedding.")
print("L6 head 5 relays from PC marker → AX marker.")
print("But where does OP_JSR get TO the PC marker in the first place?")
print("\nL5 FFN should decode opcode and write OP_JSR to PC marker.")
print("But L5 FFN needs OPCODE_BYTE_LO/HI, which comes from L5 head 2.")
print("L5 head 2 fetches opcode from CODE section using ADDR_KEY.")
print("\nLet's check if L5 is writing OP_JSR to PC marker...")
