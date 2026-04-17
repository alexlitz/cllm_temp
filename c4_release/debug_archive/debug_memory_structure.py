"""Debug memory structure and PC+1 calculation."""
import sys
sys.path.insert(0, '/home/alexlitz/Documents/misc/c4_release/c4_release')

for mod in list(sys.modules.keys()):
    if 'neural_vm' in mod:
        del sys.modules[mod]

import torch
torch.set_num_threads(1)
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM

BD = _SetDim

bytecode = [Opcode.LEA | (8 << 8), Opcode.EXIT]

def build_context(bc):
    tokens = [Token.CODE_START]
    for instr in bc:
        op = instr & 0xFF
        imm = instr >> 8
        tokens.append(op)
        for i in range(4):
            tokens.append((imm >> (i * 8)) & 0xFF)
    tokens.append(Token.CODE_END)
    tokens.append(Token.DATA_START)
    tokens.append(Token.DATA_END)
    return tokens

context = build_context(bytecode)
draft = DraftVM(bytecode)

print("Context tokens:")
for i, tok in enumerate(context):
    print(f"  Position {i:2d}: token {tok:3d}")
print()

print(f"First instruction (LEA 8):")
print(f"  Instruction index: 0")
print(f"  PC = idx_to_pc(0) = 0 * 8 + 2 = 2")
print(f"  PC+1 = 3")
print(f"  Token at position 1 (PC=2 byte address 0): {context[1]} (opcode)")
print(f"  Token at position 2 (byte address 1): {context[2]} (imm_b0)")
print(f"  Token at position 3 (byte address 2): {context[3]} (imm_b1)")
print(f"  Token at position 4 (byte address 3): {context[4]} (imm_b2)")
print()

# Check PC marker position after draft step
draft.step()
expected_tokens = draft.draft_tokens()

model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

input_tokens = context + expected_tokens[0:6]
ax_marker_pos = len(input_tokens) - 1
pc_marker_pos = len(context)

print(f"PC marker position: {pc_marker_pos}")
print(f"AX marker position: {ax_marker_pos}")
print()

with torch.no_grad():
    x = model.embed(torch.tensor([input_tokens], dtype=torch.long))

    # Run through Layer 4
    for i in range(5):
        x = model.blocks[i](x)

    print("After Layer 4, AX marker TEMP (PC+1 address):")
    for k in range(16):
        val = x[0, ax_marker_pos, BD.TEMP + k].item()
        if abs(val) > 0.01:
            print(f"  TEMP[{k}] = {val:.4f}")
    print()
    for k in range(16):
        val = x[0, ax_marker_pos, BD.TEMP + 16 + k].item()
        if abs(val) > 0.01:
            print(f"  TEMP[{k+16}] = {val:.4f}")
