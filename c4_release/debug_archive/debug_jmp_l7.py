"""Debug JMP - check if L7 FFN is firing inappropriately."""
import sys
sys.path.insert(0, '/home/alexlitz/Documents/misc/c4_release/c4_release')

# Force module reload
for mod in list(sys.modules.keys()):
    if 'neural_vm' in mod:
        del sys.modules[mod]

import torch
torch.set_num_threads(1)
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM

BD = _SetDim

bytecode = [Opcode.JMP | (0x20 << 8), Opcode.EXIT]

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
draft.step()
expected_tokens = draft.draft_tokens()

model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

# Teacher forcing up to PC marker
input_tokens = context + [expected_tokens[0]]

with torch.no_grad():
    x = model.embed(torch.tensor([input_tokens], dtype=torch.long))

    # Run through Layer 6
    for i in range(7):
        x = model.blocks[i](x)

    pc_pos = len(input_tokens) - 1

    print("JMP Debug - After Layer 6 (before L7):")
    print(f"  Position: {pc_pos} (PC marker)")
    print(f"  MARK_PC: {x[0, pc_pos, BD.MARK_PC].item():.4f}")
    print(f"  MARK_AX: {x[0, pc_pos, BD.MARK_AX].item():.4f}")
    print(f"  OP_JMP: {x[0, pc_pos, BD.OP_JMP].item():.4f}")
    print(f"  OP_LEA: {x[0, pc_pos, BD.OP_LEA].item():.4f}")
    print(f"  HAS_SE: {x[0, pc_pos, BD.HAS_SE].item():.4f}")
    print()

    # The L7 FFN condition is: MARK_AX AND OP_LEA AND NOT HAS_SE
    # At PC marker, we should have MARK_PC=1, MARK_AX=0
    # So the FFN should NOT fire.

    print("L7 FFN activation check:")
    print(f"  Condition: MARK_AX AND OP_LEA AND NOT HAS_SE")
    print(f"  Should be: 0 * LEA * (1-0) = 0 (no activation)")
    print()

    # Check ALU before/after L7
    print("ALU_LO[0] before L7:", x[0, pc_pos, BD.ALU_LO + 0].item())

    x = model.blocks[7](x)

    print("ALU_LO[0] after L7:", x[0, pc_pos, BD.ALU_LO + 0].item())
