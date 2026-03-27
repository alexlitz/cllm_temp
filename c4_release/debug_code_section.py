"""Debug opcode flags in code section."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.embedding import Opcode

def build_context(bytecode):
    tokens = [Token.CODE_START]
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        tokens.append(op)
        for i in range(4):
            tokens.append((imm >> (i * 8)) & 0xFF)
    tokens.append(Token.CODE_END)
    tokens.append(Token.DATA_START)
    tokens.append(Token.DATA_END)
    return tokens

model = AutoregressiveVM()
set_vm_weights(model)
model.compact(block_size=32)
model.compact_moe()
model.eval()

bytecode = [Opcode.JMP | (16 << 8)]
context = build_context(bytecode)

print(f"Context tokens: {context}")
print(f"Position 0: CODE_START = {context[0]}")
print(f"Position 1: JMP opcode = {context[1]} (Opcode.JMP = {Opcode.JMP})")
print(f"Position 2-5: immediate bytes = {context[2:6]}")

ctx_tensor = torch.tensor([context], dtype=torch.long)
with torch.no_grad():
    x = model.embed(ctx_tensor)

    opcode_pos = 1
    print(f"\n=== AT OPCODE POSITION {opcode_pos} ==")
    print(f"Token value: {context[opcode_pos]}")
    print(f"OP_JMP: {x[0, opcode_pos, BD.OP_JMP].item():.3f}")
    print(f"OP_IMM: {x[0, opcode_pos, BD.OP_IMM].item():.3f}")
    print(f"OP_NOP: {x[0, opcode_pos, BD.OP_NOP].item():.3f}")
    print(f"OP_EXIT: {x[0, opcode_pos, BD.OP_EXIT].item():.3f}")
