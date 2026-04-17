"""Test ADDR_KEY encoding with new addressing."""
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

# NOP + JMP 16
bytecode = [Opcode.NOP, Opcode.JMP | (16 << 8)]
context = build_context(bytecode)

print("=== CODE LAYOUT ===")
print(f"Context: {context}")
print(f"\nExpected addresses:")
print(f"  Position 1 (NOP opcode, instr 0 byte 0): addr=0")
print(f"  Position 2-5 (NOP imm, instr 0 bytes 1-4): addr=1-4")
print(f"  Position 6 (JMP opcode, instr 1 byte 0): addr=8")
print(f"  Position 7 (JMP imm[0], instr 1 byte 1): addr=9")

ctx_tensor = torch.tensor([context], dtype=torch.long)

with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._add_code_addr_keys(ctx_tensor, x)

    print("\n=== ACTUAL ADDR_KEY ===")
    for pos in [1, 2, 5, 6, 7]:
        addr_lo = x[0, pos, BD.ADDR_KEY:BD.ADDR_KEY+16].argmax().item()
        addr_mid = x[0, pos, BD.ADDR_KEY+16:BD.ADDR_KEY+32].argmax().item()
        addr = addr_lo | (addr_mid << 4)
        print(f"  Position {pos}: ADDR_KEY={addr}, token={context[pos]}")
