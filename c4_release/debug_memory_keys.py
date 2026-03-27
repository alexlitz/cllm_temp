"""Check what memory keys are assigned to context tokens."""
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

bytecode = [Opcode.IMM | (42 << 8)]
context = build_context(bytecode)

print("Context:")
for i, tok in enumerate(context):
    if tok == Token.CODE_START:
        print(f"  {i}: CODE_START")
    elif tok == Token.CODE_END:
        print(f"  {i}: CODE_END")
    elif tok == Token.DATA_START:
        print(f"  {i}: DATA_START")
    elif tok == Token.DATA_END:
        print(f"  {i}: DATA_END")
    else:
        print(f"  {i}: {tok}")

ctx_tensor = torch.tensor([context], dtype=torch.long)

with torch.no_grad():
    x = model.embed(ctx_tensor)

    print("\nBefore memory injection:")
    for i in range(len(context)):
        addr_lo = x[0, i, BD.ADDR_B0_LO:BD.ADDR_B0_LO+16]
        if addr_lo.max() > 0.5:
            print(f"  Position {i} (token {context[i]}): ADDR_LO={addr_lo.argmax().item()}")

    model._inject_code_addr_keys(ctx_tensor, x)

    print("\nAfter code address injection:")
    for i in range(len(context)):
        # Check ADDR_B0_LO/HI (memory address nibbles)
        addr_lo = x[0, i, BD.ADDR_B0_LO:BD.ADDR_B0_LO+16]
        addr_hi = x[0, i, BD.ADDR_B0_HI:BD.ADDR_B0_HI+16]

        if addr_lo.max() > 0.5:
            lo = addr_lo.argmax().item()
            hi = addr_hi.argmax().item()
            addr = lo | (hi << 4)
            print(f"  Position {i} (token {context[i]}): Memory address={addr}")
