"""Debug ADDR_KEY setup for code bytes."""
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

print(f"Context: {context}")
print(f"Opcode at pos 1: {context[1]} (IMM)")
print(f"Imm byte 0 at pos 2: {context[2]} (42)")

ctx_tensor = torch.tensor([context], dtype=torch.long)
x = model.embed(ctx_tensor)
model._add_code_addr_keys(ctx_tensor, x)

print(f"\n=== ADDR_KEY after _add_code_addr_keys ===")
for pos in range(1, 6):  # Check code byte positions
    print(f"\nPosition {pos} (token {context[pos]}):")
    # Check ADDR_KEY (48 dims: 16 lo + 16 mid + 16 hi for 12-bit address)
    for k in range(16):
        if x[0, pos, BD.ADDR_KEY + k].item() > 0.1:
            print(f"  ADDR_KEY[{k}] (lo) = {x[0, pos, BD.ADDR_KEY + k].item():.3f}")
    for k in range(16):
        if x[0, pos, BD.ADDR_KEY + 16 + k].item() > 0.1:
            print(f"  ADDR_KEY[{16+k}] (mid) = {x[0, pos, BD.ADDR_KEY + 16 + k].item():.3f}")
    for k in range(16):
        if x[0, pos, BD.ADDR_KEY + 32 + k].item() > 0.1:
            print(f"  ADDR_KEY[{32+k}] (hi) = {x[0, pos, BD.ADDR_KEY + 32 + k].item():.3f}")
