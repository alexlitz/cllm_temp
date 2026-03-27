"""Check H1 flags in code section."""
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

bytecode = [Opcode.IMM | (42 << 8)]
context = build_context(bytecode)
context.extend([Token.REG_PC, 10])

print(f"Context: {context}")
print()

ctx_tensor = torch.tensor([context], dtype=torch.long)
with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._add_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    # Run through Layer 1 (where H1 is set)
    for i in range(2):
        x = model.blocks[i](x)

    print("=== H1 FLAGS AFTER LAYER 1 ===")
    for pos in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        token = context[pos]
        h1_ax = x[0, pos, BD.H1 + 1].item()
        is_byte = x[0, pos, BD.IS_BYTE].item()
        print(f"Pos {pos:2d} token={token:3d}: IS_BYTE={is_byte:.3f}, H1[AX]={h1_ax:.3f}")
