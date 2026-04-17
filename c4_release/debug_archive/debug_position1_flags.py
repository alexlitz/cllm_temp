"""Check what flags are available at position 1 (opcode byte)."""
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

bytecode = [Opcode.JMP | (16 << 8)]
context = build_context(bytecode)

ctx_tensor = torch.tensor([context], dtype=torch.long)
with torch.no_grad():
    x = model.embed(ctx_tensor)

    pos1 = 1  # Opcode byte position
    print(f"Position {pos1} (opcode byte = {context[pos1]}):")
    print(f"  IS_BYTE: {x[0, pos1, BD.IS_BYTE].item():.3f}")
    print(f"  MARK_CS: {x[0, pos1, BD.MARK_CS].item():.3f}")
    print(f"  EMBED_LO: {[x[0, pos1, BD.EMBED_LO + k].item() for k in range(16)]}")
    print(f"  EMBED_HI: {[x[0, pos1, BD.EMBED_HI + k].item() for k in range(16)]}")
    print(f"  CLEAN_EMBED_LO: {[x[0, pos1, BD.CLEAN_EMBED_LO + k].item() for k in range(16)]}")
    print(f"  CLEAN_EMBED_HI: {[x[0, pos1, BD.CLEAN_EMBED_HI + k].item() for k in range(16)]}")

    pos0 = 0  # CODE_START
    print(f"\nPosition {pos0} (CODE_START):")
    print(f"  IS_BYTE: {x[0, pos0, BD.IS_BYTE].item():.3f}")
    print(f"  MARK_CS: {x[0, pos0, BD.MARK_CS].item():.3f}")
    print(f"  IS_MARK: {x[0, pos0, BD.IS_MARK].item():.3f}")
