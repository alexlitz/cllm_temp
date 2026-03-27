"""Check ADDR_KEY values at code section positions."""
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

    print("=== BEFORE _add_code_addr_keys ===")
    for pos in range(6):
        addr_key_lo = [x[0, pos, BD.ADDR_KEY + k].item() for k in range(16)]
        addr_key_hi = [x[0, pos, BD.ADDR_KEY + 16 + k].item() for k in range(16)]
        print(f"Position {pos} ({context[pos]}): ADDR_KEY_LO={addr_key_lo[:4]}, ADDR_KEY_HI={addr_key_hi[:4]}")

    model._add_code_addr_keys(ctx_tensor, x)

    print("\n=== AFTER _add_code_addr_keys ===")
    for pos in range(6):
        addr_key_lo = [x[0, pos, BD.ADDR_KEY + k].item() for k in range(16)]
        addr_key_hi = [x[0, pos, BD.ADDR_KEY + 16 + k].item() for k in range(16)]
        lo_max = max(range(16), key=lambda k: addr_key_lo[k]) if max(addr_key_lo) > 0 else -1
        hi_max = max(range(16), key=lambda k: addr_key_hi[k]) if max(addr_key_hi) > 0 else -1
        addr = lo_max | (hi_max << 4) if lo_max >= 0 else -1
        print(f"Position {pos} ({context[pos]}): address={addr}")
