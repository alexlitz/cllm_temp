"""Debug HAS_SE flag to understand PC increment."""
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

# Setup
model = AutoregressiveVM()
set_vm_weights(model)
model.compact(block_size=32)
model.compact_moe()
model.eval()

bytecode = [Opcode.NOP]
context = build_context(bytecode)

print(f"=== CONTEXT (length={len(context)}) ===")
print(f"Tokens: {context}")

# Forward pass through model
ctx_tensor = torch.tensor([context], dtype=torch.long)
with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._add_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    # After L2 (HAS_SE should be set)
    for i in range(3):
        x = model.blocks[i](x)

    print(f"\n=== AFTER L2 (position {len(context)-1}) ===")
    last_pos = len(context) - 1
    print(f"HAS_SE: {x[0, last_pos, BD.HAS_SE].item():.3f}")
    print(f"MARK_PC: {x[0, last_pos, BD.MARK_PC].item():.3f}")

    # After L3 attention (should have EMBED_LO/HI populated)
    x = x + model.blocks[3].attn(x)

    print(f"\n=== AFTER L3 ATTENTION ===")
    embed_lo = [f'{x[0, last_pos, BD.EMBED_LO + k].item():.2f}' for k in range(16)]
    print(f"EMBED_LO: {embed_lo}")
    embed_hi = [f'{x[0, last_pos, BD.EMBED_HI + k].item():.2f}' for k in range(16)]
    print(f"EMBED_HI: {embed_hi}")

    # After L3 FFN (should write to OUTPUT_LO/HI)
    x = x + model.blocks[3].ffn(x)

    print(f"\n=== AFTER L3 FFN ===")
    output_lo = [f'{x[0, last_pos, BD.OUTPUT_LO + k].item():.2f}' for k in range(16)]
    print(f"OUTPUT_LO: {output_lo}")
    output_hi = [f'{x[0, last_pos, BD.OUTPUT_HI + k].item():.2f}' for k in range(16)]
    print(f"OUTPUT_HI: {output_hi}")

    # Check which nibble has max value
    lo_max = x[0, last_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16].argmax().item()
    hi_max = x[0, last_pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16].argmax().item()
    pc_value = lo_max | (hi_max << 4)
    print(f"\nDecoded PC from OUTPUT: {pc_value} (expected: 0 for first step)")
