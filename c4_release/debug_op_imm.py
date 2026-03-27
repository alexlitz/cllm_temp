"""Debug why OP_IMM isn't set at AX marker."""
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
context.extend([Token.REG_PC, 10, 0, 0, 0, Token.REG_AX])

ctx_tensor = torch.tensor([context], dtype=torch.long)
with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._add_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    ax_marker_pos = len(context) - 1

    print("=== TRACKING OP_IMM AT AX MARKER ===")
    for i in range(16):
        x = model.blocks[i](x)
        op_imm = x[0, ax_marker_pos, BD.OP_IMM].item()
        if abs(op_imm) > 0.1:
            print(f"Layer {i}: OP_IMM = {op_imm:.3f}")
    
    print(f"\nFinal OP_IMM: {x[0, ax_marker_pos, BD.OP_IMM].item():.3f}")
    print(f"Expected: ~5.0 (opcode detection strength)")
