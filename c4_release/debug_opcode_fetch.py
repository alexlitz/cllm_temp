"""Debug opcode fetching and decoding."""
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

print(f"Bytecode: IMM opcode = {Opcode.IMM} = {hex(Opcode.IMM)}")
print(f"Code section: {context[1:6]}")  # opcode + 4 imm bytes
print(f"Context: {context}")

ctx_tensor = torch.tensor([context], dtype=torch.long)
with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._add_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    # Run through Layer 5 (fetch layer)
    for i in range(6):
        x = model.blocks[i](x)

    ax_marker_pos = len(context) - 1

    print(f"\n=== AFTER LAYER 5 (FETCH) AT AX MARKER ===")
    print(f"EMBED_LO/HI (PC value):")
    for k in range(16):
        val = x[0, ax_marker_pos, BD.EMBED_LO + k].item()
        if abs(val) > 0.1:
            print(f"  EMBED_LO[{k}] = {val:.3f}")
    for k in range(16):
        val = x[0, ax_marker_pos, BD.EMBED_HI + k].item()
        if abs(val) > 0.1:
            print(f"  EMBED_HI[{k}] = {val:.3f}")

    print(f"\nOPCODE decode (expecting OP_IMM):")
    print(f"  OP_IMM: {x[0, ax_marker_pos, BD.OP_IMM].item():.3f}")
    print(f"  OP_NOP: {x[0, ax_marker_pos, BD.OP_NOP].item():.3f}")
    print(f"  OP_LEA: {x[0, ax_marker_pos, BD.OP_LEA].item():.3f}")

    print(f"\nFETCH_LO/HI (immediate value, expecting 42):")
    for k in range(16):
        val = x[0, ax_marker_pos, BD.FETCH_LO + k].item()
        if abs(val) > 0.1:
            print(f"  FETCH_LO[{k}] = {val:.3f}")
    for k in range(16):
        val = x[0, ax_marker_pos, BD.FETCH_HI + k].item()
        if abs(val) > 0.1:
            print(f"  FETCH_HI[{k}] = {val:.3f}")
