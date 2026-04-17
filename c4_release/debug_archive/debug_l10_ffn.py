"""Debug Layer 10 FFN at AX marker."""
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

    # Run through layers up to L10 (including attn)
    for i in range(10):
        x = model.blocks[i](x)
    x = x + model.blocks[10].attn(x)

    ax_marker_pos = len(context) - 1

    print(f"=== AT AX MARKER (pos {ax_marker_pos}) BEFORE L10 FFN ===")
    lo_vals = [x[0, ax_marker_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
    hi_vals = [x[0, ax_marker_pos, BD.OUTPUT_HI + k].item() for k in range(16)]
    lo_max = max(range(16), key=lambda k: lo_vals[k])
    hi_max = max(range(16), key=lambda k: hi_vals[k])
    val = lo_max | (hi_max << 4)
    print(f"OUTPUT encodes: {val}")
    print(f"  OUTPUT_LO[{lo_max}] = {lo_vals[lo_max]:.3f}")
    print(f"  OUTPUT_HI[{hi_max}] = {hi_vals[hi_max]:.3f}")

    # Check opcode flags
    print(f"\nOpcode flags:")
    print(f"  MARK_AX: {x[0, ax_marker_pos, BD.MARK_AX].item():.3f}")
    print(f"  OP_IMM: {x[0, ax_marker_pos, BD.OP_IMM].item():.3f}")
    print(f"  OP_NOP: {x[0, ax_marker_pos, BD.OP_NOP].item():.3f}")

    # Run L10 FFN
    x_after = x + model.blocks[10].ffn(x)

    print(f"\n=== AFTER L10 FFN ===")
    lo_vals_after = [x_after[0, ax_marker_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
    hi_vals_after = [x_after[0, ax_marker_pos, BD.OUTPUT_HI + k].item() for k in range(16)]
    lo_max_after = max(range(16), key=lambda k: lo_vals_after[k])
    hi_max_after = max(range(16), key=lambda k: hi_vals_after[k])
    val_after = lo_max_after | (hi_max_after << 4)
    print(f"OUTPUT encodes: {val_after}")
    print(f"  OUTPUT_LO[{lo_max_after}] = {lo_vals_after[lo_max_after]:.3f}")
    print(f"  OUTPUT_HI[{hi_max_after}] = {hi_vals_after[hi_max_after]:.3f}")

    # Show which dims changed significantly
    print(f"\nSignificant changes:")
    for k in range(16):
        if abs(lo_vals_after[k] - lo_vals[k]) > 0.5:
            print(f"  OUTPUT_LO[{k}]: {lo_vals[k]:.3f} -> {lo_vals_after[k]:.3f}")
    for k in range(16):
        if abs(hi_vals_after[k] - hi_vals[k]) > 0.5:
            print(f"  OUTPUT_HI[{k}]: {hi_vals[k]:.3f} -> {hi_vals_after[k]:.3f}")
