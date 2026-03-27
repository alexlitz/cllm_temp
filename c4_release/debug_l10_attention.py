"""Debug Layer 10 attention at AX marker."""
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

    # Run through layers up to L10
    for i in range(10):
        x = model.blocks[i](x)

    ax_marker_pos = len(context) - 1

    print(f"=== AT AX MARKER (pos {ax_marker_pos}) BEFORE L10 HEAD 1 ===")
    print(f"IS_BYTE: {x[0, ax_marker_pos, BD.IS_BYTE].item():.3f}")
    print(f"H1[AX]: {x[0, ax_marker_pos, BD.H1 + 1].item():.3f}")
    print(f"MARK_AX: {x[0, ax_marker_pos, BD.MARK_AX].item():.3f}")

    # Check OUTPUT before
    lo_vals = [x[0, ax_marker_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
    lo_max = max(range(16), key=lambda k: lo_vals[k])
    print(f"OUTPUT_LO max at [{lo_max}] = {lo_vals[lo_max]:.3f}")

    # Check CLEAN_EMBED (what the head will copy)
    print(f"\nCLEAN_EMBED at AX marker:")
    for k in range(16):
        val = x[0, ax_marker_pos, BD.CLEAN_EMBED_LO + k].item()
        if abs(val) > 0.01:
            print(f"  CLEAN_EMBED_LO[{k}] = {val:.3f}")

    # Run Layer 10 attention only
    x_after_attn = x + model.blocks[10].attn(x)

    print(f"\n=== AFTER L10 ATTENTION ===")
    lo_vals_after = [x_after_attn[0, ax_marker_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
    lo_max_after = max(range(16), key=lambda k: lo_vals_after[k])
    print(f"OUTPUT_LO max at [{lo_max_after}] = {lo_vals_after[lo_max_after]:.3f}")
    print(f"OUTPUT_LO[10] changed: {lo_vals[10]:.3f} -> {lo_vals_after[10]:.3f}")
