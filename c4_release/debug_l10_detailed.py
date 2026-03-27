"""Debug Layer 10 attention and FFN separately."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM

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

def decode_nibbles(lo_vals, hi_vals):
    """Decode nibble one-hot to byte value."""
    lo_max = max(range(16), key=lambda k: lo_vals[k])
    hi_max = max(range(16), key=lambda k: hi_vals[k])
    return lo_max | (hi_max << 4)

model = AutoregressiveVM()
set_vm_weights(model)
model.compact(block_size=32)
model.compact_moe()
model.eval()

bytecode = [Opcode.JMP | (16 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

current_context = context + [draft_tokens[0]]  # Just REG_PC marker
ctx_tensor = torch.tensor([current_context], dtype=torch.long)

with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._add_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    pc_marker_pos = len(current_context) - 1

    # Run through Layer 9
    for layer_idx in range(10):
        x = model.blocks[layer_idx](x)

    print(f"=== AFTER LAYER 9 (before L10) ===")
    output_lo = [x[0, pc_marker_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
    output_hi = [x[0, pc_marker_pos, BD.OUTPUT_HI + k].item() for k in range(16)]
    output_val = decode_nibbles(output_lo, output_hi)
    print(f"OUTPUT = {output_val}")
    print(f"MARK_PC = {x[0, pc_marker_pos, BD.MARK_PC].item():.3f}")
    print(f"H1[0] (PC marker) = {x[0, pc_marker_pos, BD.H1 + 0].item():.3f}")
    print(f"H1[1] (AX marker) = {x[0, pc_marker_pos, BD.H1 + 1].item():.3f}")

    # Run Layer 10 attention only
    x_after_attn = model.blocks[10].attn(x)

    print(f"\n=== AFTER LAYER 10 ATTENTION ===")
    output_lo = [x_after_attn[0, pc_marker_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
    output_hi = [x_after_attn[0, pc_marker_pos, BD.OUTPUT_HI + k].item() for k in range(16)]
    print(f"OUTPUT_LO: {[f'{v:.2f}' if abs(v) > 0.01 else '0' for v in output_lo]}")
    print(f"OUTPUT_HI: {[f'{v:.2f}' if abs(v) > 0.01 else '0' for v in output_hi]}")
    output_val = decode_nibbles(output_lo, output_hi)
    print(f"OUTPUT = {output_val}")

    # Run Layer 10 FFN only
    x_after_ffn = model.blocks[10].ffn(x_after_attn)

    print(f"\n=== AFTER LAYER 10 FFN ===")
    output_lo = [x_after_ffn[0, pc_marker_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
    output_hi = [x_after_ffn[0, pc_marker_pos, BD.OUTPUT_HI + k].item() for k in range(16)]
    print(f"OUTPUT_LO: {[f'{v:.2f}' if abs(v) > 0.01 else '0' for v in output_lo]}")
    print(f"OUTPUT_HI: {[f'{v:.2f}' if abs(v) > 0.01 else '0' for v in output_hi]}")
    output_val = decode_nibbles(output_lo, output_hi)
    print(f"OUTPUT = {output_val}")
    print(f"\nExpected: 16")
