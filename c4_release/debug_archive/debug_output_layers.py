"""Trace OUTPUT_LO/HI through all layers for JMP."""
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

    print(f"PC marker at position {pc_marker_pos}")
    print(f"Expected final OUTPUT: 16 (JMP target)\n")
    print("=" * 60)

    # Trace through all layers
    for layer_idx in range(16):
        x = model.blocks[layer_idx](x)

        output_lo = [x[0, pc_marker_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
        output_hi = [x[0, pc_marker_pos, BD.OUTPUT_HI + k].item() for k in range(16)]
        output_val = decode_nibbles(output_lo, output_hi)

        # Also check EMBED for comparison
        embed_lo = [x[0, pc_marker_pos, BD.EMBED_LO + k].item() for k in range(16)]
        embed_hi = [x[0, pc_marker_pos, BD.EMBED_HI + k].item() for k in range(16)]
        embed_val = decode_nibbles(embed_lo, embed_hi)

        # Check flags
        mark_pc = x[0, pc_marker_pos, BD.MARK_PC].item()
        is_byte = x[0, pc_marker_pos, BD.IS_BYTE].item()
        h1_pc = x[0, pc_marker_pos, BD.H1 + 0].item() if hasattr(BD, 'H1') else 0
        has_se = x[0, pc_marker_pos, BD.HAS_SE].item()

        print(f"Layer {layer_idx:2d}: OUTPUT={output_val:3d} | EMBED={embed_val:3d} | "
              f"MARK_PC={mark_pc:.1f} IS_BYTE={is_byte:.1f} HAS_SE={has_se:.1f}")

        # Show details for layers where OUTPUT changes significantly
        if layer_idx > 0:
            if abs(output_val - prev_output) > 1:
                print(f"         ^^^ OUTPUT CHANGED: {prev_output} → {output_val}")
                print(f"         OUTPUT_LO: {[f'{v:.2f}' if abs(v) > 0.01 else '0' for v in output_lo]}")
                print(f"         OUTPUT_HI: {[f'{v:.2f}' if abs(v) > 0.01 else '0' for v in output_hi]}")

        prev_output = output_val

    # Final head prediction
    logits = model.head(x)
    predicted = logits[0, -1, :].argmax().item()

    print("=" * 60)
    print(f"\nFinal prediction (from logits): {predicted}")
    print(f"Expected (draft token):          {draft_tokens[1]}")
    print(f"\nMatch: {'✓ PASS' if predicted == draft_tokens[1] else '✗ FAIL'}")
