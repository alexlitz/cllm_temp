"""Debug Layer 10 block as a whole."""
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
    # Pre-run to initialize model state
    _ = model.forward(ctx_tensor)

    # Now recompute from scratch to get activations
    x = model.embed(ctx_tensor)

    pc_marker_pos = len(current_context) - 1

    # Run through Layer 9
    for layer_idx in range(10):
        x = model.blocks[layer_idx](x)

    print(f"=== AFTER LAYER 9 (before L10) ===")
    output_lo = [x[0, pc_marker_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
    output_hi = [x[0, pc_marker_pos, BD.OUTPUT_HI + k].item() for k in range(16)]
    output_val = decode_nibbles(output_lo, output_hi)
    print(f"OUTPUT = {output_val}")
    print(f"OUTPUT_LO max values: {max(output_lo):.2f}, argmax: {output_lo.index(max(output_lo))}")
    print(f"OUTPUT_HI max values: {max(output_hi):.2f}, argmax: {output_hi.index(max(output_hi))}")
    print(f"MARK_PC = {x[0, pc_marker_pos, BD.MARK_PC].item():.3f}")
    print(f"MARK_AX = {x[0, pc_marker_pos, BD.MARK_AX].item():.3f}")
    print(f"H1[0] (PC) = {x[0, pc_marker_pos, BD.H1 + 0].item():.3f}")
    print(f"H1[1] (AX) = {x[0, pc_marker_pos, BD.H1 + 1].item():.3f}")
    print(f"HAS_SE = {x[0, pc_marker_pos, BD.HAS_SE].item():.3f}")
    print(f"IS_BYTE = {x[0, pc_marker_pos, BD.IS_BYTE].item():.3f}")
    print(f"AX_CARRY_LO[0] = {x[0, pc_marker_pos, BD.AX_CARRY_LO + 0].item():.3f}")
    print(f"AX_CARRY_HI[0] = {x[0, pc_marker_pos, BD.AX_CARRY_HI + 0].item():.3f}")

    # Check CLEAN_EMBED values at positions that might be attended to
    print(f"\nCLEAN_EMBED_LO[0] at positions 0-9:")
    for pos in range(10):
        val = x[0, pos, BD.CLEAN_EMBED_LO + 0].item()
        print(f"  Position {pos}: {val:.2f}")

    # Run Layer 10 as a full block
    x = model.blocks[10](x)

    print(f"\n=== AFTER LAYER 10 (full block with residuals) ===")
    output_lo = [x[0, pc_marker_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
    output_hi = [x[0, pc_marker_pos, BD.OUTPUT_HI + k].item() for k in range(16)]
    print(f"OUTPUT_LO: {[f'{v:.2f}' for v in output_lo]}")
    print(f"OUTPUT_HI: {[f'{v:.2f}' for v in output_hi]}")
    print(f"OUTPUT_LO max values: {max(output_lo):.2f}, argmax: {output_lo.index(max(output_lo))}")
    print(f"OUTPUT_HI max values: {max(output_hi):.2f}, argmax: {output_hi.index(max(output_hi))}")
    output_val = decode_nibbles(output_lo, output_hi)
    print(f"OUTPUT = {output_val}")
    print(f"\nExpected: 16")
