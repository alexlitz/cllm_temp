"""Debug markers and byte index for token 7 (PC_b1) in IMM 42."""
import sys
for mod in list(sys.modules.keys()):
    if 'neural_vm' in mod:
        del sys.modules[mod]

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

model = AutoregressiveVM()
set_vm_weights(model)
model.compact(block_size=32)
model.compact_moe()
model.eval()

print("Marker Analysis for IMM 42 Token 7 (PC_b1)")
print("=" * 70)
print()

bytecode = [Opcode.IMM | (42 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

# Build context up to token 7
ctx_7 = context + draft_tokens[:7]

# Capture embedding and L6 FFN input
embed_out = None
l6_ffn_in = None

def embed_hook(module, input, output):
    global embed_out
    embed_out = output.detach().clone()

def l6_hook(module, input, output):
    global l6_ffn_in
    if isinstance(input, tuple):
        l6_ffn_in = input[0].detach().clone()
    else:
        l6_ffn_in = input.detach().clone()

model.embed.register_forward_hook(embed_hook)
model.blocks[6].ffn.register_forward_hook(l6_hook)

ctx_tensor = torch.tensor([ctx_7], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

pos = len(ctx_7) - 1
print(f"Position {pos} analysis (last token = {ctx_7[-1]}):")
print()

# Check embedding layer
if embed_out is not None:
    mark_ax = embed_out[0, pos, BD.MARK_AX].item()
    mark_pc = embed_out[0, pos, BD.MARK_PC].item()
    mark_sp = embed_out[0, pos, BD.MARK_SP].item()
    is_byte = embed_out[0, pos, BD.IS_BYTE].item()
    byte_idx_0 = embed_out[0, pos, BD.BYTE_INDEX_0].item()
    byte_idx_1 = embed_out[0, pos, BD.BYTE_INDEX_1].item()
    byte_idx_2 = embed_out[0, pos, BD.BYTE_INDEX_2].item()

    print("Embedding layer:")
    print(f"  MARK_AX:  {mark_ax:.3f}")
    print(f"  MARK_PC:  {mark_pc:.3f}")
    print(f"  MARK_SP:  {mark_sp:.3f}")
    print(f"  IS_BYTE:  {is_byte:.3f}")
    print(f"  BYTE_IDX[0]: {byte_idx_0:.3f}")
    print(f"  BYTE_IDX[1]: {byte_idx_1:.3f}")
    print(f"  BYTE_IDX[2]: {byte_idx_2:.3f}")
    print()

# Check L6 FFN input
if l6_ffn_in is not None:
    mark_ax = l6_ffn_in[0, pos, BD.MARK_AX].item()
    mark_pc = l6_ffn_in[0, pos, BD.MARK_PC].item()
    is_byte = l6_ffn_in[0, pos, BD.IS_BYTE].item()
    byte_idx_0 = l6_ffn_in[0, pos, BD.BYTE_INDEX_0].item()
    byte_idx_1 = l6_ffn_in[0, pos, BD.BYTE_INDEX_1].item()
    byte_idx_2 = l6_ffn_in[0, pos, BD.BYTE_INDEX_2].item()

    print("L6 FFN input:")
    print(f"  MARK_AX:  {mark_ax:.3f}")
    print(f"  MARK_PC:  {mark_pc:.3f}")
    print(f"  IS_BYTE:  {is_byte:.3f}")
    print(f"  BYTE_IDX[0]: {byte_idx_0:.3f}")
    print(f"  BYTE_IDX[1]: {byte_idx_1:.3f}")
    print(f"  BYTE_IDX[2]: {byte_idx_2:.3f}")
    print()

    # Check PC value
    pc_lo = l6_ffn_in[0, pos, BD.EMBED_LO:BD.EMBED_LO+16]
    pc_hi = l6_ffn_in[0, pos, BD.EMBED_HI:BD.EMBED_HI+16]
    pc_lo_val = torch.argmax(pc_lo).item()
    pc_hi_val = torch.argmax(pc_hi).item()
    pc_val = pc_lo_val + (pc_hi_val << 4)

    print(f"  PC value (from EMBED): {pc_val} (lo={pc_lo_val}, hi={pc_hi_val})")
    print()

print("Expected for token 7 (PC_b1):")
print("  - Should be outputting PC byte 1 (value 0)")
print("  - BYTE_IDX should encode 1 (byte index within register)")
print("  - MARK_PC should be 1.0")
print()

print("Analysis:")
if l6_ffn_in is not None:
    if byte_idx_0 < 0.5:  # BYTE_IDX[0] should be 1 for byte 1
        print("  ⚠️  BYTE_IDX[0] is 0, but should be 1 for PC byte 1")
        print("  This explains why PC[0] is being emitted instead of PC[1]")
    else:
        print("  ✓ BYTE_IDX[0] is correct")
