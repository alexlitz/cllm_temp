"""Debug IMM byte 1 - detailed check at position 16."""
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

print("IMM Byte 1 (Position 16) - Detailed Analysis")
print("=" * 70)
print()

bytecode = [Opcode.IMM | (42 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

# Build context up to and including AX byte 0
ctx = context + draft_tokens[:7]  # REG_PC + 4 bytes + REG_AX + AX_b0
pos = len(ctx) - 1

print(f"Context: {ctx[-3:]}")
print(f"Position {pos}: token {ctx[pos]} (AX byte 0)")
print()

# Capture L14, L15 FFN input, L15 output
l14_out = None
l15_ffn_in = None
l15_out = None

def l14_hook(module, input, output):
    global l14_out
    l14_out = output.detach().clone()

def l15_ffn_hook(module, input, output):
    global l15_ffn_in
    if isinstance(input, tuple):
        l15_ffn_in = input[0].detach().clone()
    else:
        l15_ffn_in = input.detach().clone()

def l15_hook(module, input, output):
    global l15_out
    l15_out = output.detach().clone()

model.blocks[14].register_forward_hook(l14_hook)
model.blocks[15].ffn.register_forward_hook(l15_ffn_hook)
model.blocks[15].register_forward_hook(l15_hook)

ctx_tensor = torch.tensor([ctx], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)
    pred = torch.argmax(logits[0, -1, :]).item()

print(f"Predicting token at position {pos+1} (AX byte 1):")
print(f"  Expected: {draft_tokens[7]} (= 0)")
print(f"  Predicted: {pred}")
print()

# Check what's in EMBED at position 16 (AX byte 1)
# But we haven't added it to context yet, so check position 15
if l14_out is not None:
    print(f"State at position {pos} (AX byte 0 = 42):")
    is_byte = l14_out[0, pos, BD.IS_BYTE].item()
    byte_idx_0 = l14_out[0, pos, BD.BYTE_INDEX_0].item()
    byte_idx_1 = l14_out[0, pos, BD.BYTE_INDEX_1].item()
    h1_ax = l14_out[0, pos, BD.H1 + 1].item()
    embed_lo = l14_out[0, pos, BD.EMBED_LO:BD.EMBED_LO+16]
    embed_hi = l14_out[0, pos, BD.EMBED_HI:BD.EMBED_HI+16]
    output_lo = l14_out[0, pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi = l14_out[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]

    print(f"  IS_BYTE: {is_byte:.3f}")
    print(f"  BYTE_INDEX_0: {byte_idx_0:.3f}")
    print(f"  BYTE_INDEX_1: {byte_idx_1:.3f}")
    print(f"  H1[AX]: {h1_ax:.3f}")
    print(f"  EMBED value: {torch.argmax(embed_lo).item()} + 16*{torch.argmax(embed_hi).item()} = {torch.argmax(embed_lo).item() + 16*torch.argmax(embed_hi).item()}")
    print(f"  OUTPUT (L14): {torch.argmax(output_lo).item()} + 16*{torch.argmax(output_hi).item()} = {torch.argmax(output_lo).item() + 16*torch.argmax(output_hi).item()}")
    print()

if l15_out is not None:
    output_lo = l15_out[0, pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi = l15_out[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    print(f"  OUTPUT (L15): {torch.argmax(output_lo).item()} + 16*{torch.argmax(embed_hi).item()} = {torch.argmax(output_lo).item() + 16*torch.argmax(output_hi).item()}")
    print()

print("Analysis:")
print("  The prediction at position 15 (for position 16 = AX byte 1) is wrong")
print("  Position 16 should predict 0, but predicts 42")
print("  This suggests EMBED at position 16 contains 42 (stale value)")
print("  Or OUTPUT is not being updated correctly")
