"""Debug IMM byte 1 - check nibble copy logic."""
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

print("IMM 42 - Byte 1 Debug (Nibble Copy Logic)")
print("=" * 70)
print()

bytecode = [Opcode.IMM | (42 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

# Build context up to AX byte 0
ctx = context + draft_tokens[:6]  # REG_PC + 4 bytes + REG_AX
print(f"Context length: {len(ctx)}")
print(f"Last 3 tokens: {ctx[-3:]}")
print()

# Capture L3 and L15 outputs
l3_out = None
l15_out = None

def l3_hook(module, input, output):
    global l3_out
    l3_out = output.detach().clone()

def l15_hook(module, input, output):
    global l15_out
    l15_out = output.detach().clone()

model.blocks[3].register_forward_hook(l3_hook)
model.blocks[15].register_forward_hook(l15_hook)

ctx_tensor = torch.tensor([ctx], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)
    pred = torch.argmax(logits[0, -1, :]).item()

pos = len(ctx) - 1
print(f"Position {pos} (token {ctx[pos]} = REG_AX):")
print()

print("After L3 (nibble copy layer):")
if l3_out is not None:
    is_byte = l3_out[0, pos, BD.IS_BYTE].item()
    h1_ax = l3_out[0, pos, BD.H1 + 1].item()  # AX_I = 1
    h1_pc = l3_out[0, pos, BD.H1 + 0].item()  # PC_I = 0
    embed_lo = l3_out[0, pos, BD.EMBED_LO:BD.EMBED_LO+16]
    embed_hi = l3_out[0, pos, BD.EMBED_HI:BD.EMBED_HI+16]
    output_lo = l3_out[0, pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi = l3_out[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]

    print(f"  IS_BYTE: {is_byte:.3f}")
    print(f"  H1[AX]: {h1_ax:.3f}")
    print(f"  H1[PC]: {h1_pc:.3f}")
    print(f"  EMBED: lo_max={torch.max(embed_lo).item():.3f}, hi_max={torch.max(embed_hi).item():.3f}")
    print(f"  OUTPUT: lo_max={torch.max(output_lo).item():.3f}, hi_max={torch.max(output_hi).item():.3f}")
    print()

print(f"Prediction: {pred} (expected {draft_tokens[6]})")
print()

# Now add AX byte 0 and check next position
ctx2 = ctx + [pred]
pos2 = len(ctx2) - 1

l3_out = None
l15_out = None

ctx2_tensor = torch.tensor([ctx2], dtype=torch.long)
with torch.no_grad():
    logits2 = model.forward(ctx2_tensor)
    pred2 = torch.argmax(logits2[0, -1, :]).item()

print(f"Position {pos2} (token {ctx2[pos2]} = {pred}):")
print()

print("After L3 (nibble copy layer):")
if l3_out is not None:
    is_byte = l3_out[0, pos2, BD.IS_BYTE].item()
    h1_ax = l3_out[0, pos2, BD.H1 + 1].item()  # AX_I = 1
    byte_idx_0 = l3_out[0, pos2, BD.BYTE_INDEX_0].item()
    byte_idx_1 = l3_out[0, pos2, BD.BYTE_INDEX_1].item()
    embed_lo = l3_out[0, pos2, BD.EMBED_LO:BD.EMBED_LO+16]
    embed_hi = l3_out[0, pos2, BD.EMBED_HI:BD.EMBED_HI+16]
    output_lo = l3_out[0, pos2, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi = l3_out[0, pos2, BD.OUTPUT_HI:BD.OUTPUT_HI+16]

    print(f"  IS_BYTE: {is_byte:.3f}")
    print(f"  H1[AX]: {h1_ax:.3f}")
    print(f"  BYTE_INDEX_0: {byte_idx_0:.3f}")
    print(f"  BYTE_INDEX_1: {byte_idx_1:.3f}")
    print(f"  EMBED: lo={torch.argmax(embed_lo).item()}, hi={torch.argmax(embed_hi).item()}")
    print(f"  OUTPUT: lo={torch.argmax(output_lo).item()}, hi={torch.argmax(output_hi).item()}")
    embed_val = torch.argmax(embed_lo).item() + 16 * torch.argmax(embed_hi).item()
    output_val = torch.argmax(output_lo).item() + 16 * torch.argmax(output_hi).item()
    print(f"  EMBED value: {embed_val}")
    print(f"  OUTPUT value: {output_val}")
    print()

print(f"Prediction: {pred2} (expected {draft_tokens[7]})")
print()

print("Analysis:")
print("  If IS_BYTE=1 and OUTPUT≠EMBED, nibble copy is not working")
print("  If H1[AX] is high, it might be interfering (but shouldn't - no suppression)")
