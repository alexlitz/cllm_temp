"""Debug LEA 8 data flow to understand byte propagation."""
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

bytecode = [Opcode.LEA | (8 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

print("LEA 8 Data Flow Analysis")
print("=" * 70)
print()
print(f"Expected AX: 0x00010008 = [8, 0, 1, 0]")
print(f"Draft tokens 6-9 (AX bytes): {draft_tokens[6:10]}")
print()

# Check at AX marker (position 14, after token 5 = REG_AX)
ctx_marker = context + draft_tokens[:6]
pos_marker = len(ctx_marker) - 1

# Check at AX byte 0 (position 15, after token 6 = 8)
ctx_byte0 = context + draft_tokens[:7]
pos_byte0 = len(ctx_byte0) - 1

# Capture outputs
outputs = {}

def make_hook(name):
    def hook(module, input, output):
        outputs[name] = output.detach().clone()
    return hook

for i, name in [(6, 'l6'), (8, 'l8'), (9, 'l9'), (10, 'l10'), (15, 'l15')]:
    model.blocks[i].register_forward_hook(make_hook(name))

# Run at marker position
ctx_tensor = torch.tensor([ctx_marker], dtype=torch.long)
with torch.no_grad():
    model.forward(ctx_tensor)

print("At AX MARKER (position 14, predicting AX_b0):")
print("-" * 70)
for layer in ['l6', 'l8', 'l9', 'l10', 'l15']:
    if layer in outputs:
        out = outputs[layer][0, pos_marker, :]
        output_lo = out[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
        output_hi = out[BD.OUTPUT_HI:BD.OUTPUT_HI+16]
        lo_val = torch.argmax(output_lo).item()
        hi_val = torch.argmax(output_hi).item()
        val = lo_val + 16 * hi_val
        print(f"  {layer:4s}: OUTPUT = {val:3d} (lo={lo_val}, hi={hi_val})")

print()
print("At AX BYTE 0 (position 15, predicting AX_b1):")
print("-" * 70)

# Clear outputs
outputs = {}

# Run at byte 0 position
ctx_tensor2 = torch.tensor([ctx_byte0], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor2)
    pred = torch.argmax(logits[0, -1, :]).item()

for layer in ['l6', 'l8', 'l9', 'l10', 'l15']:
    if layer in outputs:
        out = outputs[layer][0, pos_byte0, :]
        output_lo = out[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
        output_hi = out[BD.OUTPUT_HI:BD.OUTPUT_HI+16]
        embed_lo = out[BD.EMBED_LO:BD.EMBED_LO+16]
        embed_hi = out[BD.EMBED_HI:BD.EMBED_HI+16]
        lo_val = torch.argmax(output_lo).item()
        hi_val = torch.argmax(output_hi).item()
        val = lo_val + 16 * hi_val
        embed_val = torch.argmax(embed_lo).item() + 16 * torch.argmax(embed_hi).item()
        print(f"  {layer:4s}: OUTPUT = {val:3d}, EMBED = {embed_val:3d}")

print()
print(f"Final prediction: {pred} (expected 0)")
print()
print("Analysis:")
print("  EMBED at byte 0 position contains token value (8), not register byte 1 (0)")
print("  L15 nibble copy uses EMBED, so it incorrectly copies 8 instead of 0")
