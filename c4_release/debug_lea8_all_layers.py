"""Debug LEA 8 - trace OUTPUT through ALL layers at marker position."""
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

# Check at AX marker
ctx = context + draft_tokens[:6]
pos = len(ctx) - 1

print("LEA 8 - OUTPUT Trace Through All Layers (at AX marker)")
print("=" * 70)
print()

# Capture all layer outputs
layer_outputs = {}

for i in range(16):
    def make_hook(layer_idx):
        def hook(module, input, output):
            layer_outputs[layer_idx] = output.detach().clone()
        return hook
    model.blocks[i].register_forward_hook(make_hook(i))

ctx_tensor = torch.tensor([ctx], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)
    pred = torch.argmax(logits[0, -1, :]).item()

print(f"Position {pos} (REG_AX marker):")
print()

for i in range(16):
    if i in layer_outputs:
        out = layer_outputs[i][0, pos, :]
        output_lo = out[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
        output_hi = out[BD.OUTPUT_HI:BD.OUTPUT_HI+16]
        lo_val = torch.argmax(output_lo).item()
        hi_val = torch.argmax(output_hi).item()
        val = lo_val + 16 * hi_val

        # Highlight changes
        if i > 0:
            prev_out = layer_outputs[i-1][0, pos, :]
            prev_lo = prev_out[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
            prev_hi = prev_out[BD.OUTPUT_HI:BD.OUTPUT_HI+16]
            prev_val = torch.argmax(prev_lo).item() + 16 * torch.argmax(prev_hi).item()
            if val != prev_val:
                print(f"L{i:2d}: OUTPUT = {val:3d} (lo={lo_val:2d}, hi={hi_val:2d}) ← CHANGED from {prev_val}")
            else:
                print(f"L{i:2d}: OUTPUT = {val:3d} (lo={lo_val:2d}, hi={hi_val:2d})")
        else:
            print(f"L{i:2d}: OUTPUT = {val:3d} (lo={lo_val:2d}, hi={hi_val:2d})")

print()
print(f"Final prediction: {pred} (expected 8)")
print()
print("Analysis:")
print("  Looking for which layer changes OUTPUT from 8 to something else")
