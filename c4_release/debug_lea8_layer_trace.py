"""Debug LEA 8 - trace OUTPUT_HI through all layers."""
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

ctx = context + draft_tokens[:6]
pos = len(ctx) - 1

print("LEA 8 - OUTPUT_HI Trace Through Layers")
print("=" * 70)
print()

# Capture after each layer
layer_outputs = {}

def make_hook(layer_idx, sublayer):
    def hook(module, input, output):
        key = f"L{layer_idx}_{sublayer}"
        if isinstance(output, tuple):
            layer_outputs[key] = output[0].detach().clone()
        else:
            layer_outputs[key] = output.detach().clone()
    return hook

# Register hooks
for i in range(16):
    model.blocks[i].attn.register_forward_hook(make_hook(i, "attn"))
    model.blocks[i].ffn.register_forward_hook(make_hook(i, "ffn"))

ctx_tensor = torch.tensor([ctx], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

print(f"Position {pos} (REG_AX marker):")
print()

# After embedding
x = model.embed(ctx_tensor)
output_hi = x[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
print(f"After Embedding: OUTPUT_HI[0]={output_hi[0].item():.1f}, mean={output_hi.mean().item():.1f}")

# Trace through layers
for i in range(16):
    if f"L{i}_attn" in layer_outputs:
        x = layer_outputs[f"L{i}_attn"]
        output_hi = x[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
        max_val = output_hi[0].item()
        mean_val = output_hi.mean().item()
        if abs(max_val) > 100 or abs(mean_val) > 50:
            print(f"L{i} Attn: OUTPUT_HI[0]={max_val:.1f}, mean={mean_val:.1f} ⚠")
        elif abs(max_val - mean_val) > 5:
            print(f"L{i} Attn: OUTPUT_HI[0]={max_val:.1f}, mean={mean_val:.1f}")

    if f"L{i}_ffn" in layer_outputs:
        x = layer_outputs[f"L{i}_ffn"]
        output_hi = x[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
        max_val = output_hi[0].item()
        mean_val = output_hi.mean().item()
        if abs(max_val) > 100 or abs(mean_val) > 50:
            print(f"L{i} FFN:  OUTPUT_HI[0]={max_val:.1f}, mean={mean_val:.1f} ⚠")
        elif abs(max_val - mean_val) > 5:
            print(f"L{i} FFN:  OUTPUT_HI[0]={max_val:.1f}, mean={mean_val:.1f}")

print()
predicted = logits[0, -1, :].argmax().item()
print(f"Final prediction: {predicted} (expected 8)")
