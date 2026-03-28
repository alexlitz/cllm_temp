"""Debug JMP 16 - trace OUTPUT through all layers."""
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

print("JMP 16 - OUTPUT Trace Through Layers")
print("=" * 70)
print()

bytecode = [Opcode.JMP | (16 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

# Test token 1 (PC_b0)
ctx = context + [draft_tokens[0]]

# Capture after each layer
layer_outputs = {}

def make_hook(layer_idx):
    def hook(module, input, output):
        layer_outputs[layer_idx] = output.detach().clone()
    return hook

# Register hooks for all 16 layers
for i in range(16):
    model.blocks[i].register_forward_hook(make_hook(i))

ctx_tensor = torch.tensor([ctx], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

pos = len(ctx) - 1
print(f"Position {pos} (last token = {ctx[-1]} = REG_PC)")
print(f"Expected next token: {draft_tokens[1]} (16)")
print()

print("OUTPUT trace through layers:")
print("-" * 70)

for layer_idx in range(16):
    if layer_idx in layer_outputs:
        out = layer_outputs[layer_idx]
        output_lo = out[0, pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
        output_hi = out[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
        output_lo_val = torch.argmax(output_lo).item()
        output_hi_val = torch.argmax(output_hi).item()
        output_byte = output_lo_val + (output_hi_val << 4)

        # Show which layers change OUTPUT
        marker = ""
        if layer_idx > 0:
            prev_out = layer_outputs[layer_idx - 1]
            prev_lo = prev_out[0, pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
            prev_hi = prev_out[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
            prev_lo_val = torch.argmax(prev_lo).item()
            prev_hi_val = torch.argmax(prev_hi).item()
            prev_byte = prev_lo_val + (prev_hi_val << 4)

            if output_byte != prev_byte:
                marker = f" ← CHANGED from {prev_byte}"

        print(f"L{layer_idx:2d}: OUTPUT = {output_byte:3d} (lo={output_lo_val}, hi={output_hi_val}){marker}")

predicted = torch.argmax(logits[0, -1]).item()
print()
print(f"Final prediction: {predicted}")
print(f"Expected: {draft_tokens[1]}")
print()

print("Analysis:")
print("  Looking for which layer sets OUTPUT incorrectly")
print("  Expected: OUTPUT should be 16 (lo=0, hi=1) at some layer")
print("  Actual: OUTPUT becomes 8, then 0")
