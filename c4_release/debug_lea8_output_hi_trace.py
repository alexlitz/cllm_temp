"""Debug LEA 8 - trace where OUTPUT_HI gets corrupted with huge values."""
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

print("LEA 8 - OUTPUT_HI Corruption Trace")
print("=" * 70)
print()

# Capture all layer outputs
layer_outputs = {}

for i in range(11):
    def make_hook(layer_idx):
        def hook(module, input, output):
            layer_outputs[layer_idx] = output.detach().clone()
        return hook
    model.blocks[i].register_forward_hook(make_hook(i))

ctx_tensor = torch.tensor([ctx], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

print(f"Position {pos} (REG_AX marker):")
print()

for i in range(11):
    if i in layer_outputs:
        out = layer_outputs[i][0, pos, :]
        output_hi = out[BD.OUTPUT_HI:BD.OUTPUT_HI+16]

        # Check if all dims are equal (corruption signature)
        hi_mean = output_hi.mean().item()
        hi_std = output_hi.std().item()
        hi_max = output_hi.max().item()
        hi_min = output_hi.min().item()

        argmax_val = torch.argmax(output_hi).item()

        if hi_std < 0.1:  # All dimensions nearly equal
            print(f"L{i:2d}: OUTPUT_HI CORRUPTED! All dims ≈ {hi_mean:.1f} (std={hi_std:.3f})")
        elif hi_max > 1000:
            print(f"L{i:2d}: OUTPUT_HI has HUGE values (max={hi_max:.1f}, argmax={argmax_val})")
        else:
            print(f"L{i:2d}: OUTPUT_HI OK (max={hi_max:.1f}, argmax={argmax_val})")

        # Highlight changes
        if i > 0:
            prev_out = layer_outputs[i-1][0, pos, :]
            prev_hi = prev_out[BD.OUTPUT_HI:BD.OUTPUT_HI+16]
            prev_max = prev_hi.max().item()

            if abs(hi_max - prev_max) > 100:
                print(f"      ↑ BIG CHANGE from {prev_max:.1f}")

print()
print("Analysis: Find where OUTPUT_HI first gets corrupted with huge equal values")
